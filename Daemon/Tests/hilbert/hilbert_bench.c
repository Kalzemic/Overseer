#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PAGE 4096UL

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/*
 * Convert Hilbert-curve index d to (x,y) in an n×n grid.
 * n must be a power of 2. Standard bit-rotation/reflection algorithm.
 *
 * Why Hilbert over Z-order: Z-order has ~50% of transitions as stride ±1
 * (x-direction), which single-stride readahead could handle. Hilbert
 * distributes transitions evenly: ≈25% each for +1/-1/+N/-N, so no
 * single stride exceeds 25-33% even for small n.
 */
static void hilbert_d2xy(uint64_t n, uint64_t d, uint64_t *x, uint64_t *y)
{
    uint64_t rx, ry, s, t = d;
    *x = *y = 0;
    for (s = 1; s < n; s <<= 1) {
        rx = 1ULL & (t >> 1);
        ry = 1ULL & (t ^ rx);
        if (ry == 0) {
            if (rx == 1) {
                *x = s - 1 - *x;
                *y = s - 1 - *y;
            }
            uint64_t tmp = *x; *x = *y; *y = tmp;
        }
        *x += s * rx;
        *y += s * ry;
        t >>= 2;
    }
}

static void create_file(const char *path, uint64_t nbytes)
{
    printf("Creating %s (%.2f GB) ...\n", path, (double)nbytes / (1UL << 30));
    fflush(stdout);

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open for create"); exit(1); }

    /*
     * Write non-zero content in large chunks. Non-zero prevents the FS
     * from optimizing this into a sparse/zero-page file, which would
     * give zero-fill minor faults instead of real disk reads during traversal.
     */
    const size_t CHUNK = 64UL << 20; /* 64 MB */
    uint8_t *buf = malloc(CHUNK);
    if (!buf) { perror("malloc"); exit(1); }
    memset(buf, 0xAB, CHUNK);

    uint64_t done = 0;
    while (done < nbytes) {
        size_t n = nbytes - done < CHUNK ? (size_t)(nbytes - done) : CHUNK;
        ssize_t r = write(fd, buf, n);
        if (r <= 0) { perror("write"); exit(1); }
        done += (uint64_t)r;
        printf("\r  %.1f%%  (%.2f / %.2f GB)",
               100.0 * done / nbytes,
               (double)done / (1UL << 30),
               (double)nbytes / (1UL << 30));
        fflush(stdout);
    }
    printf("\n");
    free(buf);
    fsync(fd);
    close(fd);
    printf("File created and synced to disk.\n");
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [-n log_n] [file]\n"
        "  -n log_n   grid side = 2^log_n  (default 11 → N=2048, 16 GB)\n"
        "  file       path to backing file  (default /tmp/hilbert_bench.dat)\n"
        "\n"
        "log_n=9  → N=512,  1 GB  (quick verification; needs drop_caches for cold faults)\n"
        "log_n=10 → N=1024, 4 GB  (needs drop_caches for cold faults)\n"
        "log_n=11 → N=2048, 16 GB (> 15 GB RAM → continuous major faults w/o drop_caches)\n"
        "NOTE: /tmp is tmpfs on this system. Use a real-disk path (e.g. /home/kalzemic/).\n",
        prog);
}

int main(int argc, char **argv)
{
    uint64_t log_n = 11;
    const char *path = "/home/kalzemic/hilbert_bench.dat"; /* /tmp is tmpfs — use real disk */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc)
            log_n = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]); return 0;
        } else if (argv[i][0] != '-')
            path = argv[i];
    }

    if (log_n < 1 || log_n > 15) {
        fprintf(stderr, "log_n must be 1..15\n"); return 1;
    }

    uint64_t N           = 1ULL << log_n;
    uint64_t total_pages = N * N;
    uint64_t total_bytes = total_pages * PAGE;

    printf("N=2^%lu=%lu  pages=%lu  size=%.2f GB  file=%s\n",
           log_n, N, total_pages, (double)total_bytes / (1UL << 30), path);

    /* Create or reuse the backing file */
    struct stat st;
    if (stat(path, &st) != 0 || (uint64_t)st.st_size != total_bytes)
        create_file(path, total_bytes);
    else
        printf("Reusing existing file (%.2f GB).\n", (double)st.st_size / (1UL << 30));

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open for mmap"); exit(1); }

    uint8_t *base = mmap(NULL, total_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);

    /*
     * MADV_NOHUGEPAGE: force 4 KB page granularity even if THP=always.
     * MADV_RANDOM: disable kernel read-ahead (each fault is independent).
     * Both are needed to get clean per-page fault events.
     */
    madvise(base, total_bytes, MADV_NOHUGEPAGE);
    madvise(base, total_bytes, MADV_RANDOM);

    printf("Traversing %lu pages in Hilbert-curve order...\n", total_pages);
    fflush(stdout);

    volatile uint64_t checksum = 0;
    uint64_t t0 = now_ns();

    for (uint64_t d = 0; d < total_pages; d++) {
        uint64_t x, y;
        hilbert_d2xy(N, d, &x, &y);
        /* Touch first byte of page (y*N + x) in row-major layout */
        checksum += base[(y * N + x) * PAGE];
    }

    uint64_t t1 = now_ns();
    double sec = (double)(t1 - t0) * 1e-9;

    printf("Traversal complete.\n");
    printf("  elapsed   : %.3f s\n", sec);
    printf("  pages/s   : %.0f\n", (double)total_pages / sec);
    printf("  throughput: %.1f MB/s\n",
           (double)total_pages * PAGE / sec / (1UL << 20));
    printf("  checksum  : %lu  (non-zero confirms reads happened)\n", checksum);

    munmap(base, total_bytes);
    return 0;
}
