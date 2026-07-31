#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <omp.h>

#ifndef N
#define N 3000
#endif

#define MATRIX_BYTES ((size_t)N * (size_t)N * sizeof(double))

static double *map_ro(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); exit(1); }
    void *p = mmap(NULL, MATRIX_BYTES, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap ro"); exit(1); }
    close(fd);
    
    return (double *)p;
}

static double *map_rw(const char *path) {
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); exit(1); }
    if (ftruncate(fd, MATRIX_BYTES) < 0) { perror("ftruncate"); exit(1); }
    void *p = mmap(NULL, MATRIX_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap rw"); exit(1); }
    close(fd);
    
    return (double *)p;
}

int main(void)
{
    double (*a)[N] = (double (*)[N]) map_ro("data/a.bin");
    double (*b)[N] = (double (*)[N]) map_ro("data/b.bin");
    double (*c)[N] = (double (*)[N]) map_rw("data/c.bin");

    int i, j, k;
    double t1, t2;

    t1 = omp_get_wtime();

// Multiply matrices
#pragma omp parallel for shared(a, b, c) private(i, j, k)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            c[i][j] = 0.0;
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];

            // substitude by this line to reduce the cache miss
            // c[i][j] += a[i][k]*b[j][k];
            //
        }
    }

    t2 = omp_get_wtime();
    printf("Time = %f\n", t2 - t1);

    munmap(a, MATRIX_BYTES);
    munmap(b, MATRIX_BYTES);
    munmap(c, MATRIX_BYTES);
    return 0;
}