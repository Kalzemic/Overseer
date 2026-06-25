FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /bench

# Source only -- the backing data file is bind-mounted at runtime (see run
# instructions), not baked into the image or written to the container's
# writable layer, so disk I/O behaves the same as running on bare metal.
COPY hilbert_bench.c .

RUN gcc -O2 -o hilbert_bench hilbert_bench.c

ENTRYPOINT ["./hilbert_bench"]