FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /bench
COPY matmul.c .
RUN gcc -O2 -o matmul -fopenmp matmul.c

ENTRYPOINT ["./matmul"]