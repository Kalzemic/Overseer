CC     = gcc
CFLAGS = -O2 -Wall -Wextra -march=native

matmul_bench: matmul_bench.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f matmul_bench trace_faults.txt