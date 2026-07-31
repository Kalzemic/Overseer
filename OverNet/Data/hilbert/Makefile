CC     = gcc
CFLAGS = -O2 -Wall -Wextra -march=native

hilbert_bench: hilbert_bench.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f hilbert_bench trace_faults.txt
