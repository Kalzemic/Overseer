CC     = gcc
CFLAGS = -O2 -Wall -Wextra -march=native

build: matmul.c
	$(CC) $(CFLAGS) -o $@ $<
	python generate_matrices.py

clean:
	rm -f matmul 