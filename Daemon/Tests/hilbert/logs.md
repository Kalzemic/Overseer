no_overseer output:

[kalzemic@archlinux hilbert]$ make run
docker run --rm -v ~/hilbert_bench.dat:/data/hilbert_bench.dat hilbert_bench -n 11 /data/hilbert_bench.dat
N=2^11=2048 pages=4194304 size=16.00 GB file=/data/hilbert_bench.dat
Reusing existing file (16.00 GB).
Traversing 4194304 pages in Hilbert-curve order...
Traversal complete.
elapsed : 332.584 s
pages/s : 12611
throughput: 49.3 MB/s
checksum : 717225984 (non-zero confirms reads happened)

overseer output:

[kalzemic@archlinux Tests]$ ./overseer.sh
[sudo] password for kalzemic:
page cache dropped
Listening on port 9090
overseer_test
4c98a284376095f9480dd7b132bb0afbc9428ae07be8268a79cfd1e930349cd6
PID: 8935, Cgroup: system.slice/docker-4c98a284376095f9480dd7b132bb0afbc9428ae07be8268a79cfd1e930349cd6.scope, ID: 10806
N=2^11=2048 pages=4194304 size=16.00 GB file=/data/hilbert_bench.dat
Reusing existing file (16.00 GB).
Traversing 4194304 pages in Hilbert-curve order...
OverNet Running
fault collection initialized
Traversal complete.
elapsed : 338.936 s
pages/s : 12375
throughput: 48.3 MB/s
checksum : 717225984 (non-zero confirms reads happened)
=== OVERSEER major faults ===
cleaning up...
[sudo] password for kalzemic: Terminated ./OverNet
