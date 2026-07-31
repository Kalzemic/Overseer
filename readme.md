# Overseer

**A neural page prefetcher for Linux, built on eBPF telemetry and a transformer running on external hardware.**

Overseer observes a target application's page faults through eBPF, predicts its future memory access pattern with a transformer neural network, and prefetches the predicted pages into physical memory before the application faults on them. The observation, inference, and enforcement components run as three separate processes that communicate over UDP, so the neural network can live on a completely different physical machine from the application it accelerates.

On a Hilbert curve traversal benchmark designed specifically to defeat conventional stride based readahead, Overseer reduces wall clock time by 25 to 45 percent and eliminates roughly a third of the major faults that Linux readahead alone cannot catch.

![System Architecture](docs/architecture.png)
_Figure 1: end to end data flow through the three components._

---

## Motivation

The gap between CPU speed and memory access latency has been growing for decades, and the introduction of tiered memory systems (CXL attached DRAM, disaggregated memory pools, persistent memory) has made the gap explicit at page granularity rather than only at cache line granularity. When a page is not resident in fast memory, the cost of fetching it can be five or six orders of magnitude larger than the cost of accessing memory that is already there.

Linux's kernel readahead handles this cost well for sequential and stride detectable access patterns. It handles it poorly, or not at all, for workloads whose access pattern is structured but not stride-shaped: graph traversals, sparse matrix operations, tree walks with irregular branching, database index scans, ML embedding table lookups that overflow physical memory. These are exactly the workloads that dominate large scale scientific computing and modern data intensive applications.

Application level prefetching hints (`madvise(MADV_WILLNEED)`, software prefetch instructions) can close the gap when the developer knows the access pattern ahead of time and is willing to modify the application. In practice, most deployments cannot modify the application. Learned prefetchers offer a third path: observe the application's actual behavior, learn its patterns, prefetch on its behalf without requiring source access or recompilation.

Overseer is a systems architecture for deploying a learned prefetcher on stock Linux, without kernel patches, on live user processes identified by cgroup.

---

## Architecture

Overseer decomposes into three loosely coupled components connected by UDP:

**Collector.** A userspace daemon backed by an eBPF program attached to the `page_fault_user` tracepoint. For each fault raised by a process in a target cgroup, the eBPF program pushes `(instruction_pointer, faulting_address, pid, cgroup_id)` into a ring buffer. The userspace side drains the buffer, batches events per process, and ships batches of 64 to the inference host over UDP.

**OverNet.** A transformer neural network exported to TensorRT that runs on an NVIDIA GPU. For each batch of 64 fault events, it produces a prediction of the next 64 pages the process is likely to fault on. Inference happens in fewer than 500 microseconds per batch on a Jetson Orin, and in under 100 microseconds on a discrete host GPU.

**Enforcer.** A Linux kernel module that listens on UDP for prediction batches, dispatches them to per-process kernel worker threads, and calls `get_user_pages_remote` with `FOLL_FORCE` to fault the predicted pages into the target process's address space. Because the enforcer holds a reference to each target's `mm_struct`, it can act on the process's memory without the process itself running.

The three components exchange only fixed size packets over UDP, and none of them requires knowledge of the others beyond an IP address and a port. This means the inference tier can be relocated to any host on the network without changing the collector or enforcer.

![Component Interaction](docs/components.png)
_Figure 2: message flow and packet formats between the three components._

### Collector

The eBPF program is attached to the `exceptions:page_fault_user` tracepoint. On every user space page fault it captures four fields into a ring buffer entry: the faulting address, the instruction pointer that caused the fault, the process ID, and the cgroup ID. The userspace daemon polls the ring buffer, filters events to a single target cgroup, and maintains a per-PID buffer of 64 pending events. When a buffer fills, the daemon sends a `FaultReport` UDP packet (`pid` plus 64 addresses plus 64 IPs) to the inference host.

The choice of eBPF as the observation layer matters. It means Overseer can attach and detach without a kernel rebuild, filter by cgroup natively, and coexist with any Linux kernel that supports BPF tracepoints (roughly 5.4 and newer). No application changes are required, and the target process does not need to be aware that it is being observed.

### OverNet

OverNet is a transformer encoder-decoder trained to predict future page faults from past fault sequences.

**Input encoding.** Raw fault addresses are useless as neural network inputs because the address space is enormous and application specific. Instead, OverNet works on _deltas_ between successive pages, encoded through an adaptive lookup table that assigns compact ordinal ranks to whichever deltas have been seen. This turns a 48 bit address delta into a small integer in a bounded vocabulary (128 tokens in the current model). The same lookup is applied to instruction pointers.

**Model.** The encoder is a single layer transformer that ingests a sequence of 64 (IP rank, page delta rank) pairs. IP and page tokens are embedded separately, then combined via IP conditioned cross attention before entering the encoder stack. The decoder is autoregressive with a KV cache, producing 64 predicted page delta ranks token by token. Total parameter count is roughly 2.2 million.

**Deployment.** The trained model is exported to ONNX and compiled to a TensorRT engine with FP16 precision. The runtime maintains the same delta lookup table in CUDA that training used, computes deltas and ranks on device, invokes the engine, and decodes the predicted ranks back to absolute page addresses before shipping them to the enforcer.

**Weight tying.** The output projection shares weights with the input page embedding, which reduces the parameter count and improves generalization in low data regimes. This mirrors the standard tied embedding trick from modern language modeling.

![OverNet Architecture](docs/overnet.png)
_Figure 3: OverNet encoder decoder structure with IP conditioned cross attention._

### Enforcer

The enforcer is a loadable Linux kernel module with three components: a UDP socket in kernel space, a dispatcher kthread that drains incoming payloads, and a pool of per PID worker kthreads that perform the actual prefetching.

When a prediction batch arrives, the dispatcher looks up the corresponding worker (spawning one if the PID is new) and pushes the 64 predicted addresses into its per worker kfifo. The worker wakes, drains the fifo, and for each address calls `get_user_pages_remote(worker->mm, addr, 1, FOLL_FORCE, &page, NULL)`. This forces the page into physical memory (from disk, swap, or a slower memory tier) and returns without blocking the target process. When the application eventually faults on the predicted page, the page is already resident and the fault resolves as a minor fault instead of a major one.

The choice of `get_user_pages_remote` over `madvise(MADV_WILLNEED)` is deliberate: the enforcer is not the target process, so it cannot call `madvise` on the target's address space. `get_user_pages_remote` with a held `mm_struct` reference is the standard mechanism for one process to fault pages into another's memory.

---

## Data Pipeline

Overseer's model is trained on real fault traces collected from a synthetic benchmark designed to exercise the class of access patterns the model is meant to handle.

**Benchmark: Hilbert curve traversal.** A 16 GB file is memory mapped and traversed in Hilbert curve order across a 2048 by 2048 page grid. The Hilbert curve produces long runs of local access punctuated by large ordered jumps, resulting in a fault sequence that has clear structure but no single detectable stride. `MADV_NOHUGEPAGE` and `MADV_RANDOM` disable transparent huge pages and kernel readahead during data collection to isolate the model's learning signal.

**Trace collection.** `bpftrace` attaches to the same `page_fault_user` tracepoint the production collector uses and streams `(ip, addr)` pairs to disk. A Python conversion script packs the trace into a NumPy binary file of `uint64` pairs, ready for memory mapped loading during training.

**Training set.** Nine traces are collected under identical conditions, plus one held out for validation. Each trace contains roughly four million fault events, corresponding to one complete Hilbert traversal of the 16 GB file after the page cache is dropped.

**Data loader.** A PyTorch `IterableDataset` memmaps each trace, slides a window of length `2 * seq_len` across the sequence, and emits `(input, target)` pairs of length `seq_len = 64`. Delta computation and ordinal rank assignment happen on GPU during the training loop, matching exactly what the production runtime does at inference time.

---

## Training

The model is trained end to end with cross entropy loss on the ordinal rank vocabulary. Because the vocabulary is small (128 tokens for Hilbert) and the model is small, training converges quickly: 50 epochs at batch size 64 with the OneCycle learning rate schedule completes in under an hour on a single RTX 4070.

**Metrics.** Because Overseer emits 64 predictions per batch and the enforcer prefetches all 64, position accuracy is less informative than coverage: the fraction of true future faults that appear anywhere in the predicted window. On Hilbert, the trained model achieves greater than 99 percent coverage on held out validation traces.

![Training Curves](docs/training.png)
_Figure 4: training and validation coverage over 50 epochs._

---

## Results

All benchmarks run on Arch Linux with kernel 6.11, cgroup v2, an RTX 4070 (host GPU) and a Jetson Orin Nano (remote GPU), with the Hilbert traversal running inside a Docker container bind mounted to a 16 GB backing file on NVMe.

Three configurations are compared: **baseline** with no prefetcher beyond Linux's default readahead, **overseer (local)** with OverNet running on the host GPU, and **overseer (remote)** with OverNet running on the Jetson connected over USB-CDC ethernet.

![Wall Time Comparison](docs/walltime.png)
_Figure 5: box plots of wall clock time across 30 randomized runs per condition._

![Major Fault Reduction](docs/majflt.png)
_Figure 6: major fault counts, showing how much of the readahead residual Overseer catches._

**Key results:**

| Configuration             | Wall time (s) | Major faults | Successful prefetches |
| ------------------------- | ------------: | -----------: | --------------------: |
| Baseline (readahead only) |  27.8 to 38.1 |     ~123,000 |                   n/a |
| Overseer (local GPU)      |  21.2 to 24.0 |      ~85,000 |              ~366,000 |
| Overseer (remote Jetson)  |  22.6 to 28.8 |      ~91,000 |              ~365,000 |

Both Overseer configurations meaningfully outperform the readahead-only baseline. The local configuration is somewhat faster than the remote configuration, reflecting the additional latency of the USB-CDC round trip and the smaller compute budget on the Jetson, but the remote configuration still catches enough predictions in time to reduce wall time by roughly 25 percent.

Major fault count tells the causal story: Linux readahead alone catches about 97 percent of the fault opportunities on Hilbert, and Overseer removes another 32 percent of what remains. This is the fraction of the workload that stride based readahead structurally cannot handle.

---

## Related Work

Learned prefetchers are an active research area. Overseer occupies a specific niche in this space.

**Hashemi et al. (ICML 2018)** introduced neural memory access prediction using LSTMs with program counter and address delta features, applied to offline traces from a cache simulator. This work established the (PC, delta) feature choice that Overseer inherits.

**Voyager (Shi et al., ASPLOS 2021)** proposed a hierarchical LSTM that decouples page and offset prediction, improving accuracy on irregular workloads. Voyager remains a simulator based cache line prefetcher and is not intended for deployment on real hardware.

**FetchBPF (Cao et al., USENIX ATC 2024)** uses eBPF to install _rule based_ prefetching policies in the Linux kernel. FetchBPF demonstrates that eBPF is a viable enforcement mechanism for prefetching but does not incorporate any learned component.

Overseer's distinctive contribution is the combination of three properties that no prior system provides together: (1) eBPF based observation with cgroup filtering, requiring no kernel patch and no application modification; (2) disaggregated inference, allowing the neural network to run on completely separate hardware from the target process; (3) kernel module enforcement via `get_user_pages_remote`, giving the prefetcher direct access to the target's page tables without requiring the target to be prefetch aware.

---

## Limitations and Future Work

Overseer is a working prototype that demonstrates feasibility. Several directions are open.

**Multi workload generalization.** The model presented here is trained on Hilbert traces alone. Whether a single trained model can serve multiple distinct workloads is an open question. The ordinal rank vocabulary is designed to be workload adaptive (rank meanings reset per instance), which should help transfer, but this has not been validated experimentally.

**Enforcer throughput under stress.** The enforcer's `worker_run` loop calls `get_user_pages_remote` serially, one page at a time, holding the `mm` semaphore for each call. Under Hilbert this is sufficient, because the application's own fault rate is the throttle. Under a heavier or more parallel workload, the serial GUP path may become the bottleneck. Batched GUP or asynchronous readahead would address this.

**Tiered memory.** The current enforcer prefetches from disk or swap. Adapting it to CXL attached DRAM or other tiered memory systems requires substituting `get_user_pages_remote` with `move_pages` or a NUMA migration primitive, but leaves the rest of the pipeline unchanged. This is the natural next application of Overseer's architecture.

**Debug logging overhead.** `process_worker.c` emits a `pr_info` for every prefetched page, which floods the kernel log ringbuffer under load and causes lost events. Production deployment should gate these prints behind a compile time debug flag.

**TensorRT engine portability.** The compiled `.trt` engine is specific to the GPU architecture it was built on. Deploying to a fleet of heterogeneous machines requires rebuilding the engine per GPU family, which is standard for TensorRT deployments but worth noting.

---

## Building and Running

**Requirements:**

- Linux kernel 5.15 or newer with BPF and cgroup v2 enabled
- `linux-headers` matching the running kernel
- Docker for the benchmark container
- CUDA 12 and TensorRT 10 on the inference host
- A GPU compatible with TensorRT (tested on RTX 40 series and Jetson Orin)
- `sudo` access on the enforcer host

**Build the enforcer:**

```bash
cd Daemon/Enforcer
make
sudo insmod overseer_enforcer.ko
```

**Build the collector:**

```bash
cd Daemon/Collector
make
```

**Build OverNet:**

```bash
cd OverNet
mkdir build && cd build
cmake -DTENSORRT_DIR=/path/to/TensorRT ..    # host
# or: cmake ..                               # Jetson with system TensorRT
make
```

**Generate the TRT engine** (once per target GPU):

```bash
cd OverNet/build
python ../model/model.py
python ../model/engine.py
```

**Run the benchmark:**

```bash
cd Tests
./orchestrate_all.sh 30 results.csv         # 30 runs per condition
```

See `Tests/README.md` for details on the benchmark harness and result analysis.

---

## Repository Structure

```
Overseer/
├── Daemon/
│   ├── Collector/          # eBPF program and userspace collector
│   └── Enforcer/           # kernel module for prefetch enforcement
├── OverNet/
│   ├── training/           # PyTorch training code
│   ├── Data/hilbert/       # trace collection scripts
│   ├── Data/Matmul/        # alternative benchmark data
│   ├── model.py            # ONNX export
│   ├── engine.py           # TensorRT compilation
│   ├── main.cpp            # runtime inference server
│   ├── encode.cu           # CUDA delta encoding
│   ├── Lookup.cu           # CUDA ordinal rank table
│   └── Server.cpp          # UDP transport
├── Tests/
│   ├── run_baseline.sh
│   ├── run_overseer.sh          # local GPU
│   ├── run_overseer_remote.sh   # remote Jetson
│   └── orchestrate_all.sh       # randomized multi condition sweep
└── docs/                   # figures and supplementary material
```

---

## Author

This project is the capstone for [degree program], [university], [year]. Correspondence: [email].
