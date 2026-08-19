# Setup

This document walks through building each component and (if desired) running Overseer end-to-end. Full end-to-end operation requires two hosts and an inference appliance with a GPU capable of running TensorRT.

## Prerequisites

**Host** (runs Collector and Enforcer):

- Linux with a kernel that supports eBPF (5.x or newer)
- `libbpf` and `bpftool` installed
- Kernel headers matching the running kernel (for the Enforcer)
- `clang`, `g++`, `make`, `libelf`, `zlib`
- `vmlinux.h` at the Collector build root (see below)

**Inference appliance** (runs OverNet in production):

- CUDA-capable GPU (Overseer's reference deployment uses a Jetson Orin Nano)
- CUDA toolkit
- TensorRT (not pip-installable in general; install per NVIDIA's documentation for your platform)
- Python 3 with PyTorch

**Training environment** (produces the model, can be any machine with a GPU):

- Python 3 with the dependencies listed in `OverNet/training/pyproject.toml`

### Generating vmlinux.h

The Collector's BPF program needs `vmlinux.h`, which is not shipped in the repository because it is kernel-specific. Generate it on the host:

```bash
bpftool btf dump file /sys/kernel/btf/vmlinux format c > Daemon/Collector/vmlinux.h
```

This requires that the running kernel was built with BTF enabled (`CONFIG_DEBUG_INFO_BTF=y`), which is the default on most modern distributions.

## Building the components

Each component builds from its own directory with a single `make` command.

### Collector

```bash
cd Daemon/Collector
make
```

Produces `overseer_collector`.

### Enforcer

```bash
cd Daemon/Enforcer
make
```

Produces `overseer_enforcer.ko`, an out-of-tree kernel module built against the running kernel's headers.

### OverNet (production)

The production OverNet code needs a serialized TensorRT engine file to run. Producing that engine requires training a model first, see the next section. Once a trained model is available:

```bash
cd Daemon/OverNet/model
python model.py    # converts OverNet.pt to ONNX
python engine.py   # converts ONNX to TensorRT (.trt)

cd ../build
make               # builds the OverNet inference daemon
```

Then move the `.trt` file into the `build/` directory.

## Training a model

OverNet is trained separately from deployment. The workflow is:

1. Generate benchmark trace data.
2. Train the model with PyTorch.
3. Transfer the resulting `.pt` file to the inference appliance for ONNX and TensorRT conversion.

### Preparing training data: Hilbert

```bash
cd OverNet/Data/hilbert
make hilbert_bench                           # compile the benchmark
./overpin_run.sh                             # produce .bin trace files
mkdir -p ../../training/data/hilbert         # create target dir if needed
mv *.bin ../../training/data/hilbert/
```

### Preparing training data: Matmul

```bash
cd OverNet/Data/matmul
python gen_matrices.py                       # generate matrices on disk
make matmul_bench                            # compile the benchmark
./record_run.sh                              # produce .bin trace files
mkdir -p ../../training/data/matmul          # create target dir if needed
mv *.bin ../../training/data/matmul/
```

### Configuring training

Open `OverNet/training/main.py` and ensure the workload constants `HILBERT_TRAIN`, `HILBERT_VAL`, `MATMUL_TRAIN`, `MATMUL_VAL` are uncommented, and that lines 69–70 pass them as arguments to the dataset constructors.

### Training

```bash
cd OverNet/training
python main.py
```

Produces `OverNet_Ordinal_<num_layers>.pt` (default `num_layers=1`, so `OverNet_Ordinal_1.pt`).

### Deploying the trained model

Transfer the `.pt` file to the inference appliance, rename it to `OverNet.pt`, and place it in `Daemon/OverNet/model/`. Then follow the OverNet build steps above (`model.py` → `engine.py` → `make`).

## Building the benchmarks

Both benchmark harnesses live under `Tests/`.

```bash
cd Tests/hilbert
make build             # builds as a Docker image
```

```bash
cd Tests/matmul
make build
```

## Running the benchmarks

Four orchestration scripts live in `Tests/`:

- `hilbert_baseline.sh` ->Hilbert without Overseer.
- `hilbert_overseer.sh` -> Hilbert with Overseer.
- `matmul_baseline.sh` -> Matmul without Overseer.
- `matmul_overseer.sh` -> Matmul with Overseer.

The Overseer scripts assume the full pipeline is up: Collector attached to the target cgroup and pointing at the appliance's IP and port, Enforcer kernel module loaded on the host, and the OverNet daemon running on the appliance and reachable over the network.

Results are logged to CSV. Run `Tests/metrics.py` on the resulting CSVs to produce the runtime and major-fault histograms shown in the README.

## Notes

- **Remote inference requires physical hardware.** Overseer's disaggregated design assumes the OverNet daemon runs on a separate GPU appliance (a Jetson Orin Nano in our reference deployment). Running the appliance side on the same host as the target is possible but not the intended operating mode.
- **TensorRT bindings are not fully pip-installable.** They must be installed following NVIDIA's platform-specific instructions before `engine.py` will work. This is an outstanding gap in the pip requirements and is on the list to address.
