# Environment Setup

How to create the project environment and prepare the local data. Running the research steps that follow is described in [running the pipeline](running-the-pipeline.md).

The environment is pinned for reproducibility: every direct dependency has an exact version in [`pyproject.toml`](../pyproject.toml), and the Python dependencies, including indirect ones, are locked with artifact SHA-256 hashes (Secure Hash Algorithm, 256-bit) in [`uv.lock`](../uv.lock). Isolated build dependencies are version-constrained in `pyproject.toml`; the system prerequisites and the native build toolchain are recorded separately (step 2). The environment is managed with [uv](https://docs.astral.sh/uv/), a Python package and project manager.

**Tested platform:** Linux x86-64 (Ubuntu 24.04 on Windows Subsystem for Linux 2, WSL2), Python 3.12.3 (pinned in [`.python-version`](../.python-version)), uv 0.13.0, and an NVIDIA graphics processing unit (GPU) with a driver that supports CUDA 12.8 (NVIDIA's GPU computing platform; check with `nvidia-smi`). The lock covers this platform only.

Terminal commands that produce research output are run by the project owner ([execution rules](../AGENTS.md#experiment-execution-responsibility)).

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Install the system prerequisites of LightGBM's CUDA build

[DR-013](decisions/DR-013-training-device-policy.md) §2 records the build. LightGBM is built from source for the project GPU (compute capability sm_120), because the standard wheel has no CUDA support. The recorded build is kept, and LightGBM trains with its CPU learner (DR-013 §1–§2).

- The CUDA Toolkit 12.8 from NVIDIA's WSL-Ubuntu repository, with the compiler at `/usr/local/cuda-12.8/bin/nvcc` (tested: 12.8.93), and GCC, the GNU Compiler Collection (tested: 13.3.0).
- NCCL (the NVIDIA Collective Communications Library) 2.26.2 for CUDA 12.8. The WSL-Ubuntu repository has no NCCL packages, so the two packages are taken from NVIDIA's Ubuntu 24.04 repository, installed at this exact version and held, so that an upgrade cannot pull in another CUDA version:

  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnccl2_2.26.2-1+cuda12.8_amd64.deb
  ```

  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnccl-dev_2.26.2-1+cuda12.8_amd64.deb
  ```

  ```bash
  sudo apt-get install -y ./libnccl2_2.26.2-1+cuda12.8_amd64.deb ./libnccl-dev_2.26.2-1+cuda12.8_amd64.deb
  ```

  ```bash
  sudo apt-mark hold libnccl2 libnccl-dev
  ```

  SHA-256 of the tested packages: `libnccl2` `91a95bb422b54fa6603f481ab72e6b2aa2bb805b9210053efc2dac91132a5274`; `libnccl-dev` `45824393a1e70d3407537e6f168186f312fd3d588edabd836d01fef4198e1c0c`.

The build settings (CUDA on, target architecture 120, compiler path) and the versions of the build tools (scikit-build-core, CMake, Ninja) are fixed in [`pyproject.toml`](../pyproject.toml); CMake and Ninja are required at those versions even when other versions are installed on the system, and a build is accepted only under the acceptance rule of DR-013 §2 (Ninja generator, matching tool versions, no conflicting overrides). The CUDA runtime and NCCL are linked statically into LightGBM.

## 3. Create the environment

Create the environment exactly as locked, with the tree-based (`ml`), neural (`dl`, PyTorch for CUDA 12.8) and development (`dev`) groups. This step builds LightGBM from source:

```bash
uv sync --locked --extra ml --extra dl --extra dev
```

## 4. Obtain the raw data

Accept the Competition Rules on Kaggle and place both files in `data/raw/` ([dataset documentation](dataset.md)): `train_transaction.csv` and `train_identity.csv`. The data are never committed.

## 5. Validate the raw data and prepare the development files, once

```bash
.venv/bin/review-capacity --log-file logs/prepare-data.log prepare-data
```

After this one-time build, every step reads only the verified processed files in `data/processed/`, and the raw files are read again only by the confirmatory Test evaluation ([data access rule](dataset.md#data-access-after-the-parquet-build)).

The provenance records of that build are re-recorded with the feature-declaration fingerprint, once and without rewriting any data file, by `migrate-provenance`; `--check-only` verifies without writing:

```bash
.venv/bin/review-capacity --log-file logs/migrate-provenance.log migrate-provenance
```

Every command logs its progress with timestamps to the terminal; `--log-file PATH` (before the command name) also keeps the log in a file, as an evidence-producing run should. `review-capacity --help` lists the commands, the options and the exit codes.
