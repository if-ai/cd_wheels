cd_wheels — FlashAttention wheels (Ubuntu 22.04, CUDA 12.x, CPython 3.11)

### What is this?
- Prebuilt and reproducible build instructions for FlashAttention v2.8.2 wheels targeting Linux x86_64 (Ubuntu 22.04), CUDA 12.x, and Python 3.11.

### Included wheels
- `wheels/flash_attn-2.8.2-cp311-cp311-linux_x86_64.whl`

### Quick install
If you already have a Python 3.11 environment on Linux with a CUDA-enabled PyTorch that matches your system CUDA version:

```bash
# Optional (for ComfyUI users)
micromamba activate comfy

pip install wheels/flash_attn-2.8.2-cp311-cp311-linux_x86_64.whl
```

### Requirements and compatibility
- OS: Ubuntu 22.04 (glibc compatible Linux x86_64)
- Python: 3.11 (cp311)
- GPU architectures baked in: `sm80 sm86 sm89 sm90` (e.g., A100, RTX 30/40 series, H100)
- CUDA: 12.x runtime (match your PyTorch CUDA build)
- PyTorch: CUDA-enabled build installed before installing the wheel

Check your PyTorch/CUDA pairing:
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
```

### Verify installation
```bash
python -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'flash_attn', flash_attn.__version__)"
```

### Using with ComfyUI
- Ensure your ComfyUI environment uses Python 3.11.
- Activate your environment, then install the wheel:
```bash
micromamba activate comfy
pip install wheels/flash_attn-2.8.2-cp311-cp311-linux_x86_64.whl
```

### Build from source (reproducible)
This repo includes a script to build the wheel on Ubuntu 22.04 with CUDA available at `/usr/local/cuda` and NVCC on `PATH`.

Option A — run the provided script:
```bash
bash scripts/flash-attn-wheel.sh
```
Notes:
- The script checks out `flash-attention` v2.8.2, sets `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"`, and limits `MAX_JOBS=8` for stable builds.
- It installs build deps (`ninja-build`, `build-essential`) and Python packaging tools.
- By default it copies the built wheel to `/workspace/`. Adjust that path as needed.

Option B — manual steps:
```bash
sudo apt-get update && sudo apt-get install -y ninja-build build-essential
pip install --upgrade pip setuptools wheel ninja packaging
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.2
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
export MAX_JOBS=8
python setup.py bdist_wheel
ls dist/*.whl
```

### Troubleshooting
- Mismatched CUDA: Ensure `torch.version.cuda` matches the CUDA runtime you intend to use.
- Unsupported GPU: Cards older than `sm80` (Ampere) are not included; rebuild with the right `TORCH_CUDA_ARCH_LIST` if needed.
- OOM/low RAM during build: Lower `MAX_JOBS` (e.g., `export MAX_JOBS=4`).
- Different OS/GLIBC: These wheels are Linux-only; build from source on your target system.

### Directory layout
- `scripts/`: build scripts
- `wheels/`: prebuilt wheels ready to install

### Licensing
FlashAttention is developed by Dao-AILab. Refer to the upstream repository for license details and notices.
