cd_wheels — FlashAttention wheels (Ubuntu 22.04, CUDA 12.x, CPython 3.11)

### What is this?
- Prebuilt and reproducible build instructions for FlashAttention v2.8.2 wheels targeting Linux x86_64 (Ubuntu 22.04), CUDA 12.x, and Python 3.11.

### Included wheels
- `wheels/flash_attn-2.8.2-cp311-cp311-linux_x86_64_comfy.whl`


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
pip install wheels/flash_attn-2.8.2-cp311-cp311-linux_x86_64_comfy.whl
```
### Using with Comfy-Deploy
- Ensure your ComfyUI environment uses Python 3.11.
- Activate your environment, then install the wheel:
```bash
RUN pip install https://github.com/if-ai/cd_wheels/raw/main/wheels/flash_attn-2.8.2-cp311-cp311-linux_x86_64_comfy.whl
```



### Directory layout
- `scripts/`: build scripts
- `wheels/`: prebuilt wheels ready to install

### Licensing
FlashAttention is developed by Dao-AILab. Refer to the upstream repository for license details and notices.
