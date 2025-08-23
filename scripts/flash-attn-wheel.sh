#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Clean up any artifacts from previous runs, just in case
if [ -d "flash-attention" ]; then
    cd flash-attention
    echo "--- Cleaning up previous build directories ---"
    rm -rf build dist
    cd ..
fi

# Update and install essential build tools
apt-get update
apt-get install -y ninja-build build-essential

# Upgrade Python packaging tools
pip install --upgrade pip setuptools wheel ninja packaging

# --- Environment Verification ---
echo "--- Verifying Build Environment ---"
nvcc --version
gcc --version
g++ --version
echo "---------------------------------"

# Clone Flash Attention v2.8.2 if it doesn't exist
if [ ! -d "flash-attention" ] ; then
    git clone https://github.com/Dao-AILab/flash-attention.git
fi
cd flash-attention
git checkout v2.8.2

# Set environment variables for a more portable wheel
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# --- CRITICAL CHANGE ---
# Reduce MAX_JOBS to a very safe level to prevent system thrashing.
export MAX_JOBS=8
export CUDA_HOME=/usr/local/cuda

echo "--- Building with a safe MAX_JOBS limit set to ${MAX_JOBS} ---"

# Build the wheel
echo "--- Starting Flash Attention wheel build ---"
python setup.py bdist_wheel

# Show the generated wheel
echo "--- Generated wheel file ---"
ls -la dist/*.whl

# Test the installation of the generated wheel
pip install dist/*.whl
python -c "import flash_attn; print('Successfully installed flash_attn version:', flash_attn.__version__)"

# Copy the wheel to the workspace for download
cp dist/*.whl /workspace/

# Show the final location of the wheel
echo "--- Wheel file copied to workspace ---"
ls -la /workspace/*.whl
