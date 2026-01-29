# Installation Guide

This guide provides detailed installation instructions for ConsciousnessX.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Standard Installation](#standard-installation)
- [Development Installation](#development-installation)
- [Hardware-Specific Installation](#hardware-specific-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Python Version

ConsciousnessX requires Python 3.9 or higher. Check your Python version:

```bash
python --version
# or
python3 --version
```

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB free space

**Recommended Requirements:**
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA GPU with CUDA 11.0+ or AMD GPU with ROCm 5.0+
- Storage: 20+ GB free space

### Optional Dependencies

For GPU acceleration:
- CUDA Toolkit 11.0+ (NVIDIA)
- ROCm 5.0+ (AMD)

For distributed computing:
- MPI library (OpenMPI or MPICH)

## Standard Installation

### Using pip

The simplest way to install ConsciousnessX:

```bash
pip install consciousnessx
```

This installs the base package with all core dependencies.

### Installing with Optional Dependencies

Install with development tools:

```bash
pip install consciousnessx[dev]
```

Install with testing tools:

```bash
pip install consciousnessx[test]
```

Install with documentation tools:

```bash
pip install consciousnessx[docs]
```

Install with HPC support:

```bash
pip install consciousnessx[hpc]
```

Install with all optional dependencies:

```bash
pip install consciousnessx[dev,test,docs,hpc]
```

## Development Installation

### Clone the Repository

```bash
git clone https://github.com/Napiersnotes/consciousnessX.git
cd consciousnessX
```

### Create Virtual Environment

**Using venv:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**

```bash
conda create -n consciousnessx python=3.10
conda activate consciousnessx
```

### Install in Development Mode

```bash
pip install -e ".[dev,test]"
```

This installs the package in editable mode, allowing you to make changes without reinstalling.

### Install Pre-commit Hooks (Optional)

```bash
pre-commit install
```

This will automatically run linting and formatting on your commits.

## Hardware-Specific Installation

### NVIDIA GPU Support

1. Install CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

2. Install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Verify GPU installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### AMD GPU Support

1. Install ROCm from [AMD's website](https://rocm.docs.amd.com/)

2. Install PyTorch with ROCm support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

3. Verify GPU installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Cray LUX Supercomputer

1. Load required modules:

```bash
module load craype
module load cray-libsci
module load cudatoolkit
module load cray-mpich
```

2. Install with HPC support:

```bash
pip install consciousnessx[hpc]
```

3. Use the Cray-specific configuration:

```bash
cx-train --config configs/hpc/cray_lux.yml
```

### AMD MI355X Accelerator

1. Install ROCm and MIOpen:

```bash
# Ubuntu/Debian
sudo apt-get install rocm-dev miopen-hip

# RHEL/CentOS
sudo yum install rocm-dev miopen-hip
```

2. Set environment variables:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache
```

3. Install with HPC support:

```bash
pip install consciousnessx[hpc]
```

4. Use the AMD-specific configuration:

```bash
cx-train --config configs/hpc/amd_mi355x.yml
```

## Verification

After installation, verify that ConsciousnessX is working correctly:

### Check Installation

```bash
python -c "import src; print(src.__version__)"
```

### Test Core Modules

```bash
python -c "from src.core import MicrotubuleSimulator"
python -c "from src.training import ConsciousnessTrainer"
python -c "from src.evaluation import ConsciousnessAssessment"
```

### Test CLI Tools

```bash
consciousnessx --help
cx-train --help
cx-simulate --help
cx-assess --help
cx-visualize --help
```

### Run Test Suite

```bash
pytest tests/ -v
```

### Run Basic Simulation

```python
from src.core import MicrotubuleSimulator

simulator = MicrotubuleSimulator(num_tubulins=100)
simulator.initialize_quantum_state()
print("Installation successful!")
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Ensure you're in the correct directory and the package is installed:

```bash
cd consciousnessX
pip install -e .
```

### GPU Not Detected

**Problem:** `torch.cuda.is_available()` returns False

**Solution:** 
1. Verify CUDA/ROCm installation
2. Check GPU drivers: `nvidia-smi` or `rocm-smi`
3. Reinstall PyTorch with GPU support

### Quantum Backend Errors

**Problem:** Errors related to Qiskit or PennyLane

**Solution:** Install quantum backends:

```bash
pip install qiskit[aer]
pip install pennylane
```

### Memory Errors

**Problem:** Out of memory errors during simulation

**Solution:** Reduce batch size or use CPU instead of GPU:

```python
simulator = MicrotubuleSimulator(num_tubulins=50)  # Reduce from 100
# Or use CPU in config
```

### MPI Errors (HPC)

**Problem:** MPI-related errors on HPC systems

**Solution:** Ensure MPI is properly configured:

```bash
module load openmpi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mpi/lib
```

### Permission Errors

**Problem:** Permission denied when installing

**Solution:** Use user installation:

```bash
pip install --user consciousnessx
```

Or use a virtual environment (recommended).

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/Napiersnotes/consciousnessX/issues)
2. Search existing discussions
3. Create a new issue with:
   - Python version
   - OS and hardware details
   - Complete error traceback
   - Steps to reproduce

## Next Steps

After successful installation:

- Read the [Usage Guide](usage.md)
- Check out [Examples](../examples/)
- Explore the [API Reference](api/)