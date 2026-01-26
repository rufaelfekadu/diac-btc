# Automatic Diacritization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rufaelfekadu/diac-btc/blob/main/notebooks/test.ipynb)

## Getting started

### Installation

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the package:

   ```bash
   pip install -e .
   ```

3. Install k2 (required for CTC decoding):

   ```bash
   # For CUDA 12.8 with PyTorch 2.9.1
   pip install k2==1.24.4.dev20251118+cuda12.8.torch2.9.1 -f https://k2-fsa.github.io/k2/cuda.html

   # For other CUDA/PyTorch versions, check: https://k2-fsa.github.io/k2/installation/index.html
   ```

**Note:** k2 requires CUDA and must match your PyTorch and CUDA versions. Visit the [k2 installation guide](https://k2-fsa.github.io/k2/installation/index.html) to find the correct version for your setup.
