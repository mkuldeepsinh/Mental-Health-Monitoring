#!/usr/bin/env bash
set -euo pipefail

echo "==== Setup: Conda env 'mh' for TensorFlow (Python 3.10) ===="

# Check conda
if ! command -v conda &> /dev/null; then
  echo "ERROR: conda not found. Install Miniconda or Anaconda first: https://docs.conda.io"
  exit 1
fi

# Ensure conda shell functions available in script
CONDA_BASE=$(conda info --base)
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_NAME="mh"
PY_VER="3.10"

# Create env if missing
if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "Conda env '${ENV_NAME}' already exists — activating it."
else
  echo "Creating conda env '${ENV_NAME}' with python=${PY_VER}..."
  conda create -n "${ENV_NAME}" python="${PY_VER}" -y
fi

conda activate "${ENV_NAME}"

echo "Upgrading pip/tools..."
pip install --upgrade pip setuptools wheel

echo "Pinning NumPy to 1.26.4 (important)..."
pip install --upgrade "numpy==1.26.4"

echo "Installing pandas and scikit-learn..."
pip install pandas scikit-learn

# detect architecture
ARCH=$(uname -m)
echo "System architecture: ${ARCH}"

if [ "${ARCH}" = "arm64" ]; then
  echo "Detected Apple Silicon (arm64). Installing Apple TensorFlow dependencies..."
  # apple channel provides system deps for tensorflow-macos
  conda install -c apple tensorflow-deps -y

  echo "Installing tensorflow-macos and tensorflow-metal via pip..."
  # NOTE: not pinning exact TF version here—if you prefer a pinned version change below
  pip install tensorflow-macos tensorflow-metal
else
  echo "Detected non-arm64 (likely Intel). Installing TensorFlow from conda-forge..."
  # conda-forge TF tends to work well on Intel macs / Linux
  conda install -c conda-forge tensorflow -y
fi

# Optional plotting lib
pip install matplotlib

echo ""
echo "==== Verification step: importing numpy and tensorflow ===="
python - <<PY
import sys
import numpy as np
print("numpy:", np.__version__)
try:
    import tensorflow as tf
    print("tensorflow:", tf.__version__)
    try:
        devices = tf.config.list_physical_devices()
        print("tf devices:", devices)
    except Exception as e:
        print("Error when listing devices:", e)
except Exception as e:
    print("ERROR importing TensorFlow:", e)
    sys.exit(2)
PY

echo ""
echo "Setup finished. To use the env:"
echo "  conda activate ${ENV_NAME}"
echo "  python main.py"
echo ""
echo "If tensorflow import failed, consider:"
echo " - Ensuring the script ran under Python 3.10 (we created env with 3.10)"
echo " - Removing any old conflicting tensorflow/numpy installations in other envs"
echo " - Share errors with me and I will help."
