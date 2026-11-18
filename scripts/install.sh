#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mmstta}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This installer requires Linux x86-64.' >&2
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo 'Conda is required. Install Miniconda or Anaconda, then rerun this script.' >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# Reuse the environment when the installer is rerun.
if ! conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    conda create -n "$ENV_NAME" python=3.9.23 -y
fi
conda activate "$ENV_NAME"

# Match the Python/MKL versions and pip-installed packages in BoostAdapter.
conda install -y python=3.9.23 mkl=2025.0.0

python -m pip install \
    torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
python -m pip check

cd "$PROJECT_ROOT"
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

import torch
import torchvision
import faiss
import faiss.contrib.torch_utils
import ot
import wandb
import loguru
import IPython
import clip
import utils

if sys.version_info[:3] != (3, 9, 23):
    raise RuntimeError(f'Expected Python 3.9.23, got {sys.version.split()[0]}')
expected = {
    'torch': '2.2.0+cu118',
    'torchvision': '0.17.0+cu118',
    'torchaudio': '2.2.0+cu118',
}
for line in Path('requirements.txt').read_text().splitlines():
    if line and not line.startswith('#'):
        name, pinned = line.split('==')
        expected[name] = pinned
for name, pinned in expected.items():
    installed = version(name)
    if installed != pinned:
        raise RuntimeError(f'{name}: expected {pinned}, got {installed}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable. Check the NVIDIA driver and GPU visibility.')
resources = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatL2(32))
features = torch.randn(4, 32, device='cuda')
index.add(features)
index.search(features, 1)
print('Dependency and GPU FAISS checks passed.')
PY

printf '\nInstallation complete. In your terminal, run:\n  conda activate %s\nThen follow the README to prepare datasets and run evaluation.\n' "$ENV_NAME"
