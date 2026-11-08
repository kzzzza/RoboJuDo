#!/usr/bin/env bash
# Optional TensorRT dependencies stay in the project's visual environment.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_path="${1:-.venv-video/bin/python}"
"$python_path" -c 'import pathlib, sys; root = pathlib.Path.cwd(); assert sys.prefix != sys.base_prefix and pathlib.Path(sys.prefix).resolve().is_relative_to(root), "Use a virtual environment inside this checkout"'
mkdir -p .local/cache/pip .local/setup
export PIP_CACHE_DIR="$project_root/.local/cache/pip"
export TMPDIR="$project_root/.local/setup"
"$python_path" -m pip install --no-input 'tensorrt-cu12==10.13.3.9' 'onnx==1.19.1' \
  'ml_dtypes==0.5.3' 'numpy==1.26.4'
"$python_path" -c 'import tensorrt, onnx; print("TensorRT", tensorrt.__version__, "ONNX", onnx.__version__)'
