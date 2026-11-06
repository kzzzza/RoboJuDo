#!/usr/bin/env bash
# All sources, libraries, caches and Python installs stay in this checkout.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_path="${1:-.venv/bin/python}"
sdk_commit=814556d15970dd2ecf1c9984e845ca02ab07e206
dds_commit=9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd
"$python_path" -c 'import sys; assert sys.prefix != sys.base_prefix, "Use a project virtual environment"'
mkdir -p .local/cache/pip
if [[ ! -d .local/unitree_sdk2_python ]]; then
  git clone https://github.com/unitreerobotics/unitree_sdk2_python.git .local/unitree_sdk2_python
  git -C .local/unitree_sdk2_python checkout --detach "$sdk_commit"
fi
[[ "$(git -C .local/unitree_sdk2_python rev-parse HEAD)" == "$sdk_commit" ]]
if [[ ! -d .local/cyclonedds-src ]]; then
  git clone --depth 1 --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git .local/cyclonedds-src
fi
[[ "$(git -C .local/cyclonedds-src rev-parse HEAD)" == "$dds_commit" ]]
cmake -S .local/cyclonedds-src -B .local/cyclonedds-build \
  -DCMAKE_INSTALL_PREFIX="$project_root/.local/cyclonedds" -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build .local/cyclonedds-build -j4
cmake --install .local/cyclonedds-build
export CYCLONEDDS_HOME="$project_root/.local/cyclonedds"
"$python_path" -m pip install --cache-dir .local/cache/pip 'cyclonedds==0.10.2' 'opencv-python==4.11.0.86'
"$python_path" -m pip install --cache-dir .local/cache/pip --no-build-isolation --no-deps -e .local/unitree_sdk2_python
"$python_path" -c 'from unitree_sdk2py.utils.crc import CRC; from unitree_sdk2py.core.channel import ChannelSubscriber; CRC(); print("Official G1 DDS SDK ready")'
