#!/usr/bin/env bash
# Install Microsoft's amd64 runtime into this checkout without changing apt/Conda.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
[[ "$(uname -m)" == x86_64 ]] || { echo 'This installer requires x86_64'; exit 1; }
mkdir -p .local/k4a/downloads .local/k4a/runtime
base_url=https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk
for package in libk4a1.4 libk4a1.4-dev; do
  package_file=".local/k4a/downloads/${package}_1.4.2_amd64.deb"
  if [[ ! -f "$package_file" ]]; then
    curl -fL --retry 2 --connect-timeout 20 --max-time 600 \
      "$base_url/$package/${package}_1.4.2_amd64.deb" -o "$package_file.part"
    mv "$package_file.part" "$package_file"
  fi
done
sha256sum --check <<'CHECKSUMS'
1f090b65f696f78739507d0f55f3280d5df6330401b825616ef9eca0125ba8b7  .local/k4a/downloads/libk4a1.4_1.4.2_amd64.deb
f7e694eae34c4b8216f249f0bf23eb0f3fd9ea00538d41b214c9e8ac2da2b3bc  .local/k4a/downloads/libk4a1.4-dev_1.4.2_amd64.deb
CHECKSUMS
for package_file in .local/k4a/downloads/*_1.4.2_amd64.deb; do
  dpkg-deb -x "$package_file" .local/k4a/runtime
done
rule_file=.local/k4a/99-k4a.rules
rule_sha=be647ffb970170364c03ab093e336ed4a37d25e6218a25c7f269ac66ca108def
if [[ ! -f "$rule_file" ]] || [[ "$(sha256sum "$rule_file" | cut -d ' ' -f 1)" != "$rule_sha" ]]; then
  curl -fsSL --retry 2 --retry-all-errors --connect-timeout 20 --max-time 60 \
    https://raw.githubusercontent.com/microsoft/Azure-Kinect-Sensor-SDK/v1.4.2/scripts/99-k4a.rules \
    -o "$rule_file.part"
  mv "$rule_file.part" "$rule_file"
fi
sha256sum --check <<'CHECKSUMS'
be647ffb970170364c03ab093e336ed4a37d25e6218a25c7f269ac66ca108def  .local/k4a/99-k4a.rules
CHECKSUMS
echo 'libk4a 1.4.2 installed under .local/k4a/runtime (includes depth engine).'
echo 'If USB access fails, install the official rule with administrator privileges:'
echo "sudo install -m 0644 '$project_root/.local/k4a/99-k4a.rules' /etc/udev/rules.d/99-k4a.rules"
echo 'sudo udevadm control --reload-rules'
echo 'Then unplug/replug the Azure Kinect USB cable.'
