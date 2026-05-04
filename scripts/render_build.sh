#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p tools

if [ ! -d tools/blackbox-tools ]; then
  git clone --depth 1 https://github.com/betaflight/blackbox-tools.git tools/blackbox-tools
fi

make -C tools/blackbox-tools obj/blackbox_decode

./tools/blackbox-tools/obj/blackbox_decode --help >/dev/null

echo "blackbox_decode built successfully."
