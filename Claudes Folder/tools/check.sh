#!/usr/bin/env bash
# Compile-check MOA Renderer in a GPU-less Linux workspace, then render the CPU mock test.
# Needs: pip install --break-system-packages "nvidia-cuda-runtime-cu12==12.9.*" \
#        "nvidia-cuda-nvcc-cu12==12.9.*" "nvidia-cuda-nvrtc-cu12==12.9.*" pillow
# Usage: "Claudes Folder/tools/check.sh" [tall] [short] [launches] [background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TOOLS="$ROOT/Claudes Folder/tools"
OUT="${OUT:-/tmp/moa-check}"
CUDA_PIP="${CUDA_PIP:-$(python3 -c 'import nvidia,os;print(nvidia.__path__[0])')}"
export CUDA_PIP
OPTIX_INC="${OPTIX_INC:-$OUT/optix-dev/include}"
mkdir -p "$OUT/lib"
if [ ! -f "$OPTIX_INC/optix.h" ]; then
  git clone -q --depth 1 --branch v9.1.0 https://github.com/NVIDIA/optix-dev.git "$OUT/optix-dev"
fi
ln -sf "$CUDA_PIP/cuda_runtime/lib/libcudart.so.12" "$OUT/lib/libcudart.so"
INC=(-I"$CUDA_PIP/cuda_runtime/include" -I"$CUDA_PIP/cuda_nvcc/include" -I"$OPTIX_INC" -I"$ROOT/src" -I"$ROOT/third_party")

echo "== Device code (NVRTC -> PTX)"
python3 -I "$TOOLS/nvrtc_check.py" "$ROOT/src/device/pathtracer.cu" "$OUT/pathtracer.ptx" "$OPTIX_INC" "$ROOT/src"
echo "   launch params in PTX: $(grep -o 'params\[[0-9]*\]' "$OUT/pathtracer.ptx" | head -1)"

echo "== Host code (g++)"
g++ -std=c++17 -O2 -Wall -Wextra "${INC[@]}" "$ROOT"/src/*.cpp -L"$OUT/lib" -lcudart -ldl -o "$OUT/moa"
printf '#include "device/LaunchParams.h"\n#include <cstdio>\nint main(){std::printf("%%zu\\n",sizeof(moa::LaunchParams));}\n' > "$OUT/sz.cpp"
g++ -std=c++17 "${INC[@]}" "$OUT/sz.cpp" -o "$OUT/sz"
echo "   sizeof(LaunchParams) on host: $("$OUT/sz") (must match the PTX value above)"

echo "== CPU mock render"
g++ -std=c++17 -O2 -pthread -w -I"$TOOLS/cpu_mock" -I"$CUDA_PIP/cuda_runtime/include" -I"$CUDA_PIP/cuda_nvcc/include" \
    -I"$ROOT/src" -I"$ROOT/src/device" -I"$ROOT/third_party" \
    "$TOOLS/cpu_mock/cpu_pt.cpp" "$ROOT/src/Scene.cpp" "$ROOT/src/ImageIO.cpp" -o "$OUT/cpu_pt"
(cd "$OUT" && ./cpu_pt "${1:-diffuse}" "${2:-diffuse}" "${3:-16}" "${4:-0}")
echo "   images written to $OUT/cpu_*.png"
