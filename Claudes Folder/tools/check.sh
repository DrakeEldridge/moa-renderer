#!/usr/bin/env bash
# Compile-check MOA Renderer in a GPU-less Linux workspace, then render the CPU mock test.
# Also needs: apt-get install libgl-dev libglfw3-dev (for the live window code)
# Needs: pip install --break-system-packages "nvidia-cuda-runtime-cu12==12.9.*" \
#        "nvidia-cuda-nvcc-cu12==12.9.*" "nvidia-cuda-nvrtc-cu12==12.9.*" pillow
# Usage: "Claudes Folder/tools/check.sh" [tall] [short] [launches] [background] [model.obj|.stl] [zup]
#        (with a model, the tall-block material is used for the model)
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
IMGUI="$OUT/imgui-src"
if [ ! -f "$IMGUI/imgui.h" ]; then
  git clone -q --depth 1 --branch v1.91.9b https://github.com/ocornut/imgui.git "$IMGUI"
fi
ln -sf "$CUDA_PIP/cuda_runtime/lib/libcudart.so.12" "$OUT/lib/libcudart.so"
INC=(-I"$CUDA_PIP/cuda_runtime/include" -I"$CUDA_PIP/cuda_nvcc/include" -I"$OPTIX_INC" -I"$ROOT/src" -I"$ROOT/third_party" -I"$IMGUI" -I"$IMGUI/backends")
mkdir -p "$OUT/imgui"
for f in imgui imgui_draw imgui_tables imgui_widgets backends/imgui_impl_glfw backends/imgui_impl_opengl3; do
  o="$OUT/imgui/$(basename $f).o"; [ -f "$o" ] || g++ -std=c++17 -O1 -c "${INC[@]}" "$IMGUI/$f.cpp" -o "$o"
done

echo "== Device code (NVRTC -> PTX)"
python3 -I "$TOOLS/nvrtc_check.py" "$ROOT/src/device/pathtracer.cu" "$OUT/pathtracer.ptx" "$OPTIX_INC" "$ROOT/src"
echo "   launch params in PTX: $(grep -o 'params\[[0-9]*\]' "$OUT/pathtracer.ptx" | head -1)"

# Same job as cmake/EmbedFile.cmake: put the PTX into a C++ array for the engine library.
python3 -I - "$OUT/pathtracer.ptx" "$OUT/EmbeddedPtx.cpp" <<'PY'
import sys
data = open(sys.argv[1], 'rb').read()
rows = ',\n'.join(','.join('0x%02x' % b for b in data[i:i + 32]) for i in range(0, len(data), 32))
open(sys.argv[2], 'w').write('#include <cstddef>\nnamespace moa {\nextern const unsigned char moaEmbeddedPtx[] = {\n%s,\n0x00};\n'
                             'extern const std::size_t moaEmbeddedPtxSize = %d;\n}\n' % (rows, len(data)))
PY

echo "== Host code (g++)"
g++ -std=c++17 -O2 -Wall -Wextra "${INC[@]}" "$ROOT"/src/*.cpp "$OUT/EmbeddedPtx.cpp" "$OUT"/imgui/*.o -L"$OUT/lib" -lcudart -ldl -lglfw -lGL -o "$OUT/moa"
printf '#include "device/LaunchParams.h"\n#include <cstdio>\nint main(){std::printf("%%zu\\n",sizeof(moa::LaunchParams));}\n' > "$OUT/sz.cpp"
g++ -std=c++17 "${INC[@]}" "$OUT/sz.cpp" -o "$OUT/sz"
echo "   sizeof(LaunchParams) on host: $("$OUT/sz") (must match the PTX value above)"

echo "== CPU mock render"
g++ -std=c++17 -O2 -pthread -w -I"$TOOLS/cpu_mock" -I"$CUDA_PIP/cuda_runtime/include" -I"$CUDA_PIP/cuda_nvcc/include" \
    -I"$ROOT/src" -I"$ROOT/src/device" -I"$ROOT/third_party" \
    "$TOOLS/cpu_mock/cpu_pt.cpp" "$ROOT/src/Scene.cpp" "$ROOT/src/ImageIO.cpp" "$ROOT/src/MeshLoader.cpp" "$ROOT/src/Image.cpp" -o "$OUT/cpu_pt"
(cd "$OUT" && ./cpu_pt "${1:-diffuse}" "${2:-diffuse}" "${3:-16}" "${4:-0}" ${5:+"$5"} ${6:+"$6"})
echo "   images written to $OUT/cpu_*.png"
