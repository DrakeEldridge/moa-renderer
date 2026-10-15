"""Compile a .cu file to PTX with NVRTC (no GPU needed), to check device code compiles."""
import ctypes, sys

import os
N = os.environ.get("CUDA_PIP", "/usr/local/lib/python3.13/dist-packages/nvidia")
lib = ctypes.CDLL(f"{N}/cuda_nvrtc/lib/libnvrtc.so.12")

src_path, out_path = sys.argv[1], sys.argv[2]
includes = sys.argv[3:]
src = open(src_path, "rb").read()

prog = ctypes.c_void_p()
r = lib.nvrtcCreateProgram(ctypes.byref(prog), src, src_path.encode(), 0, None, None)
assert r == 0, r
opts = [b"-std=c++17", b"-arch=compute_75", b"-use_fast_math", b"-lineinfo",
        b"-default-device", b"-rdc=true", b"-D__x86_64",
        f"-I{N}/cuda_runtime/include".encode(), f"-I{N}/cuda_cccl/include".encode()]
opts += [f"-I{i}".encode() for i in includes]
arr = (ctypes.c_char_p * len(opts))(*opts)
res = lib.nvrtcCompileProgram(prog, len(opts), arr)
logsize = ctypes.c_size_t()
lib.nvrtcGetProgramLogSize(prog, ctypes.byref(logsize))
log = ctypes.create_string_buffer(logsize.value)
lib.nvrtcGetProgramLog(prog, log)
if log.value.strip():
    print(log.value.decode())
if res != 0:
    print("NVRTC compile FAILED", res)
    sys.exit(1)
size = ctypes.c_size_t()
lib.nvrtcGetPTXSize(prog, ctypes.byref(size))
ptx = ctypes.create_string_buffer(size.value)
lib.nvrtcGetPTX(prog, ptx)
open(out_path, "wb").write(ptx.value)
print("OK, PTX bytes:", size.value)
