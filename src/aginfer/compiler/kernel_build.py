"""CMake-owned provenance for its exact AOT CUBIN; no performance claims."""
import argparse
import os
from pathlib import Path
import subprocess

from .identity import canonical, digest, file_identity
from ..aim import _validate_cubin
from ..errors import ValidationError
from ..schema import CudaArch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cubin", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--arch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(os.environ.get(x) for x in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN")):
        raise ValidationError("unrecorded NVCC environment overrides are not supported")
    _validate_cubin(args.cubin.read_bytes(), CudaArch(args.arch), "built kernel")
    cuobjdump = args.nvcc.parent / "cuobjdump"
    if subprocess.check_output([str(cuobjdump), "--dump-ptx", str(args.cubin)]).strip():
        raise ValidationError("built kernel contains PTX")
    sources = {str(p.relative_to(args.root)): file_identity(p) for p in sorted((args.root / "kernels").glob("*.cu"))}
    sources["CMakeLists.txt"] = file_identity(args.root / "CMakeLists.txt")
    record = {"schema": "aginfer.kernel-build.v1", "arch": args.arch,
        "module": file_identity(args.cubin), "sources": sources,
        "nvcc": {**file_identity(args.nvcc), "version": subprocess.check_output([str(args.nvcc), "--version"], text=True).strip()},
        "flags": ["-cubin", f"-arch=sm_{args.arch}", "--std=c++17"],
        "contains_ptx": False}
    args.output.write_bytes(canonical({**record, "build_sha256": digest(record)}))


if __name__ == "__main__":
    main()
