from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import adapter_names
from .compiler import CompileOptions, compile_model
from .aim import AimReader
from .errors import AgInferError
from .schema import CudaArch, Platform


def _platform(value: str) -> Platform:
    try:
        return Platform.parse(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _arch(value: str) -> CudaArch:
    try:
        return CudaArch.parse(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelc", description="AgInfer target-specific AIM compiler")
    parser.add_argument("--version", action="version", version="modelc 0.1.0 (AIM schema 1.0, ABI 1)")
    commands = parser.add_subparsers(dest="command", required=True)

    compile_parser = commands.add_parser("compile", help="validate a checkpoint and build a target-specific AIM")
    compile_parser.add_argument("--source", required=True)
    compile_parser.add_argument("--revision")
    compile_parser.add_argument("--adapter", required=True, choices=adapter_names())
    compile_parser.add_argument("--platform", required=True, type=_platform)
    compile_parser.add_argument("--cuda-arch", required=True, action="append", type=_arch, dest="cuda_arches")
    compile_parser.add_argument("--profile", required=True, type=Path)
    compile_parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="verified AOT outputs: toolchain.json and <arch>/{kernels.cubin,plan.json,tactics.json}",
    )
    compile_parser.add_argument("--backbone", help="local Qwen2.5-VL checkpoint required by lingbot_vla_1")
    compile_parser.add_argument("--offline", action="store_true")
    compile_parser.add_argument("--output", required=True, type=Path)

    inspect_parser = commands.add_parser("inspect", help="verify and display an AIM manifest")
    inspect_parser.add_argument("aim", type=Path)
    inspect_parser.add_argument("--no-payload-checksums", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compile":
            info = compile_model(
                CompileOptions(
                    source=args.source,
                    revision=args.revision,
                    adapter=args.adapter,
                    platform=args.platform,
                    cuda_arches=tuple(args.cuda_arches),
                    profile_path=args.profile,
                    artifact_dir=args.artifact_dir,
                    output=args.output,
                    offline=args.offline,
                    backbone=args.backbone,
                )
            )
            print(f"wrote {info.path} ({info.file_size} bytes, sha256={info.file_sha256})")
            return 0
        info = AimReader.read(args.aim, verify_payloads=not args.no_payload_checksums)
        output = {
            "path": str(info.path),
            "schema": f"{info.schema_major}.{info.schema_minor}",
            "runtime_abi": info.runtime_abi,
            "platform": info.platform.triple,
            "cuda_arches": [variant.arch.name_string for variant in info.variants],
            "file_size": info.file_size,
            "file_sha256": info.file_sha256,
            "manifest": info.manifest,
            "graph": info.graph,
            "tensor_count": info.tensors.get("count"),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except (AgInferError, ValueError) as exc:
        print(f"modelc: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

