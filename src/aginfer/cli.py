from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aim import AimReader
from .checkpoint import resolve_source
from .errors import AgInferError
from .processor_assets import inspect_source_asset_contract
from .source_manifest import build_source_manifest
from .source_package import open_source_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aginfer",
        description="Compile, inspect and verify experimental AgInfer artifacts",
    )
    parser.add_argument("--version", action="version", version="aginfer 0.1.0 (experimental AIM schema 2.0)")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="verify and display an AIM manifest")
    inspect_parser.add_argument("aim", type=Path)
    inspect_parser.add_argument("--no-payload-checksums", action="store_true")

    compile_parser = commands.add_parser("compile", help="compile an offline source checkpoint to a candidate AIM")
    compile_parser.add_argument("source", type=Path)
    compile_parser.add_argument("--frontend", choices=["pi05"], default="pi05")
    compile_parser.add_argument("--cubin", type=Path, required=True)
    compile_parser.add_argument("--kernel-build-record", type=Path, required=True)
    compile_parser.add_argument("--algorithms", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--scratch", type=Path)
    compile_parser.add_argument("--selection-report", type=Path)

    selection_parser = commands.add_parser("select-algorithms", help="select fixed algorithms offline with native cuBLASLt AlgoCheck")
    selection_parser.add_argument("source", type=Path)
    selection_parser.add_argument("--cubin", type=Path, required=True)
    selection_parser.add_argument("--selector", type=Path, required=True)
    selection_parser.add_argument("--output", type=Path, required=True)
    selection_parser.add_argument("--report", type=Path, required=True)
    selection_parser.add_argument("--workspace-limit", type=int, default=4 * 1024 * 1024,
        help="offline linear/patch GEMM workspace cap in bytes (default: 4194304)")

    verify_parser = commands.add_parser("verify", help="validate v2 executable payloads and compiler identities without CUDA execution")
    verify_parser.add_argument("aim", type=Path)
    verify_parser.add_argument("--require-build-record", action="store_true")

    source_parser = commands.add_parser(
        "source-manifest",
        help="inspect checkpoint assets and write a lightweight source manifest",
    )
    source_parser.add_argument("source")
    source_parser.add_argument("--revision", help="optional Hub revision or local source label")
    source_parser.add_argument("--offline", action="store_true")
    source_parser.add_argument("--output", type=Path)

    contract_parser = commands.add_parser(
        "source-contract",
        help="validate and display model IO, processor, state, and external asset contracts",
    )
    contract_parser.add_argument("source")
    contract_parser.add_argument("--revision", help="optional Hub revision or local source label")
    contract_parser.add_argument("--offline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compile":
            from .compiler.pipeline import compile_source
            output = compile_source(args.source, output=args.output, cubin_path=args.cubin,
                kernel_record_path=args.kernel_build_record, algorithms_path=args.algorithms,
                frontend=args.frontend, scratch=args.scratch, selection_report_path=args.selection_report)
        elif args.command == "select-algorithms":
            from .compiler.selection import select_source_algorithms
            output = select_source_algorithms(args.source, cubin_path=args.cubin, selector_path=args.selector,
                output=args.output, report_path=args.report, workspace_limit=args.workspace_limit)
        elif args.command == "verify":
            from .compiler.verify import verify_artifact
            output = verify_artifact(args.aim, require_build_record=args.require_build_record)
        elif args.command == "inspect":
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
                "compatibility": {
                    "cuda_driver": [
                        info.compatibility.cuda_driver_min,
                        info.compatibility.cuda_driver_max,
                    ],
                    "cuda_runtime": [
                        info.compatibility.cuda_runtime_min,
                        info.compatibility.cuda_runtime_max,
                    ],
                    "providers": [
                        {
                            "id": provider.provider_id,
                            "abi_min": provider.abi_min,
                            "abi_max": provider.abi_max,
                        }
                        for provider in info.compatibility.providers
                    ],
                },
            }
        elif args.command == "source-manifest":
            root, revision = resolve_source(args.source, args.revision, args.offline)
            manifest = build_source_manifest(root, revision=revision)
            if args.output is not None:
                manifest.write(args.output)
            namespace_counts: dict[str, int] = {}
            for tensor in manifest.tensors:
                namespace_counts[tensor.namespace] = namespace_counts.get(tensor.namespace, 0) + 1
            output = {
                "source": manifest.source,
                "revision": manifest.revision,
                "asset_count": len(manifest.assets),
                "tensor_count": len(manifest.tensors),
                "tensor_namespaces": namespace_counts,
                "output": str(args.output) if args.output is not None else None,
            }
        else:
            source = open_source_package(
                args.source,
                revision=args.revision,
                offline=args.offline,
            )
            contract = inspect_source_asset_contract(source)
            output = {
                "source": source.manifest.source,
                "revision": source.manifest.revision,
                "source_type": contract.source_type,
                "asset_count": len(source.manifest.assets),
                "tensor_count": len(source.manifest.tensors),
                "input_features": [
                    {"name": item.name, "type": item.kind, "shape": list(item.shape)}
                    for item in contract.input_features
                ],
                "output_features": [
                    {"name": item.name, "type": item.kind, "shape": list(item.shape)}
                    for item in contract.output_features
                ],
                "pipelines": [
                    {
                        "namespace": pipeline.namespace,
                        "name": pipeline.name,
                        "source_asset_id": pipeline.source_asset_id,
                        "steps": [
                            {
                                "index": step.index,
                                "registry_name": step.registry_name,
                                "state_asset_id": step.state_asset_id,
                            }
                            for step in pipeline.steps
                        ],
                    }
                    for pipeline in contract.pipelines
                ],
                "external_requirements": [
                    {
                        "kind": requirement.kind,
                        "locator": requirement.locator,
                        "requested_by": requirement.requested_by,
                    }
                    for requirement in contract.external_requirements
                ],
            }
        print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except (AgInferError, ValueError, OSError) as exc:
        print(f"aginfer: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
