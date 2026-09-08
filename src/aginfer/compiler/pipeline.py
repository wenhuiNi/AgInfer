"""Source-to-AIM candidate compilation. No input executable plan is accepted."""
import hashlib
from pathlib import Path
import platform
import tempfile

from .identity import canonical, compiler_identity, digest, file_identity, read_json
from .native import FixedAlgorithms, lower_native_program
from ..aim import AimWriter, Compatibility, FileVariantPayload, ProviderRequirement, _validate_cubin, _reject_ptx
from ..errors import ValidationError
from ..executable import compile_executable_plan
from ..ir import dump_program
from ..lowering import dump_execution_schedule, dump_lowering_inventory
from ..lowering.memory import dump_memory_plan
from ..packed_weights import pack_command_weights
from ..recipes import Pi05SourceFrontend
from ..schema import CudaArch, Platform
from ..source_package import open_source_package


def validate_kernel_record(record, cubin):
    fields = {"schema", "arch", "module", "sources", "nvcc", "flags", "contains_ptx", "build_sha256"}
    if not isinstance(record, dict) or set(record) != fields or record["schema"] != "aginfer.kernel-build.v1":
        raise ValidationError("missing or unsupported kernel build record")
    identity = {"bytes": len(cubin), "sha256": hashlib.sha256(cubin).hexdigest()}
    if (record["module"] != identity or record["arch"] != 120
            or record["contains_ptx"] is not False
            or record["flags"] != ["-cubin", "-arch=sm_120", "--std=c++17"]
            or record["build_sha256"] != digest({k: v for k, v in record.items() if k != "build_sha256"})):
        raise ValidationError("kernel build record differs from the requested CUBIN/target")
    if (not isinstance(record["sources"], dict) or not record["sources"]
            or not isinstance(record["nvcc"], dict)
            or not isinstance(record["nvcc"].get("version"), str)):
        raise ValidationError("incomplete kernel source/toolchain identity")
    for name, item in record["sources"].items():
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or not isinstance(item, dict) or set(item) != {"bytes", "sha256"}
                or type(item["bytes"]) is not int or item["bytes"] <= 0
                or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in item["sha256"])):
            raise ValidationError("invalid kernel source identity")
    _validate_cubin(cubin, CudaArch.SM120, "compiler input")
    _reject_ptx(cubin, "compiler input")


def compile_source(source_path, *, output, cubin_path, kernel_record_path, algorithms_path,
                   frontend="pi05", scratch=None, selection_report_path=None, fuse_projections=False, resident_kv=False,
                   constant_evaluator=None, fuse_gelu_mul=False, fuse_ffn=False, fuse_residual=False):
    if frontend != "pi05":
        raise ValidationError("no delivered source frontend for this request")
    output = Path(output)
    if output.exists():
        raise ValidationError("compile output already exists; choose a new artifact path")
    if not output.parent.is_dir():
        raise ValidationError("compile output directory does not exist")
    cubin_path = Path(cubin_path)
    if cubin_path.stat().st_size > 256 * 1024 * 1024:
        raise ValidationError("CUBIN exceeds compilation size limit")
    cubin = cubin_path.read_bytes()
    kernel_record = read_json(kernel_record_path)
    validate_kernel_record(kernel_record, cubin)
    selection = read_json(algorithms_path)
    algorithms = FixedAlgorithms.from_dict(selection)
    source = open_source_package(str(source_path), offline=True)
    imported = Pi05SourceFrontend().import_program(source)
    program, constants, fusion = imported.program, source.constants, None
    if fuse_projections or fuse_ffn:
        from .projection_fusion import fuse_projections as transform, FusionConstantView
        fusion = transform(program, qkv=fuse_projections, ffn=fuse_ffn)
        program = fusion.program
        constants = FusionConstantView(source.constants, fusion.constants)
    resident = None
    if resident_kv:
        from .resident_kv import make_kv_resident
        resident = make_kv_resident(program)
        program = resident.program
    selection_report = None
    if selection_report_path is not None:
        from .selection import validate_selection_report
        selection_report = validate_selection_report(read_json(selection_report_path), selection, cubin,
            hashlib.sha256(dump_program(program).encode()).hexdigest())
    source_files = {asset.path: file_identity(source.assets.root / asset.path) for asset in source.manifest.assets}
    inventory, schedule, memory, commands, literals, capabilities, placements = lower_native_program(
        program, cubin, algorithms, include_placements=True, fuse_gelu_mul=fuse_gelu_mul,
        fuse_ffn=fuse_ffn, fuse_residual=fuse_residual)
    with tempfile.TemporaryDirectory(prefix="aginfer-compile-", dir=scratch) as temporary:
        root = Path(temporary)
        # Freeze the exact kernel bytes read above against concurrent source edits.
        frozen_cubin = root / "module.cubin"
        frozen_cubin.write_bytes(cubin)
        computed, folding = None, None
        if constant_evaluator is not None:
            from .constant_folding import fold_native_constants
            memory, commands, computed, folding = fold_native_constants(root,
                evaluator=constant_evaluator, cubin_path=frozen_cubin, schedule=schedule,
                memory=memory, commands=commands, placements=placements, inventory=inventory,
                constants=constants, literals=literals)
            used = {cmd.capability_digest for cmd in commands.commands}
            capabilities = tuple(cap for cap in capabilities if cap.digest in used)
        from .constant_casts import fold_constant_casts
        memory, commands, widened, cast_report = fold_constant_casts(schedule, memory, commands, placements)
        used = {cmd.capability_digest for cmd in commands.commands}
        capabilities = tuple(cap for cap in capabilities if cap.digest in used)
        weights = pack_command_weights(root / "weights.bin", schedule, memory, inventory, commands,
            constants=constants, literal_materialization=literals, computed_constants=computed,
            widened_constants=widened)
        plan = compile_executable_plan(schedule, memory, commands, weights.spans, weights_bytes=weights.byte_size)
        plan_path = root / "plan.bin"
        plan_path.write_bytes(plan.data)
        after = {asset.path: file_identity(source.assets.root / asset.path) for asset in source.manifest.assets}
        if after != source_files:
            raise ValidationError("checkpoint changed during compilation")
        cap_records = {cap.digest: cap.to_dict() for cap in capabilities}
        build = {"schema": "aginfer.candidate-build.v1", "frontend": Pi05SourceFrontend.frontend_id,
            "compiler": compiler_identity(), "python": platform.python_version(),
            "source_files": source_files, "kernel_build": kernel_record,
            "fixed_algorithms": selection,
            "program_sha256": hashlib.sha256(dump_program(program).encode()).hexdigest(),
            "inventory_sha256": hashlib.sha256(dump_lowering_inventory(inventory).encode()).hexdigest(),
            "schedule_sha256": hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest(),
            "memory_sha256": hashlib.sha256(dump_memory_plan(memory).encode()).hexdigest(),
            "command_stream_sha256": hashlib.sha256(commands.to_bytes()).hexdigest(),
            "plan_sha256": hashlib.sha256(plan.data).hexdigest(),
            "weights": {"bytes": weights.byte_size, "sha256": weights.sha256},
            "capabilities": cap_records,
            "validation": {"status": "not_run", "capture_verified": False, "performance_claim": None}}
        if fusion is not None:
            build["projection_fusion"] = fusion.report()
        if resident is not None:
            build["resident_kv"] = resident.report()
        if folding is not None:
            build["constant_folding"] = folding
        if cast_report is not None:
            build["constant_casts"] = cast_report
        if fusion is not None or resident is not None:
            build["source_program_sha256"] = hashlib.sha256(dump_program(imported.program).encode()).hexdigest()
        if selection_report is not None:
            build["algorithm_selection"] = selection_report
        manifest = {"status": "unvalidated_candidate", "build": build, "build_sha256": digest(build)}
        info = AimWriter.write_streaming(output, platform=Platform.LINUX_X86_64_GNU,
            manifest=manifest,
            graph={"inputs": list(schedule.entry_inputs), "outputs": list(schedule.entry_outputs)},
            tensors={"count": len(schedule.values)},
            compatibility=Compatibility(cuda_driver_min=13000, cuda_runtime_min=12080,
                cuda_runtime_max=12999, providers=(ProviderRequirement(1, 12, 12),)),
            variants=(FileVariantPayload(CudaArch.SM120, frozen_cubin, weights.path, plan_path),))
    return {"status": "compiled_candidate", "build_sha256": manifest["build_sha256"],
        "aim_checksum": info.file_sha256, "file": file_identity(output),
        "commands": len(commands.commands), "capabilities": len(capabilities),
        "arena_bytes": memory.arena_bytes, "workspace_bytes": commands.workspace_bytes,
        "validation": "not_run"}
