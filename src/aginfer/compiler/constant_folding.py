"""Post-selection folding of pure, constant-rooted native command subgraphs.

No CPU approximation, model-name matching, arbitrary receipt input or runtime
constant evaluation. Only the delivered F32 row-one linear/time/SiLU forms are
eligible initially. Execute a compiler-built constant-only AIM using the same
native provider payloads, then retain its compact boundary values as weights.
"""
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import struct
import subprocess

from .identity import digest, file_identity
from .. import providers as p
from ..aim import AimWriter, Compatibility, FileVariantPayload, ProviderRequirement
from ..errors import FormatError, ValidationError
from ..executable import compile_executable_plan
from ..lowering import dump_execution_schedule
from ..lowering.assemble import order_command_placements
from ..lowering.command import OperandAccess, CommandTag
from ..lowering.memory import (AllocationRegion as R, ValueAllocation, dump_memory_plan,
    replan_memory_for_commands)
from ..packed_weights import pack_command_weights, _roots
from ..schema import Platform

POLICY = "native-constant-rooted-f32-row-one.v1"
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_COMMANDS = 4096


def _pure(command):
    if (command.abi_major, command.abi_minor) != (1, 0):
        return False
    try:
        if command.provider_id == 1 and command.tag == CommandTag.CUBLASLT_MATMUL:
            payload = p.CublasLtLinearPayload.from_bytes(command.payload)
            return payload.problem.dtype == p.CublasLtDType.F32 and payload.problem.m == 1
        if command.provider_id == 2 and command.tag == CommandTag.CUDA_KERNEL:
            if command.payload.startswith(b"AITEM1\0\0"):
                p.TimeEmbeddingPayload.from_bytes(command.payload)
                return True
            payload = p.CudaKernelPayload.from_bytes(command.payload)
            return payload.problem.kernel_id == p.CudaKernelId.SILU_F32
    except (FormatError, ValidationError, ValueError):
        return False
    return False


@dataclass(frozen=True)
class ConstantRegion:
    selected: tuple
    remaining: tuple
    boundary: tuple[int, ...]
    outputs: frozenset[int]


def discover_constant_region(schedule, memory, placements):
    roots = _roots(memory)
    allocations = memory.allocations
    ordered = order_command_placements(schedule, memory, placements)
    known = {a.value_id for a in allocations if a.region == R.CONSTANT}
    external = {roots[v] for _, v in schedule.entry_outputs}
    writes = Counter(roots[o.value_id] for entry in ordered for o in entry.command.operands
                     if o.access != OperandAccess.READ)
    selected, remaining, outputs = [], [], set()
    for entry in ordered:
        cmd = entry.command
        reads = [roots[o.value_id] for o in cmd.operands if o.access == OperandAccess.READ]
        produced = [roots[o.value_id] for o in cmd.operands if o.access == OperandAccess.WRITE]
        eligible = (reads and len(produced) == 1 and _pure(cmd)
            and all(o.byte_offset == 0 and o.access != OperandAccess.READ_WRITE for o in cmd.operands)
            and all(v in known for v in reads)
            and all(allocations[v].region == R.ARENA and v not in external and writes[v] == 1
                    and schedule.values[v].type.dtype == "f32"
                    and allocations[v].byte_size <= MAX_OUTPUT_BYTES for v in produced))
        if eligible:
            selected.append(entry); outputs.update(produced); known.update(produced)
        else:
            remaining.append(entry)
    boundary = sorted({roots[o.value_id] for entry in remaining for o in entry.command.operands
                       if o.access == OperandAccess.READ and roots[o.value_id] in outputs})
    if not selected or not remaining or not boundary:
        raise ValidationError("no foldable constant-only native boundary in this program")
    if len(selected) > MAX_COMMANDS or sum(allocations[v].byte_size for v in outputs) > MAX_OUTPUT_BYTES:
        raise ValidationError("native constant folding exceeds bounded command/output budget")
    return ConstantRegion(tuple(selected), tuple(remaining), tuple(boundary), frozenset(outputs))


def _stream(schedule, template, memory, placements):
    ordered = order_command_placements(schedule, memory, placements)
    return replace(template, commands=tuple(x.command for x in ordered),
        memory_plan_sha256=hashlib.sha256(dump_memory_plan(memory).encode()).hexdigest(),
        arena_bytes=memory.arena_bytes, state_bytes=memory.state_bytes)


def evaluation_plan(schedule, memory, commands, region):
    # Keep source value IDs/coverage anchors for provenance, but no dynamic ports,
    # state or unrelated allocation enters the executable evaluation plan.
    roots = _roots(memory)
    used = {o.value_id for e in region.selected for o in e.command.operands}
    used.update(roots[v] for v in tuple(used))
    evaluation = replace(schedule, entry_inputs=(),
        entry_outputs=tuple((f"constant_{v}", v) for v in region.boundary), states=())
    allocations = tuple(a if a.value_id in used else ValueAllocation(a.value_id, R.UNUSED, a.byte_size)
                        for a in memory.allocations)
    base = replace(memory, schedule_sha256=hashlib.sha256(dump_execution_schedule(evaluation).encode()).hexdigest(),
        allocations=allocations, state_bytes=0, entry_input_bytes=0,
        entry_output_bytes=sum(memory.allocations[v].byte_size for v in region.boundary))
    evaluated_memory = replan_memory_for_commands(evaluation, base, region.selected)
    return evaluation, evaluated_memory, _stream(evaluation, commands, evaluated_memory, region.selected)


def apply_constants(schedule, memory, commands, region, values):
    if set(values) != set(region.boundary):
        raise ValidationError("constant evaluator boundary set differs from discovery")
    for v, data in values.items():
        if (not isinstance(data, bytes) or len(data) != memory.allocations[v].byte_size
                or len(data) % 4 or not all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", data))):
            raise ValidationError("computed F32 constant is short or non-finite")
    roots = _roots(memory)
    external_outputs = {roots[v] for _, v in schedule.entry_outputs}
    allocations = []
    for a in memory.allocations:
        if a.value_id in values:
            allocations.append(ValueAllocation(a.value_id, R.CONSTANT, a.byte_size))
        elif a.value_id in external_outputs:
            allocations.append(ValueAllocation(a.value_id, R.ARENA, a.byte_size, 0, a.byte_size))
        else:
            allocations.append(a)
    base = replace(memory, allocations=tuple(allocations),
        constant_bytes=memory.constant_bytes + sum(map(len, values.values())))
    folded = replan_memory_for_commands(schedule, base, region.remaining)
    return folded, _stream(schedule, commands, folded, region.remaining)


def fold_native_constants(root, *, evaluator, cubin_path, schedule, memory, commands,
                          placements, inventory, constants, literals):
    root, evaluator = Path(root), Path(evaluator).resolve()
    helper_identity = file_identity(evaluator)
    region = discover_constant_region(schedule, memory, placements)
    es, em, ec = evaluation_plan(schedule, memory, commands, region)
    weights = pack_command_weights(root / "evaluation-weights.bin", es, em, inventory, ec,
        constants=constants, literal_materialization=literals)
    plan = compile_executable_plan(es, em, ec, weights.spans, weights_bytes=weights.byte_size)
    plan_path = root / "evaluation-plan.bin"; plan_path.write_bytes(plan.data)
    evaluation_aim = root / "evaluation.aim"
    AimWriter.write_streaming(evaluation_aim, platform=Platform.LINUX_X86_64_GNU,
        manifest={"kind": "offline-constant-evaluation", "policy": POLICY}, graph={}, tensors={},
        compatibility=Compatibility(cuda_driver_min=13000, cuda_runtime_min=12080,
            cuda_runtime_max=12999, providers=(ProviderRequirement(1, 12, 12),)),
        variants=(FileVariantPayload(commands.target_arch, cubin_path, weights.path, plan_path),))
    output = root / "evaluated.bin"
    completed = subprocess.run([str(evaluator), str(evaluation_aim), str(output)],
        capture_output=True, text=True, timeout=120, check=False)
    expected = f"AGINFER_CONSTANT_EVAL_V1 {len(region.boundary)} {len(region.selected)} 2\n"
    if completed.returncode or completed.stdout != expected or file_identity(evaluator) != helper_identity:
        raise ValidationError("native constant evaluator failed or changed: " + completed.stderr[-2000:])
    total = sum(memory.allocations[v].byte_size for v in region.boundary)
    if output.stat().st_size != total:
        raise ValidationError("constant evaluator returned an invalid byte count")
    data = output.read_bytes()
    values, offset = {}, 0
    for v in region.boundary:
        count = memory.allocations[v].byte_size
        values[v] = data[offset:offset + count]; offset += count
    folded, stream = apply_constants(schedule, memory, commands, region, values)
    report = {"policy": POLICY, "evaluator": helper_identity,
        "evaluation_plan_sha256": hashlib.sha256(plan.data).hexdigest(),
        "evaluation_weights": {"bytes": weights.byte_size, "sha256": weights.sha256},
        "source_commands_sha256": hashlib.sha256(commands.to_bytes()).hexdigest(),
        "removed_commands": len(region.selected), "graph_repeat_bit_exact": True,
        "output_bytes": len(data), "output_sha256": hashlib.sha256(data).hexdigest(),
        "values": [{"value_id": v, "bytes": len(values[v]), "sha256": hashlib.sha256(values[v]).hexdigest()}
                   for v in region.boundary]}
    report["sha256"] = digest(report)
    return folded, stream, values, report


def validate_folding_report(report, plan):
    from ..executable import ExecutableValueRegion, ExecutableDType
    fields = {"policy", "evaluator", "evaluation_plan_sha256", "evaluation_weights",
        "source_commands_sha256", "removed_commands", "graph_repeat_bit_exact", "output_bytes",
        "output_sha256", "values", "sha256"}
    if (not isinstance(report, dict) or set(report) != fields or report["policy"] != POLICY
            or report["sha256"] != digest({k: v for k, v in report.items() if k != "sha256"})
            or report["graph_repeat_bit_exact"] is not True
            or type(report["removed_commands"]) is not int or not 1 <= report["removed_commands"] <= MAX_COMMANDS
            or type(report["output_bytes"]) is not int or not 1 <= report["output_bytes"] <= MAX_OUTPUT_BYTES
            or not isinstance(report["values"], list) or not 1 <= len(report["values"]) <= MAX_COMMANDS):
        raise ValidationError("invalid native constant folding report")
    def sha(value):
        return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)
    for key in ("evaluation_plan_sha256", "source_commands_sha256", "output_sha256"):
        if not sha(report[key]):
            raise ValidationError("invalid folding source/output identity")
    for key in ("evaluator", "evaluation_weights"):
        entry = report[key]
        if (not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}
                or type(entry["bytes"]) is not int or entry["bytes"] <= 0 or not sha(entry["sha256"])):
            raise ValidationError("invalid folding evaluator/weights identity")
    last, total = -1, 0
    for entry in report["values"]:
        if (not isinstance(entry, dict) or set(entry) != {"value_id", "bytes", "sha256"}
                or type(entry["value_id"]) is not int or not last < entry["value_id"] < len(plan.values)
                or type(entry["bytes"]) is not int or not sha(entry["sha256"])):
            raise ValidationError("invalid folding boundary record")
        value = plan.values[entry["value_id"]]
        if (value.region != ExecutableValueRegion.WEIGHTS or value.dtype != ExecutableDType.F32
                or value.byte_size != entry["bytes"] or value.sha256 != entry["sha256"]):
            raise ValidationError("folding boundary differs from packed executable value")
        last = entry["value_id"]; total += entry["bytes"]
    if total != report["output_bytes"]:
        raise ValidationError("folding boundary total bytes differ")
