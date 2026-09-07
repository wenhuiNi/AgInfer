from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from ..errors import ValidationError
from ..ir import Device, Op, Program, TensorType, dump_program, verify_program


LOWERING_INVENTORY_SCHEMA = "aginfer.lowering-inventory.v1"
DEFAULT_MAX_EXPANDED_OPS = 1_000_000


class LoweringKind(str, Enum):
    BUILTIN_META = "builtin_meta"
    MEMORY = "memory"
    GEMM = "gemm"
    ATTENTION = "attention"
    AOT_CUDA = "aot_cuda"
    BLOCKED = "blocked"


class RequirementStatus(str, Enum):
    META = "meta"
    REQUIRED = "required"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class TensorSignature:
    dtype: str
    shape: tuple[int | str, ...]
    layout: str
    device: str

    @classmethod
    def from_type(cls, tensor_type: TensorType) -> "TensorSignature":
        return cls(
            tensor_type.dtype.value,
            tensor_type.shape,
            tensor_type.layout.value,
            tensor_type.device.value,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class InventoryOp:
    function: str
    op_index: int
    opcode: str
    executions: int
    lowering_kind: LoweringKind
    status: RequirementStatus
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    input_types: tuple[TensorSignature, ...]
    output_types: tuple[TensorSignature, ...]
    attributes: tuple[tuple[str, object], ...]
    constant_identity: str | None = None
    blocker: str | None = None

    @property
    def site_id(self) -> str:
        return f"{self.function}:{self.op_index}"

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "site": self.site_id,
            "function": self.function,
            "op_index": self.op_index,
            "opcode": self.opcode,
            "executions": self.executions,
            "lowering_kind": self.lowering_kind.value,
            "status": self.status.value,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "input_types": [item.to_dict() for item in self.input_types],
            "output_types": [item.to_dict() for item in self.output_types],
            "attributes": {key: value for key, value in self.attributes},
        }
        if self.constant_identity is not None:
            result["constant_identity"] = self.constant_identity
        if self.blocker is not None:
            result["blocker"] = self.blocker
        return result


@dataclass(frozen=True, slots=True)
class ExecutionRef:
    function: str
    op_index: int

    @property
    def site_id(self) -> str:
        return f"{self.function}:{self.op_index}"


@dataclass(frozen=True, slots=True)
class LoweringInventory:
    program_sha256: str
    entry: str
    function_invocations: tuple[tuple[str, int], ...]
    ops: tuple[InventoryOp, ...]
    execution_order: tuple[ExecutionRef, ...]

    @property
    def blocked(self) -> tuple[InventoryOp, ...]:
        return tuple(item for item in self.ops if item.status == RequirementStatus.BLOCKED)

    def to_dict(self) -> dict[str, object]:
        opcode_counts: dict[str, list[int]] = {}
        kind_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        for item in self.ops:
            counts = opcode_counts.setdefault(item.opcode, [0, 0])
            counts[0] += 1
            counts[1] += item.executions
            kind_counts[item.lowering_kind.value] += item.executions
            status_counts[item.status.value] += item.executions
        return {
            "schema": LOWERING_INVENTORY_SCHEMA,
            "program_sha256": self.program_sha256,
            "entry": self.entry,
            "function_invocations": [
                {"function": name, "count": count}
                for name, count in self.function_invocations
            ],
            "summary": {
                "static_op_sites": len(self.ops),
                "expanded_all_ops": sum(item.executions for item in self.ops),
                "expanded_runtime_ops": len(self.execution_order),
                "blocked_static_sites": len(self.blocked),
                "by_opcode": {
                    opcode: {"sites": counts[0], "executions": counts[1]}
                    for opcode, counts in sorted(opcode_counts.items())
                },
                "expanded_by_lowering_kind": dict(sorted(kind_counts.items())),
                "expanded_by_status": dict(sorted(status_counts.items())),
            },
            "ops": [item.to_dict() for item in self.ops],
            "execution_order": [item.site_id for item in self.execution_order],
        }


def build_lowering_inventory(
    program: Program,
    *,
    max_expanded_ops: int = DEFAULT_MAX_EXPANDED_OPS,
) -> LoweringInventory:
    """Build a payload-free, call-expanded provider requirement inventory.

    ``required`` means that lowering needs a provider capability; it does not
    claim that such a provider is registered. Only compile-time records use
    ``meta``. Unsupported device or semantic envelopes are recorded as
    ``blocked`` with a stable reason.
    """

    verify_program(program)
    if (
        not isinstance(max_expanded_ops, int)
        or isinstance(max_expanded_ops, bool)
        or max_expanded_ops <= 0
    ):
        raise ValidationError("lowering inventory max_expanded_ops must be a positive integer")

    functions = {function.name: function for function in program.functions}
    site_types: dict[tuple[str, int], tuple[tuple[TensorType, ...], tuple[TensorType, ...]]] = {}
    for function in program.functions:
        value_types = {value.value_id: value.type for value in function.inputs}
        for op_index, op in enumerate(function.body.ops):
            site_types[(function.name, op_index)] = (
                tuple(value_types[value_id] for value_id in op.inputs),
                tuple(value.type for value in op.outputs),
            )
            for value in op.outputs:
                value_types[value.value_id] = value.type
    invocation_counts: Counter[str] = Counter()
    execution_order: list[ExecutionRef] = []
    expanded_function_calls = 0

    def expand(function_name: str) -> None:
        nonlocal expanded_function_calls
        expanded_function_calls += 1
        if expanded_function_calls > max_expanded_ops:
            raise ValidationError("lowering inventory call expansion exceeds max_expanded_ops")
        function = functions[function_name]
        invocation_counts[function_name] += 1
        for op_index, op in enumerate(function.body.ops):
            if op.opcode == "call":
                repeat = int(op.attribute("repeat"))
                if repeat > max_expanded_ops:
                    raise ValidationError("lowering inventory call repeat exceeds max_expanded_ops")
                for _ in range(repeat):
                    expand(str(op.attribute("callee")))
                continue
            input_types, output_types = site_types[(function_name, op_index)]
            kind, _, _ = _classify(op, input_types, output_types)
            if kind == LoweringKind.BUILTIN_META:
                continue
            execution_order.append(ExecutionRef(function_name, op_index))
            if len(execution_order) > max_expanded_ops:
                raise ValidationError("lowering inventory execution order exceeds max_expanded_ops")

    expand(program.entry)

    inventory_ops: list[InventoryOp] = []
    for function in sorted(program.functions, key=lambda item: item.name):
        for op_index, op in enumerate(function.body.ops):
            input_types, output_types = site_types[(function.name, op_index)]
            kind, status, blocker = _classify(op, input_types, output_types)
            constant_identity = None
            if op.opcode == "constant_ref":
                constant_identity = f"{op.attribute('namespace')}:{op.attribute('name')}"
            inventory_ops.append(
                InventoryOp(
                    function=function.name,
                    op_index=op_index,
                    opcode=op.opcode,
                    executions=invocation_counts[function.name],
                    lowering_kind=kind,
                    status=status,
                    inputs=op.inputs,
                    outputs=tuple(value.value_id for value in op.outputs),
                    input_types=tuple(TensorSignature.from_type(item) for item in input_types),
                    output_types=tuple(TensorSignature.from_type(item) for item in output_types),
                    attributes=tuple(sorted(op.attributes)),
                    constant_identity=constant_identity,
                    blocker=blocker,
                )
            )

    return LoweringInventory(
        program_sha256=hashlib.sha256(dump_program(program).encode("utf-8")).hexdigest(),
        entry=program.entry,
        function_invocations=tuple(sorted(invocation_counts.items())),
        ops=tuple(inventory_ops),
        execution_order=tuple(execution_order),
    )


def dump_lowering_inventory(inventory: LoweringInventory) -> str:
    if not isinstance(inventory, LoweringInventory):
        raise ValidationError("lowering inventory dump requires a LoweringInventory")
    return json.dumps(
        inventory.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ) + "\n"


def _classify(
    op: Op,
    input_types: tuple[TensorType, ...],
    output_types: tuple[TensorType, ...],
) -> tuple[LoweringKind, RequirementStatus, str | None]:
    if op.opcode in {"constant", "constant_ref", "call"}:
        return LoweringKind.BUILTIN_META, RequirementStatus.META, None

    tensor_types = input_types + output_types
    if any(item.device != Device.CUDA for item in tensor_types):
        return (
            LoweringKind.BLOCKED,
            RequirementStatus.BLOCKED,
            "E2 native inventory currently requires CUDA tensors",
        )

    if op.opcode in {
        "broadcast_in_dim",
        "concat",
        "reshape",
        "slice",
        "state_read",
        "state_write",
        "transpose",
    }:
        return LoweringKind.MEMORY, RequirementStatus.REQUIRED, None
    if op.opcode in {"linear", "matmul"}:
        return LoweringKind.GEMM, RequirementStatus.REQUIRED, None
    if op.opcode == "scaled_dot_product_attention":
        return LoweringKind.ATTENTION, RequirementStatus.REQUIRED, None
    if op.opcode == "conv2d":
        weight = input_types[1]
        pads = tuple(int(value) for value in op.attribute("pads"))
        strides = tuple(int(value) for value in op.attribute("strides"))
        kernel = weight.shape[2:]
        if pads == (0, 0, 0, 0) and tuple(kernel) == strides:
            return LoweringKind.GEMM, RequirementStatus.REQUIRED, None
        return (
            LoweringKind.BLOCKED,
            RequirementStatus.BLOCKED,
            "overlapping or padded conv2d has no E2 provider class",
        )
    if op.opcode in {
        "add",
        "cast",
        "cumulative_sum",
        "gather",
        "gelu",
        "layer_norm",
        "logical_and",
        "mul",
        "reduce_max",
        "reduce_sum",
        "relu",
        "rms_norm",
        "rope",
        "rope_default",
        "silu",
        "sinusoidal_embedding",
        "softmax",
    }:
        return LoweringKind.AOT_CUDA, RequirementStatus.REQUIRED, None
    return (
        LoweringKind.BLOCKED,
        RequirementStatus.BLOCKED,
        f"opcode {op.opcode} has no E2 provider class",
    )
