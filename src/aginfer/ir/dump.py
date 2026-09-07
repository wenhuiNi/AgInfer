from __future__ import annotations

import json

from .model import Function, Op, Program, TensorType, Value
from .verify import verify_program


def dump_program(program: Program) -> str:
    """Return a deterministic, diff-friendly ProgramIR v1 text form."""

    verify_program(program)
    lines = [
        f"program schema={program.schema_major}.{program.schema_minor} entry=@{program.entry}"
    ]
    for dimension in sorted(program.shape_domain.dimensions, key=lambda item: item.symbol):
        lines.append(
            f"shape {dimension.symbol} [{dimension.minimum},{dimension.preferred},{dimension.maximum}]"
        )
    for state in sorted(program.states, key=lambda item: item.name):
        lines.append(f"state @{state.name}: {_type(state.type)} {state.access.value}")
    for function in sorted(program.functions, key=lambda item: item.name):
        lines.extend(_function(function))
    return "\n".join(lines) + "\n"


def _function(function: Function) -> list[str]:
    inputs = ", ".join(_value(value) for value in function.inputs)
    outputs = ", ".join(f"%{value_id}" for value_id in function.outputs)
    result = [f"func @{function.name}({inputs}) -> ({outputs}) {{"]
    result.extend(f"  {_op(op)}" for op in function.body.ops)
    result.append(f"  return {outputs}")
    result.append("}")
    return result


def _op(op: Op) -> str:
    outputs = ", ".join(_value(value) for value in op.outputs)
    inputs = ", ".join(f"%{value_id}" for value_id in op.inputs)
    attributes = ""
    if op.attributes:
        attributes = " " + json.dumps(
            dict(op.attributes), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    prefix = f"{outputs} = " if outputs else ""
    return f"{prefix}{op.opcode}({inputs}){attributes}"


def _value(value: Value) -> str:
    return f"%{value.value_id}: {_type(value.type)}"


def _type(tensor: TensorType) -> str:
    shape = "x".join(str(dimension) for dimension in tensor.shape)
    return f"{tensor.dtype.value}[{shape}]<{tensor.layout.value},{tensor.device.value}>"
