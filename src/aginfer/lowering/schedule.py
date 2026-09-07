from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from ..errors import ValidationError
from ..ir import Program, TensorType, verify_program
from .inventory import (
    DEFAULT_MAX_EXPANDED_OPS,
    TensorSignature,
    build_lowering_inventory,
    dump_lowering_inventory,
)


EXECUTION_SCHEDULE_SCHEMA = "aginfer.execution-schedule.v1"


class ValueStorage(str, Enum):
    ENTRY_INPUT = "entry_input"
    CONSTANT = "constant"
    STATE = "state"
    STATE_ALIAS = "state_alias"
    TEMPORARY = "temporary"


@dataclass(frozen=True, slots=True)
class ScheduledValue:
    value_id: int
    debug_name: str
    type: TensorSignature
    storage: ValueStorage
    producer: int | None = None
    alias_of: int | None = None
    constant_identity: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "value_id": self.value_id,
            "debug_name": self.debug_name,
            "type": self.type.to_dict(),
            "storage": self.storage.value,
        }
        if self.producer is not None:
            result["producer"] = self.producer
        if self.alias_of is not None:
            result["alias_of"] = self.alias_of
        if self.constant_identity is not None:
            result["constant_identity"] = self.constant_identity
        return result


@dataclass(frozen=True, slots=True)
class FunctionInvocation:
    invocation_id: int
    function: str
    parent_invocation: int | None
    call_site: str | None
    repeat_index: int

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "invocation_id": self.invocation_id,
            "function": self.function,
            "repeat_index": self.repeat_index,
        }
        if self.parent_invocation is not None:
            result["parent_invocation"] = self.parent_invocation
        if self.call_site is not None:
            result["call_site"] = self.call_site
        return result


@dataclass(frozen=True, slots=True)
class ScheduledOp:
    execution_index: int
    invocation_id: int
    site: str
    opcode: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    attributes: tuple[tuple[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index,
            "invocation_id": self.invocation_id,
            "site": self.site,
            "opcode": self.opcode,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attributes": {key: value for key, value in self.attributes},
        }


@dataclass(frozen=True, slots=True)
class ExecutionSchedule:
    program_sha256: str
    inventory_sha256: str
    entry: str
    entry_inputs: tuple[tuple[str, int], ...]
    entry_outputs: tuple[tuple[str, int], ...]
    states: tuple[tuple[str, int], ...]
    invocations: tuple[FunctionInvocation, ...]
    values: tuple[ScheduledValue, ...]
    ops: tuple[ScheduledOp, ...]

    def to_dict(self) -> dict[str, object]:
        storage_counts: dict[str, int] = {}
        for value in self.values:
            storage_counts[value.storage.value] = storage_counts.get(value.storage.value, 0) + 1
        return {
            "schema": EXECUTION_SCHEDULE_SCHEMA,
            "program_sha256": self.program_sha256,
            "inventory_sha256": self.inventory_sha256,
            "entry": self.entry,
            "summary": {
                "invocations": len(self.invocations),
                "values": len(self.values),
                "runtime_ops": len(self.ops),
                "values_by_storage": dict(sorted(storage_counts.items())),
            },
            "entry_inputs": [
                {"name": name, "value_id": value_id} for name, value_id in self.entry_inputs
            ],
            "entry_outputs": [
                {"name": name, "value_id": value_id} for name, value_id in self.entry_outputs
            ],
            "states": [
                {"name": name, "value_id": value_id} for name, value_id in self.states
            ],
            "invocation_records": [item.to_dict() for item in self.invocations],
            "values": [item.to_dict() for item in self.values],
            "ops": [item.to_dict() for item in self.ops],
        }


def build_execution_schedule(
    program: Program,
    *,
    max_expanded_ops: int = DEFAULT_MAX_EXPANDED_OPS,
) -> ExecutionSchedule:
    """Inline fixed calls into an SSA schedule without loading constant payloads."""

    verify_program(program)
    inventory = build_lowering_inventory(program, max_expanded_ops=max_expanded_ops)
    functions = {function.name: function for function in program.functions}
    states_by_name = {state.name: state for state in program.states}
    values: list[ScheduledValue] = []
    ops: list[ScheduledOp] = []
    invocations: list[FunctionInvocation] = []
    shared_constants: dict[tuple[object, ...], int] = {}

    def add_value(
        debug_name: str,
        tensor_type: TensorType,
        storage: ValueStorage,
        *,
        producer: int | None = None,
        alias_of: int | None = None,
        constant_identity: str | None = None,
    ) -> int:
        value_id = len(values)
        values.append(
            ScheduledValue(
                value_id,
                debug_name,
                TensorSignature.from_type(tensor_type),
                storage,
                producer,
                alias_of,
                constant_identity,
            )
        )
        maximum_values = max_expanded_ops * 8 + len(program.states) + len(
            functions[program.entry].inputs
        )
        if len(values) > maximum_values:
            raise ValidationError("execution schedule value count exceeds its bounded expansion")
        return value_id

    entry_function = functions[program.entry]
    entry_inputs = tuple(
        (
            value.value_id,
            add_value(
                f"entry_input:{value.value_id}",
                value.type,
                ValueStorage.ENTRY_INPUT,
            ),
        )
        for value in entry_function.inputs
    )
    state_values = tuple(
        (
            state.name,
            add_value(f"state:{state.name}", state.type, ValueStorage.STATE),
        )
        for state in sorted(program.states, key=lambda item: item.name)
    )
    state_ids = dict(state_values)

    def inline(
        function_name: str,
        input_ids: tuple[int, ...],
        *,
        parent_invocation: int | None,
        call_site: str | None,
        repeat_index: int,
    ) -> tuple[int, ...]:
        function = functions[function_name]
        if len(input_ids) != len(function.inputs):
            raise AssertionError("verified call signature lost input arity")
        invocation_id = len(invocations)
        invocations.append(
            FunctionInvocation(
                invocation_id,
                function_name,
                parent_invocation,
                call_site,
                repeat_index,
            )
        )
        local_values = {
            value.value_id: global_id for value, global_id in zip(function.inputs, input_ids)
        }
        for op_index, op in enumerate(function.body.ops):
            site = f"{function_name}:{op_index}"
            if op.opcode in {"constant", "constant_ref"}:
                output = op.outputs[0]
                if op.opcode == "constant_ref":
                    identity = f"{op.attribute('namespace')}:{op.attribute('name')}"
                    key = ("constant_ref", identity)
                else:
                    identity = f"literal:{site}"
                    key = ("literal", function_name, op_index)
                if key not in shared_constants:
                    shared_constants[key] = add_value(
                        f"constant:{site}",
                        output.type,
                        ValueStorage.CONSTANT,
                        constant_identity=identity,
                    )
                elif values[shared_constants[key]].type != TensorSignature.from_type(output.type):
                    raise ValidationError(
                        f"execution schedule constant identity has inconsistent type: {identity}"
                    )
                local_values[op.outputs[0].value_id] = shared_constants[key]
                continue
            if op.opcode == "call":
                current_inputs = tuple(local_values[value_id] for value_id in op.inputs)
                for selected_repeat in range(int(op.attribute("repeat"))):
                    current_inputs = inline(
                        str(op.attribute("callee")),
                        current_inputs,
                        parent_invocation=invocation_id,
                        call_site=site,
                        repeat_index=selected_repeat,
                    )
                for output, global_id in zip(op.outputs, current_inputs):
                    local_values[output.value_id] = global_id
                continue

            execution_index = len(ops)
            output_ids: list[int] = []
            for output in op.outputs:
                if op.opcode == "state_read":
                    state_name = str(op.attribute("state"))
                    output_ids.append(
                        add_value(
                            f"state_alias:{invocation_id}:{output.value_id}",
                            output.type,
                            ValueStorage.STATE_ALIAS,
                            producer=execution_index,
                            alias_of=state_ids[state_name],
                        )
                    )
                else:
                    output_ids.append(
                        add_value(
                            f"temporary:{invocation_id}:{output.value_id}",
                            output.type,
                            ValueStorage.TEMPORARY,
                            producer=execution_index,
                        )
                    )
            inputs = tuple(local_values[value_id] for value_id in op.inputs)
            ops.append(
                ScheduledOp(
                    execution_index,
                    invocation_id,
                    site,
                    op.opcode,
                    inputs,
                    tuple(output_ids),
                    tuple(sorted(op.attributes)),
                )
            )
            for output, global_id in zip(op.outputs, output_ids):
                local_values[output.value_id] = global_id
            if op.opcode in {"state_write", "state_update"}:
                state_name = str(op.attribute("state"))
                if state_name not in states_by_name:
                    raise AssertionError("verified state write lost its state")
        return tuple(local_values[value_id] for value_id in function.outputs)

    outputs = inline(
        program.entry,
        tuple(value_id for _, value_id in entry_inputs),
        parent_invocation=None,
        call_site=None,
        repeat_index=0,
    )
    entry_outputs = tuple(zip(entry_function.outputs, outputs))
    if tuple(op.site for op in ops) != tuple(item.site_id for item in inventory.execution_order):
        raise AssertionError("execution schedule disagrees with verified lowering inventory order")
    _verify_schedule(values, ops, entry_inputs, entry_outputs, state_values)
    return ExecutionSchedule(
        program_sha256=inventory.program_sha256,
        inventory_sha256=hashlib.sha256(
            dump_lowering_inventory(inventory).encode("utf-8")
        ).hexdigest(),
        entry=program.entry,
        entry_inputs=entry_inputs,
        entry_outputs=entry_outputs,
        states=state_values,
        invocations=tuple(invocations),
        values=tuple(values),
        ops=tuple(ops),
    )


def dump_execution_schedule(schedule: ExecutionSchedule) -> str:
    if not isinstance(schedule, ExecutionSchedule):
        raise ValidationError("execution schedule dump requires an ExecutionSchedule")
    return json.dumps(
        schedule.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ) + "\n"


def _verify_schedule(
    values: list[ScheduledValue],
    ops: list[ScheduledOp],
    entry_inputs: tuple[tuple[str, int], ...],
    entry_outputs: tuple[tuple[str, int], ...],
    states: tuple[tuple[str, int], ...],
) -> None:
    ids = {value.value_id for value in values}
    if ids != set(range(len(values))):
        raise AssertionError("execution schedule value IDs are not contiguous")
    produced: dict[int, int] = {}
    for expected_index, op in enumerate(ops):
        if op.execution_index != expected_index:
            raise AssertionError("execution schedule op indices are not contiguous")
        if any(value_id not in ids for value_id in op.inputs + op.outputs):
            raise AssertionError("execution schedule contains an undefined operand")
        if any(
            values[value_id].producer is not None
            and values[value_id].producer >= op.execution_index
            for value_id in op.inputs
        ):
            raise AssertionError("execution schedule contains a forward operand reference")
        for value_id in op.outputs:
            if value_id in produced:
                raise AssertionError("execution schedule contains a duplicate value definition")
            produced[value_id] = op.execution_index
    for value in values:
        if value.producer is None:
            if value.storage in {ValueStorage.STATE_ALIAS, ValueStorage.TEMPORARY}:
                raise AssertionError("runtime value has no producer")
        elif produced.get(value.value_id) != value.producer:
            raise AssertionError("runtime value producer receipt is inconsistent")
        if value.storage == ValueStorage.STATE_ALIAS:
            if value.alias_of is None or values[value.alias_of].storage != ValueStorage.STATE:
                raise AssertionError("state alias does not reference state storage")
        elif value.alias_of is not None:
            raise AssertionError("non-state value has an alias target")
    roots = entry_inputs + entry_outputs + states
    if any(value_id not in ids for _, value_id in roots):
        raise AssertionError("execution schedule root references an undefined value")
