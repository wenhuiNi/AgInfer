from __future__ import annotations

import hashlib

from aginfer.executable import PackedWeightSpan, compile_executable_plan
from aginfer.ir import (
    DType,
    Device,
    Function,
    Op,
    Program,
    Region,
    State,
    StateAccess,
    TensorType,
    Value,
    attributes,
)
from aginfer.lowering import (
    CommandOperand,
    CommandStream,
    CommandTag,
    OperandAccess,
    ProviderCommand,
    build_execution_schedule,
    build_memory_plan,
    dump_memory_plan,
)
from aginfer.schema import CudaArch


WEIGHT_BYTES = b"\x00\x00\x80?"


def executable_fixture():
    tensor = TensorType(DType.F32, (1,), device=Device.CUDA)
    program = Program(
        functions=(
            Function(
                "main",
                (Value("input", tensor),),
                ("output",),
                Region(
                    (
                        Op("state_write", ("input",), (), attributes(state="cache")),
                        Op(
                            "state_read",
                            (),
                            (Value("cached", tensor),),
                            attributes(state="cache"),
                        ),
                        Op(
                            "constant_ref",
                            (),
                            (Value("weight", tensor),),
                            attributes(namespace="model", name="weight"),
                        ),
                        Op("add", ("cached", "weight"), (Value("output", tensor),)),
                    )
                ),
            ),
        ),
        entry="main",
        states=(State("cache", tensor, StateAccess.READ_WRITE),),
    )
    schedule = build_execution_schedule(program)
    memory = build_memory_plan(schedule)
    memory_digest = hashlib.sha256(dump_memory_plan(memory).encode()).hexdigest()
    input_id = schedule.entry_inputs[0][1]
    output_id = schedule.entry_outputs[0][1]
    state_id = schedule.states[0][1]
    alias_id = schedule.ops[1].outputs[0]
    weight_id = next(
        value.value_id
        for value in schedule.values
        if value.constant_identity == "model:weight"
    )
    commands = CommandStream(
        target_arch=CudaArch.SM120,
        memory_plan_sha256=memory_digest,
        value_count=len(schedule.values),
        arena_bytes=memory.arena_bytes,
        state_bytes=memory.state_bytes,
        workspace_bytes=512,
        commands=(
            ProviderCommand(
                CommandTag.MEMORY_COPY,
                7,
                1,
                0,
                hashlib.sha256(b"copy-capability").hexdigest(),
                (
                    CommandOperand(input_id, OperandAccess.READ),
                    CommandOperand(state_id, OperandAccess.WRITE),
                ),
                b"copy-payload",
                capture_safe=True,
            ),
            ProviderCommand(
                CommandTag.CUDA_KERNEL,
                9,
                2,
                3,
                hashlib.sha256(b"add-capability").hexdigest(),
                (
                    CommandOperand(alias_id, OperandAccess.READ),
                    CommandOperand(weight_id, OperandAccess.READ),
                    CommandOperand(output_id, OperandAccess.WRITE),
                ),
                b"add-payload",
                workspace_offset=256,
                workspace_bytes=256,
                capture_safe=True,
            ),
        ),
    )
    plan = compile_executable_plan(
        schedule,
        memory,
        commands,
        (
            PackedWeightSpan(
                weight_id,
                0,
                len(WEIGHT_BYTES),
                hashlib.sha256(WEIGHT_BYTES).hexdigest(),
            ),
        ),
        weights_bytes=256,
    )
    return plan
