from __future__ import annotations

import dataclasses
import hashlib
import unittest
from types import SimpleNamespace

from aginfer.errors import ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value
from aginfer.lowering import (
    CommandOperand,
    CommandPlacement,
    CommandTag,
    OperandAccess,
    ProviderCommand,
    assemble_command_stream,
    build_execution_schedule,
    build_memory_plan,
    placements_from_partial_lowering,
)
from aginfer.schema import CudaArch
from aginfer.lowering.assemble import order_command_placements
from aginfer.lowering.memory import AllocationRegion, replan_memory_for_commands


def _fixture():
    tensor = TensorType(DType.F32, (1,), device=Device.CUDA)
    program = Program(
        (
            Function(
                "main",
                (Value("x", tensor),),
                ("z",),
                Region(
                    (
                        Op("relu", ("x",), (Value("y", tensor),)),
                        Op("relu", ("y",), (Value("z", tensor),)),
                    )
                ),
            ),
        ),
        "main",
    )
    schedule = build_execution_schedule(program)
    memory = build_memory_plan(schedule)
    command = ProviderCommand(
        CommandTag.CUDA_KERNEL,
        7,
        1,
        0,
        hashlib.sha256(b"fused-relu").hexdigest(),
        (
            CommandOperand(schedule.entry_inputs[0][1], OperandAccess.READ),
            CommandOperand(schedule.entry_outputs[0][1], OperandAccess.WRITE),
        ),
        b"fused-relu-payload",
        capture_safe=True,
    )
    return schedule, memory, CommandPlacement(0, (0, 1), command)


class CommandAssemblyTests(unittest.TestCase):
    def test_command_memory_discards_fused_internals_and_uses_external_output(self) -> None:
        schedule, memory, placement = _fixture()
        planned = replan_memory_for_commands(schedule, memory, (placement,))
        self.assertEqual(planned.arena_bytes, 0)
        self.assertEqual(planned.allocations[schedule.ops[0].outputs[0]].region, AllocationRegion.UNUSED)
        self.assertEqual(planned.allocations[schedule.entry_outputs[0][1]].region, AllocationRegion.EXTERNAL)
        self.assertEqual(planned.to_dict()['liveness_index_space'], 'provider_command')
        self.assertIsNotNone(planned.command_schedule_sha256)

    def test_fused_boundary_dependency_orders_later_projection_before_earlier_anchor(self) -> None:
        schedule, memory, placement = _fixture()
        # The source index is only a tie-breaker: the declared boundary is x -> y -> z.
        x, y, z = 0, schedule.ops[0].outputs[0], schedule.entry_outputs[0][1]
        producer = CommandPlacement(1, (1,), dataclasses.replace(placement.command,
            operands=(CommandOperand(x, OperandAccess.READ), CommandOperand(y, OperandAccess.WRITE))))
        consumer = CommandPlacement(0, (0,), dataclasses.replace(placement.command,
            operands=(CommandOperand(y, OperandAccess.READ), CommandOperand(z, OperandAccess.WRITE))))
        ordered = order_command_placements(schedule, memory, (consumer, producer))
        self.assertEqual(tuple(p.execution_index for p in ordered), (1, 0))
        planned = replan_memory_for_commands(schedule, memory, (consumer, producer))
        self.assertEqual((planned.intervals[0].start, planned.intervals[0].end), (0, 1))
        stream = assemble_command_stream(schedule, planned, (consumer, producer), target_arch=CudaArch.SM120)
        self.assertEqual(stream.commands, (producer.command, consumer.command))
        missing = dataclasses.replace(producer, command=dataclasses.replace(producer.command,
            operands=(CommandOperand(z, OperandAccess.READ), CommandOperand(y, OperandAccess.WRITE))))
        with self.assertRaisesRegex(ValidationError, 'cycle'):
            order_command_placements(schedule, memory, (consumer, missing))
        with self.assertRaisesRegex(ValidationError, 'unproduced'):
            replan_memory_for_commands(schedule, memory, (consumer,))

    def test_fused_coverage_produces_one_deterministic_stream(self) -> None:
        schedule, memory, placement = _fixture()
        first = assemble_command_stream(
            schedule, memory, (placement,), target_arch=CudaArch.SM120
        )
        second = assemble_command_stream(
            schedule, memory, (placement,), target_arch=CudaArch.SM120
        )
        self.assertEqual(first.to_bytes(), second.to_bytes())
        self.assertEqual(len(first.commands), 1)
        self.assertEqual(first.workspace_bytes, 0)

    def test_incomplete_overlap_bad_anchor_and_elided_work_fail_closed(self) -> None:
        schedule, memory, placement = _fixture()
        with self.assertRaisesRegex(ValidationError, "missing"):
            assemble_command_stream(
                schedule,
                memory,
                (dataclasses.replace(placement, covered_execution_indices=(0,)),),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "overlap"):
            assemble_command_stream(
                schedule,
                memory,
                (
                    placement,
                    dataclasses.replace(
                        placement,
                        execution_index=1,
                        covered_execution_indices=(1,),
                    ),
                ),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "anchor"):
            CommandPlacement(2, (0, 1), placement.command)

    def test_shared_fused_producer_is_owned_by_next_command_in_invocation(self) -> None:
        schedule, _, placement = _fixture()
        lowered = SimpleNamespace(
            commands=(
                SimpleNamespace(
                    execution_index=1,
                    fused_execution_indices=(1,),
                    command=placement.command,
                ),
            ),
            fused_execution_indices=(0, 1),
        )
        normalized = placements_from_partial_lowering(schedule, lowered)
        self.assertEqual(normalized[0].execution_index, 1)
        self.assertEqual(normalized[0].covered_execution_indices, (0, 1))


if __name__ == "__main__":
    unittest.main()
