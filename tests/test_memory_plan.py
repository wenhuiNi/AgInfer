from __future__ import annotations

import unittest

from aginfer.errors import ValidationError
from aginfer.ir import (
    DType,
    Device,
    DimensionRange,
    Function,
    Op,
    Program,
    Region,
    ShapeDomain,
    State,
    StateAccess,
    TensorType,
    Value,
    attributes,
)
from aginfer.lowering import (
    AllocationRegion,
    build_execution_schedule,
    build_memory_plan,
    dump_memory_plan,
)


def _reuse_program() -> Program:
    tensor = TensorType(DType.F32, (4,), device=Device.CUDA)
    return Program(
        functions=(
            Function(
                "main",
                (Value("x", tensor),),
                ("output",),
                Region(
                    (
                        Op("relu", ("x",), (Value("a", tensor),)),
                        Op("relu", ("a",), (Value("dead", tensor),)),
                        Op("relu", ("x",), (Value("output", tensor),)),
                    )
                ),
            ),
        ),
        entry="main",
    )


class MemoryPlanTests(unittest.TestCase):
    def test_linear_scan_reuses_expired_storage_but_not_same_op_input(self) -> None:
        schedule = build_execution_schedule(_reuse_program())
        plan = build_memory_plan(schedule, alignment=16)
        first = plan.allocations[schedule.ops[0].outputs[0]]
        same_op_output = plan.allocations[schedule.ops[1].outputs[0]]
        later = plan.allocations[schedule.ops[2].outputs[0]]
        self.assertEqual(first.region, AllocationRegion.ARENA)
        self.assertNotEqual(first.offset, same_op_output.offset)
        self.assertEqual(first.offset, later.offset)
        self.assertEqual(plan.arena_bytes, 32)
        self.assertEqual(plan.peak_live_arena_bytes, 32)
        self.assertEqual(plan.naive_arena_bytes, 48)
        self.assertEqual(plan.to_dict()["summary"]["arena_reuse_bytes"], 16)  # type: ignore[index]

    def test_entry_output_remains_live_through_schedule_end(self) -> None:
        tensor = TensorType(DType.F32, (4,), device=Device.CUDA)
        program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("output",),
                    Region(
                        (
                            Op("relu", ("x",), (Value("output", tensor),)),
                            Op("relu", ("x",), (Value("later", tensor),)),
                        )
                    ),
                ),
            ),
            entry="main",
        )
        schedule = build_execution_schedule(program)
        plan = build_memory_plan(schedule, alignment=16)
        output_id = schedule.entry_outputs[0][1]
        output_interval = next(item for item in plan.intervals if item.value_id == output_id)
        later_interval = next(
            item for item in plan.intervals if item.value_id == schedule.ops[1].outputs[0]
        )
        self.assertEqual(output_interval.end, len(schedule.ops))
        self.assertNotEqual(output_interval.offset, later_interval.offset)

    def test_contiguous_reshape_chain_aliases_input_and_extends_root_lifetime(self) -> None:
        flat = TensorType(DType.F32, (4,), device=Device.CUDA)
        matrix = TensorType(DType.F32, (2, 2), device=Device.CUDA)
        program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", flat),),
                    ("output",),
                    Region(
                        (
                            Op(
                                "reshape",
                                ("x",),
                                (Value("matrix", matrix),),
                                attributes(shape=(2, 2)),
                            ),
                            Op(
                                "reshape",
                                ("matrix",),
                                (Value("flat", flat),),
                                attributes(shape=(4,)),
                            ),
                            Op("relu", ("flat",), (Value("output", flat),)),
                        )
                    ),
                ),
            ),
            entry="main",
        )
        schedule = build_execution_schedule(program)
        plan = build_memory_plan(schedule, alignment=16)
        input_id = schedule.entry_inputs[0][1]
        first_alias = plan.allocations[schedule.ops[0].outputs[0]]
        second_alias = plan.allocations[schedule.ops[1].outputs[0]]
        self.assertEqual((first_alias.region, first_alias.alias_of), (AllocationRegion.ALIAS, input_id))
        self.assertEqual((second_alias.region, second_alias.alias_of), (AllocationRegion.ALIAS, input_id))
        self.assertEqual(plan.elided_ops, (0, 1))

    def test_reshape_alias_use_prevents_premature_root_reuse(self) -> None:
        tensor = TensorType(DType.F32, (4,), device=Device.CUDA)
        program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("output",),
                    Region(
                        (
                            Op("relu", ("x",), (Value("root", tensor),)),
                            Op(
                                "reshape",
                                ("root",),
                                (Value("alias", tensor),),
                                attributes(shape=(4,)),
                            ),
                            Op("relu", ("x",), (Value("branch", tensor),)),
                            Op("add", ("alias", "branch"), (Value("output", tensor),)),
                        )
                    ),
                ),
            ),
            entry="main",
        )
        schedule = build_execution_schedule(program)
        plan = build_memory_plan(schedule, alignment=16)
        root = plan.allocations[schedule.ops[0].outputs[0]]
        branch = plan.allocations[schedule.ops[2].outputs[0]]
        alias = plan.allocations[schedule.ops[1].outputs[0]]
        self.assertEqual(alias.alias_of, root.value_id)
        self.assertNotEqual(root.offset, branch.offset)

    def test_row_major_transpose_of_singleton_axes_is_a_view(self) -> None:
        source = TensorType(DType.BF16, (1, 50, 1, 256), device=Device.CUDA)
        target = TensorType(DType.BF16, (1, 1, 50, 256), device=Device.CUDA)
        program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", source),),
                    ("out",),
                    Region(
                        (
                            Op(
                                "transpose",
                                ("x",),
                                (Value("out", target),),
                                attributes(permutation=(0, 2, 1, 3)),
                            ),
                        )
                    ),
                ),
            ),
            entry="main",
        )
        schedule = build_execution_schedule(program)
        plan = build_memory_plan(schedule)
        allocation = plan.allocations[schedule.entry_outputs[0][1]]
        self.assertEqual(
            (allocation.region, allocation.alias_of),
            (AllocationRegion.ALIAS, schedule.entry_inputs[0][1]),
        )
        self.assertEqual(plan.elided_ops, (0,))

    def test_broadcast_that_only_inserts_singleton_axes_is_a_view(self) -> None:
        source = TensorType(DType.F32, (2, 3), device=Device.CUDA)
        target = TensorType(DType.F32, (1, 2, 3), device=Device.CUDA)
        program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", source),),
                    ("out",),
                    Region(
                        (
                            Op(
                                "broadcast_in_dim",
                                ("x",),
                                (Value("out", target),),
                                attributes(
                                    shape=(1, 2, 3),
                                    broadcast_dimensions=(1, 2),
                                ),
                            ),
                        )
                    ),
                ),
            ),
            entry="main",
        )
        schedule = build_execution_schedule(program)
        plan = build_memory_plan(schedule)
        output = plan.allocations[schedule.entry_outputs[0][1]]
        self.assertEqual(
            (output.region, output.alias_of),
            (AllocationRegion.ALIAS, schedule.entry_inputs[0][1]),
        )
        self.assertEqual(plan.elided_ops, (0,))

    def test_broadcast_with_real_repetition_is_not_a_view(self) -> None:
        cases = (
            (
                TensorType(DType.F32, (1,), device=Device.CUDA),
                TensorType(DType.F32, (4,), device=Device.CUDA),
                (0,),
            ),
            (
                TensorType(DType.F32, (2, 3), device=Device.CUDA),
                TensorType(DType.F32, (2, 4, 3), device=Device.CUDA),
                (0, 2),
            ),
        )
        for source, target, dimensions in cases:
            with self.subTest(source=source.shape, target=target.shape):
                program = Program(
                    functions=(
                        Function(
                            "main",
                            (Value("x", source),),
                            ("out",),
                            Region(
                                (
                                    Op(
                                        "broadcast_in_dim",
                                        ("x",),
                                        (Value("out", target),),
                                        attributes(
                                            shape=target.shape,
                                            broadcast_dimensions=dimensions,
                                        ),
                                    ),
                                )
                            ),
                        ),
                    ),
                    entry="main",
                )
                schedule = build_execution_schedule(program)
                plan = build_memory_plan(schedule)
                output = plan.allocations[schedule.entry_outputs[0][1]]
                self.assertEqual(output.region, AllocationRegion.ARENA)
                self.assertFalse(plan.elided_ops)

    def test_state_has_persistent_region_and_read_is_an_alias(self) -> None:
        tensor = TensorType(DType.BF16, (3,), device=Device.CUDA)
        program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("out",),
                    Region(
                        (
                            Op("state_write", ("x",), (), attributes(state="cache")),
                            Op(
                                "state_read",
                                (),
                                (Value("out", tensor),),
                                attributes(state="cache"),
                            ),
                        )
                    ),
                ),
            ),
            entry="main",
            states=(State("cache", tensor, StateAccess.READ_WRITE),),
        )
        schedule = build_execution_schedule(program)
        plan = build_memory_plan(schedule)
        state_id = schedule.states[0][1]
        state = plan.allocations[state_id]
        alias = plan.allocations[schedule.entry_outputs[0][1]]
        self.assertEqual((state.region, state.offset, state.allocation_bytes), (AllocationRegion.STATE, 0, 256))
        self.assertEqual((alias.region, alias.alias_of), (AllocationRegion.ALIAS, state_id))
        self.assertEqual(plan.state_bytes, 256)
        self.assertEqual(plan.elided_ops, (1,))

    def test_symbolic_shape_and_bad_alignment_fail_closed(self) -> None:
        symbolic = TensorType(DType.F32, ("N",), device=Device.CUDA)
        program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", symbolic),),
                    ("out",),
                    Region((Op("relu", ("x",), (Value("out", symbolic),)),)),
                ),
            ),
            entry="main",
            shape_domain=ShapeDomain((DimensionRange("N", 1, 2, 4),)),
        )
        schedule = build_execution_schedule(program)
        with self.assertRaisesRegex(ValidationError, "specialized static shape"):
            build_memory_plan(schedule)
        with self.assertRaisesRegex(ValidationError, "power of two"):
            build_memory_plan(build_execution_schedule(_reuse_program()), alignment=24)

    def test_dump_is_stable(self) -> None:
        first = build_memory_plan(build_execution_schedule(_reuse_program()), alignment=16)
        second = build_memory_plan(build_execution_schedule(_reuse_program()), alignment=16)
        self.assertEqual(dump_memory_plan(first), dump_memory_plan(second))


if __name__ == "__main__":
    unittest.main()
