from __future__ import annotations

import unittest

from aginfer.errors import ValidationError
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
from aginfer.lowering import ValueStorage, build_execution_schedule, dump_execution_schedule


def _repeat_program(*, reverse_functions: bool = False) -> Program:
    tensor = TensorType(DType.F32, (1,), device=Device.CUDA)
    transform = Function(
        "transform",
        (Value("x", tensor),),
        ("out",),
        Region(
            (
                Op(
                    "constant",
                    (),
                    (Value("one", tensor),),
                    attributes(value=(1.0,)),
                ),
                Op("add", ("x", "one"), (Value("out", tensor),)),
            )
        ),
    )
    main = Function(
        "main",
        (Value("input", tensor),),
        ("result",),
        Region(
            (
                Op(
                    "call",
                    ("input",),
                    (Value("result", tensor),),
                    attributes(callee="transform", repeat=3),
                ),
            )
        ),
    )
    functions = (transform, main) if reverse_functions else (main, transform)
    return Program(functions=functions, entry="main")


class ExecutionScheduleTests(unittest.TestCase):
    def test_repeat_feedback_uses_unique_temporaries_and_one_static_constant(self) -> None:
        schedule = build_execution_schedule(_repeat_program())
        self.assertEqual(len(schedule.invocations), 4)
        self.assertEqual([item.function for item in schedule.invocations], ["main", "transform"] * 1 + ["transform", "transform"])
        self.assertEqual(len(schedule.ops), 3)
        constants = [item for item in schedule.values if item.storage == ValueStorage.CONSTANT]
        self.assertEqual(len(constants), 1)
        constant_id = constants[0].value_id
        input_id = schedule.entry_inputs[0][1]
        first, second, third = schedule.ops
        self.assertEqual(first.inputs, (input_id, constant_id))
        self.assertEqual(second.inputs, (first.outputs[0], constant_id))
        self.assertEqual(third.inputs, (second.outputs[0], constant_id))
        self.assertEqual(schedule.entry_outputs, (("result", third.outputs[0]),))
        self.assertEqual({item.site for item in schedule.ops}, {"transform:1"})

    def test_dump_is_independent_of_function_tuple_order(self) -> None:
        first = build_execution_schedule(_repeat_program())
        second = build_execution_schedule(_repeat_program(reverse_functions=True))
        self.assertEqual(dump_execution_schedule(first), dump_execution_schedule(second))

    def test_logical_constant_ref_is_shared_across_distinct_static_sites(self) -> None:
        tensor = TensorType(DType.F32, (1,), device=Device.CUDA)

        def transform(name: str) -> Function:
            return Function(
                name,
                (Value("x", tensor),),
                ("out",),
                Region(
                    (
                        Op(
                            "constant_ref",
                            (),
                            (Value("shared", tensor),),
                            attributes(namespace="model", name="shared"),
                        ),
                        Op("add", ("x", "shared"), (Value("out", tensor),)),
                    )
                ),
            )

        main = Function(
            "main",
            (Value("x", tensor),),
            ("out",),
            Region(
                (
                    Op(
                        "call",
                        ("x",),
                        (Value("first", tensor),),
                        attributes(callee="first", repeat=1),
                    ),
                    Op(
                        "call",
                        ("first",),
                        (Value("out", tensor),),
                        attributes(callee="second", repeat=1),
                    ),
                )
            ),
        )
        schedule = build_execution_schedule(
            Program((transform("first"), main, transform("second")), "main")
        )
        constants = [item for item in schedule.values if item.storage == ValueStorage.CONSTANT]
        self.assertEqual(len(constants), 1)
        self.assertEqual(constants[0].constant_identity, "model:shared")
        self.assertEqual(schedule.ops[0].inputs[1], schedule.ops[1].inputs[1])

    def test_state_reads_alias_dedicated_state_and_writes_remain_scheduled(self) -> None:
        tensor = TensorType(DType.BF16, (1, 2), device=Device.CUDA)
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
                                (Value("output", tensor),),
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
        self.assertEqual([item.opcode for item in schedule.ops], ["state_write", "state_read"])
        state_id = schedule.states[0][1]
        alias = schedule.values[schedule.entry_outputs[0][1]]
        self.assertEqual(alias.storage, ValueStorage.STATE_ALIAS)
        self.assertEqual(alias.alias_of, state_id)
        self.assertEqual(schedule.ops[0].inputs, (schedule.entry_inputs[0][1],))

    def test_nested_calls_preserve_parent_and_repeat_receipts(self) -> None:
        tensor = TensorType(DType.F32, (1,), device=Device.CUDA)
        leaf = Function(
            "leaf",
            (Value("x", tensor),),
            ("out",),
            Region((Op("relu", ("x",), (Value("out", tensor),)),)),
        )
        middle = Function(
            "middle",
            (Value("x", tensor),),
            ("out",),
            Region(
                (
                    Op(
                        "call",
                        ("x",),
                        (Value("out", tensor),),
                        attributes(callee="leaf", repeat=2),
                    ),
                )
            ),
        )
        main = Function(
            "main",
            (Value("x", tensor),),
            ("out",),
            Region(
                (
                    Op(
                        "call",
                        ("x",),
                        (Value("out", tensor),),
                        attributes(callee="middle", repeat=2),
                    ),
                )
            ),
        )
        schedule = build_execution_schedule(Program((leaf, main, middle), "main"))
        self.assertEqual(len(schedule.invocations), 7)
        self.assertEqual([item.function for item in schedule.invocations], ["main", "middle", "leaf", "leaf", "middle", "leaf", "leaf"])
        self.assertEqual([item.repeat_index for item in schedule.invocations], [0, 0, 0, 1, 1, 0, 1])
        self.assertEqual(len(schedule.ops), 4)
        for previous, current in zip(schedule.ops, schedule.ops[1:]):
            self.assertEqual(current.inputs, previous.outputs)

    def test_expansion_limit_is_shared_with_inventory(self) -> None:
        with self.assertRaisesRegex(ValidationError, "repeat exceeds"):
            build_execution_schedule(_repeat_program(), max_expanded_ops=2)


if __name__ == "__main__":
    unittest.main()
