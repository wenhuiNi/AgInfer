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
from aginfer.lowering import (
    LoweringKind,
    RequirementStatus,
    build_lowering_inventory,
    dump_lowering_inventory,
)


def _call_program(*, functions_reversed: bool = False) -> Program:
    tensor = TensorType(DType.F32, (1, 2), device=Device.CUDA)
    weight = TensorType(DType.F32, (2, 2), device=Device.CUDA)
    bias = TensorType(DType.F32, (2,), device=Device.CUDA)
    leaf = Function(
        "leaf",
        (Value("x", tensor),),
        ("out",),
        Region(
            (
                Op(
                    "constant_ref",
                    (),
                    (Value("weight", weight),),
                    attributes(namespace="model", name="block.weight"),
                ),
                Op(
                    "constant_ref",
                    (),
                    (Value("bias", bias),),
                    attributes(namespace="model", name="block.bias"),
                ),
                Op("linear", ("x", "weight", "bias"), (Value("linear", tensor),)),
                Op("relu", ("linear",), (Value("out", tensor),)),
            )
        ),
    )
    main = Function(
        "main",
        (Value("input", tensor),),
        ("output",),
        Region(
            (
                Op(
                    "call",
                    ("input",),
                    (Value("output", tensor),),
                    attributes(callee="leaf", repeat=2),
                ),
            )
        ),
    )
    functions = (leaf, main) if functions_reversed else (main, leaf)
    return Program(functions=functions, entry="main")


class LoweringInventoryTests(unittest.TestCase):
    def test_inventory_is_stable_call_expanded_and_keeps_constant_identity(self) -> None:
        first = build_lowering_inventory(_call_program())
        second = build_lowering_inventory(_call_program(functions_reversed=True))
        self.assertEqual(dump_lowering_inventory(first), dump_lowering_inventory(second))
        self.assertEqual(first.function_invocations, (("leaf", 2), ("main", 1)))
        self.assertEqual(
            tuple(item.site_id for item in first.execution_order),
            ("leaf:2", "leaf:3", "leaf:2", "leaf:3"),
        )

        by_site = {item.site_id: item for item in first.ops}
        self.assertEqual(by_site["leaf:0"].constant_identity, "model:block.weight")
        self.assertEqual(by_site["leaf:0"].status, RequirementStatus.META)
        self.assertEqual(by_site["leaf:2"].executions, 2)
        self.assertEqual(by_site["leaf:2"].lowering_kind, LoweringKind.GEMM)
        self.assertEqual(by_site["leaf:3"].lowering_kind, LoweringKind.AOT_CUDA)
        self.assertEqual(by_site["main:0"].lowering_kind, LoweringKind.BUILTIN_META)
        summary = first.to_dict()["summary"]  # type: ignore[assignment]
        self.assertEqual(summary["expanded_all_ops"], 9)
        self.assertEqual(summary["expanded_runtime_ops"], 4)

    def test_state_ops_are_explicit_memory_requirements(self) -> None:
        tensor = TensorType(DType.F32, (1,), device=Device.CUDA)
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
        inventory = build_lowering_inventory(program)
        self.assertEqual(
            [(item.lowering_kind, item.status) for item in inventory.ops],
            [
                (LoweringKind.MEMORY, RequirementStatus.REQUIRED),
                (LoweringKind.MEMORY, RequirementStatus.REQUIRED),
            ],
        )
        self.assertEqual(tuple(item.site_id for item in inventory.execution_order), ("main:0", "main:1"))

    def test_cpu_and_overlapping_conv_are_reported_as_blocked(self) -> None:
        cpu = TensorType(DType.F32, (1,), device=Device.CPU)
        cpu_program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", cpu),),
                    ("out",),
                    Region((Op("relu", ("x",), (Value("out", cpu),)),)),
                ),
            ),
            entry="main",
        )
        cpu_item = build_lowering_inventory(cpu_program).ops[0]
        self.assertEqual(cpu_item.lowering_kind, LoweringKind.BLOCKED)
        self.assertIn("CUDA", cpu_item.blocker or "")

        source = TensorType(DType.F32, (1, 1, 4, 4), device=Device.CUDA)
        weight = TensorType(DType.F32, (2, 1, 2, 2), device=Device.CUDA)
        bias = TensorType(DType.F32, (2,), device=Device.CUDA)
        output = TensorType(DType.F32, (1, 2, 3, 3), device=Device.CUDA)
        conv_program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", source), Value("w", weight), Value("b", bias)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "conv2d",
                                ("x", "w", "b"),
                                (Value("out", output),),
                                attributes(pads=(0, 0, 0, 0), strides=(1, 1)),
                            ),
                        )
                    ),
                ),
            ),
            entry="main",
        )
        conv_item = build_lowering_inventory(conv_program).ops[0]
        self.assertEqual(conv_item.status, RequirementStatus.BLOCKED)
        self.assertIn("overlapping", conv_item.blocker or "")

    def test_non_overlapping_patch_projection_maps_to_gemm(self) -> None:
        source = TensorType(DType.F32, (1, 1, 4, 4), device=Device.CUDA)
        weight = TensorType(DType.F32, (2, 1, 2, 2), device=Device.CUDA)
        bias = TensorType(DType.F32, (2,), device=Device.CUDA)
        output = TensorType(DType.F32, (1, 2, 2, 2), device=Device.CUDA)
        program = Program(
            functions=(
                Function(
                    "main",
                    (Value("x", source), Value("w", weight), Value("b", bias)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "conv2d",
                                ("x", "w", "b"),
                                (Value("out", output),),
                                attributes(pads=(0, 0, 0, 0), strides=(2, 2)),
                            ),
                        )
                    ),
                ),
            ),
            entry="main",
        )
        item = build_lowering_inventory(program).ops[0]
        self.assertEqual((item.lowering_kind, item.status), (LoweringKind.GEMM, RequirementStatus.REQUIRED))

    def test_expansion_limit_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "repeat exceeds"):
            build_lowering_inventory(_call_program(), max_expanded_ops=1)
        with self.assertRaisesRegex(ValidationError, "positive integer"):
            build_lowering_inventory(_call_program(), max_expanded_ops=True)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
