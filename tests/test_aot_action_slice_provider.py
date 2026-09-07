from __future__ import annotations

import dataclasses
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.ir import (
    DType,
    Device,
    Function,
    Op,
    Program,
    Region,
    Tensor,
    TensorType,
    Value,
    execute,
)
from aginfer.lowering import (
    OperandAccess,
    build_execution_schedule,
    build_lowering_inventory,
    build_memory_plan,
    resolve_provider_capabilities,
)
from aginfer.providers import (
    ACTION_SLICE_PAYLOAD,
    ActionSlicePayload,
    ActionSliceProblem,
    ActionSliceValidationReceipt,
    dump_action_slice_partial_lowering,
    lower_action_slice_commands,
    make_action_slice_capabilities,
)
from aginfer.schema import CudaArch


MODULE_DIGEST = "c" * 64


def _program(*, consumer: bool = False, device: Device = Device.CUDA) -> Program:
    source = TensorType(DType.F32, (1, 50, 32), device=device)
    output = TensorType(DType.F32, (1, 50, 7), device=device)
    ops = [
        Op(
            "slice",
            ("padded",),
            (Value("action", output),),
            (("axis", 2), ("start", 0), ("stop", 7)),
        )
    ]
    outputs = ("action",)
    if consumer:
        ops.append(Op("relu", ("action",), (Value("post", output),)))
        outputs = ("post",)
    return Program(
        functions=(
            Function("main", (Value("padded", source),), outputs, Region(tuple(ops))),
        ),
        entry="main",
    )


def _receipt() -> ActionSliceValidationReceipt:
    item = build_lowering_inventory(_program()).ops[0]
    return ActionSliceValidationReceipt(
        ActionSliceProblem.from_inventory(item, target_arch=CudaArch.SM120),
        MODULE_DIGEST,
        181_000,
        12_080,
        12_080,
        13_000,
        2,
        True,
        True,
        True,
        True,
        True,
        0,
        0.002,
        0.0021,
        False,
    )


class AotActionSliceProviderTests(unittest.TestCase):
    def test_reference_copies_each_stride_32_row_not_a_contiguous_prefix(self) -> None:
        values = tuple(float(index) for index in range(50 * 32))
        result = execute(
            _program(device=Device.CPU),
            {"padded": Tensor.from_values(DType.F32, (1, 50, 32), values)},
        ).outputs[0].data
        expected = tuple(values[row * 32 + column] for row in range(50) for column in range(7))
        self.assertEqual(result, expected)
        self.assertNotEqual(result, values[:350])

    def test_problem_payload_and_target_are_exact(self) -> None:
        inventory = build_lowering_inventory(_program())
        item = inventory.ops[0]
        problem = ActionSliceProblem.from_inventory(item, target_arch=CudaArch.SM120)
        payload = ActionSlicePayload(problem, 181_000, MODULE_DIGEST)
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(ActionSlicePayload.from_bytes(encoded), payload)
        fields = list(ACTION_SLICE_PAYLOAD.unpack(encoded))
        fields[9] = 8
        with self.assertRaisesRegex(FormatError, "exact variant"):
            ActionSlicePayload.from_bytes(ACTION_SLICE_PAYLOAD.pack(*fields))
        with self.assertRaisesRegex(ValidationError, "exact envelope"):
            ActionSliceProblem.from_inventory(
                dataclasses.replace(item, attributes=(("axis", 2), ("start", 1), ("stop", 8))),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "requires SM120"):
            ActionSliceProblem.from_inventory(item, target_arch=CudaArch.SM110)

    def test_receipt_requires_bit_exact_capture_negative_and_racecheck(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_output_bit_exact", False, "bit-exact"),
            ("contiguous_prefix_negative_detected", False, "negative control"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("latency_ms", 0.0, "latency evidence"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_capability_and_terminal_lowering_are_deterministic(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_action_slice_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        self.assertTrue(
            resolve_provider_capabilities(
                inventory, capabilities, target_arch=CudaArch.SM120, strict=True
            ).complete
        )
        lowering = lower_action_slice_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(
            tuple(item.access for item in lowering.commands[0].command.operands),
            (OperandAccess.READ, OperandAccess.WRITE),
        )
        again = lower_action_slice_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_action_slice_partial_lowering(lowering),
            dump_action_slice_partial_lowering(again),
        )
        internal = _program(consumer=True)
        internal_inventory = build_lowering_inventory(internal)
        with self.assertRaisesRegex(ValidationError, "terminal entry output"):
            lower_action_slice_commands(
                build_execution_schedule(internal),
                internal_inventory,
                build_memory_plan(build_execution_schedule(internal)),
                receipt,
                target_arch=CudaArch.SM120,
            )


if __name__ == "__main__":
    unittest.main()
