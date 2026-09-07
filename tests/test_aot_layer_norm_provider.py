from __future__ import annotations

import dataclasses
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value
from aginfer.lowering import (
    OperandAccess,
    build_execution_schedule,
    build_lowering_inventory,
    build_memory_plan,
)
from aginfer.providers import (
    LAYER_NORM_PAYLOAD,
    LayerNormPayload,
    LayerNormProblem,
    LayerNormValidationReceipt,
    dump_layer_norm_partial_lowering,
    lower_layer_norm_commands,
    make_layer_norm_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "6" * 64


def _program() -> Program:
    tensor = TensorType(DType.F32, (1, 256, 1152), device=Device.CUDA)
    parameter = TensorType(DType.F32, (1152,), device=Device.CUDA)
    return Program(
        functions=(
            Function(
                "main",
                (
                    Value("input", tensor),
                    Value("weight", parameter),
                    Value("bias", parameter),
                ),
                ("output",),
                Region(
                    (
                        Op(
                            "layer_norm",
                            ("input", "weight", "bias"),
                            (Value("output", tensor),),
                            (("epsilon", 1.0e-6),),
                        ),
                    )
                ),
            ),
        ),
        entry="main",
    )


def _receipt() -> LayerNormValidationReceipt:
    return LayerNormValidationReceipt(
        LayerNormProblem(CudaArch.SM120),
        IMPLEMENTATION_DIGEST,
        64_000,
        12_080,
        12_080,
        13_000,
        2,
        True,
        True,
        True,
        True,
        0,
        0.9999999,
        0.00001,
        0.000005,
        False,
    )


class AotLayerNormProviderTests(unittest.TestCase):
    def test_problem_accepts_only_exact_layer_norm(self) -> None:
        inventory = build_lowering_inventory(_program())
        item = next(op for op in inventory.ops if op.opcode == "layer_norm")
        self.assertEqual(
            LayerNormProblem.from_inventory(item, target_arch=CudaArch.SM120),
            LayerNormProblem(CudaArch.SM120),
        )
        with self.assertRaisesRegex(ValidationError, "rows256/width1152"):
            LayerNormProblem.from_inventory(
                dataclasses.replace(item, attributes=(("epsilon", 1.0e-5),)),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "requires SM120"):
            LayerNormProblem.from_inventory(item, target_arch=CudaArch.SM110)

    def test_payload_is_fixed_and_fail_closed(self) -> None:
        payload = LayerNormPayload(
            LayerNormProblem(CudaArch.SM120), 64_000, IMPLEMENTATION_DIGEST
        )
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(LayerNormPayload.from_bytes(encoded), payload)
        fields = list(LAYER_NORM_PAYLOAD.unpack(encoded))
        fields[9] = 257
        with self.assertRaisesRegex(FormatError, "exact variant"):
            LayerNormPayload.from_bytes(LAYER_NORM_PAYLOAD.pack(*fields))
        with self.assertRaisesRegex(FormatError, "fixed size"):
            LayerNormPayload.from_bytes(encoded[:-1])

    def test_receipt_requires_correctness_capture_racecheck_and_no_ptx(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("cosine", 0.9, "correctness gate"),
            ("max_abs", 1.0, "correctness gate"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_exact_lowering_is_deterministic(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_layer_norm_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        lowering = lower_layer_norm_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 1)
        command = lowering.commands[0].command
        self.assertEqual(
            tuple(item.access for item in command.operands),
            (
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.WRITE,
            ),
        )
        again = lower_layer_norm_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_layer_norm_partial_lowering(lowering),
            dump_layer_norm_partial_lowering(again),
        )


if __name__ == "__main__":
    unittest.main()
