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
    RMS_NORM_PAYLOAD,
    RmsNormPayload,
    RmsNormProblem,
    RmsNormValidationReceipt,
    RmsNormVariant,
    dump_rms_norm_partial_lowering,
    lower_rms_norm_commands,
    make_rms_norm_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "7" * 64


def _program() -> Program:
    f32_tensor = TensorType(DType.F32, (1, 50, 1024), device=Device.CUDA)
    bf16_tensor = TensorType(DType.BF16, (1, 968, 2048), device=Device.CUDA)
    f32_weight = TensorType(DType.F32, (1024,), device=Device.CUDA)
    bf16_weight = TensorType(DType.F32, (2048,), device=Device.CUDA)
    return Program(
        functions=(
            Function(
                "main",
                (
                    Value("f32_input", f32_tensor),
                    Value("f32_weight", f32_weight),
                    Value("bf16_input", bf16_tensor),
                    Value("bf16_weight", bf16_weight),
                ),
                ("f32_output", "bf16_output"),
                Region(
                    (
                        Op(
                            "rms_norm",
                            ("f32_input", "f32_weight"),
                            (Value("f32_output", f32_tensor),),
                            (("epsilon", 1.0e-6),),
                        ),
                        Op(
                            "rms_norm",
                            ("bf16_input", "bf16_weight"),
                            (Value("bf16_output", bf16_tensor),),
                            (("epsilon", 1.0e-6),),
                        ),
                    )
                ),
            ),
        ),
        entry="main",
    )


def _receipt(variant: RmsNormVariant) -> RmsNormValidationReceipt:
    bf16 = variant == RmsNormVariant.BF16_ROWS968_WIDTH2048
    return RmsNormValidationReceipt(
        RmsNormProblem(CudaArch.SM120, variant),
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
        0.999999,
        0.005 if bf16 else 0.00001,
        0.002 if bf16 else 0.000005,
        False,
    )


def _receipts() -> tuple[RmsNormValidationReceipt, ...]:
    return tuple(_receipt(variant) for variant in RmsNormVariant)


class AotRmsNormProviderTests(unittest.TestCase):
    def test_problem_accepts_only_two_exact_envelopes(self) -> None:
        inventory = build_lowering_inventory(_program())
        items = tuple(op for op in inventory.ops if op.opcode == "rms_norm")
        self.assertEqual(
            {RmsNormProblem.from_inventory(item, target_arch=CudaArch.SM120).variant for item in items},
            set(RmsNormVariant),
        )
        with self.assertRaisesRegex(ValidationError, "exact envelopes"):
            RmsNormProblem.from_inventory(
                dataclasses.replace(items[0], attributes=(("epsilon", 1.0e-5),)),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "require SM120"):
            RmsNormProblem.from_inventory(items[0], target_arch=CudaArch.SM110)

    def test_payload_is_fixed_and_fail_closed_for_both_variants(self) -> None:
        for variant in RmsNormVariant:
            with self.subTest(variant=variant):
                payload = RmsNormPayload(
                    RmsNormProblem(CudaArch.SM120, variant),
                    64_000,
                    IMPLEMENTATION_DIGEST,
                )
                encoded = payload.to_bytes()
                self.assertEqual(len(encoded), 192)
                self.assertEqual(RmsNormPayload.from_bytes(encoded), payload)
                fields = list(RMS_NORM_PAYLOAD.unpack(encoded))
                fields[10] += 1
                with self.assertRaisesRegex(FormatError, "exact variants"):
                    RmsNormPayload.from_bytes(RMS_NORM_PAYLOAD.pack(*fields))
                with self.assertRaisesRegex(FormatError, "fixed size"):
                    RmsNormPayload.from_bytes(encoded[:-1])

    def test_receipt_requires_correctness_capture_racecheck_and_no_ptx(self) -> None:
        receipt = _receipt(RmsNormVariant.BF16_ROWS968_WIDTH2048)
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
        receipts = _receipts()
        capabilities = make_rms_norm_capabilities(
            inventory, receipts, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 2)
        lowering = lower_rms_norm_commands(
            schedule, inventory, memory_plan, receipts, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 2)
        for item in lowering.commands:
            self.assertEqual(
                tuple(operand.access for operand in item.command.operands),
                (OperandAccess.READ, OperandAccess.READ, OperandAccess.WRITE),
            )
        again = lower_rms_norm_commands(
            schedule, inventory, memory_plan, receipts, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_rms_norm_partial_lowering(lowering),
            dump_rms_norm_partial_lowering(again),
        )

    def test_receipt_tuple_must_cover_each_declared_problem_once(self) -> None:
        inventory = build_lowering_inventory(_program())
        receipt = _receipt(RmsNormVariant.F32_ROWS50_WIDTH1024)
        with self.assertRaisesRegex(ValidationError, "duplicate problem"):
            make_rms_norm_capabilities(
                inventory, (receipt, receipt), target_arch=CudaArch.SM120
            )


if __name__ == "__main__":
    unittest.main()
