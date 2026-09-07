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
    ADAPTIVE_RMS_NORM_PAYLOAD,
    AdaptiveRmsNormPayload,
    AdaptiveRmsNormProblem,
    AdaptiveRmsNormValidationReceipt,
    dump_adaptive_rms_norm_partial_lowering,
    lower_adaptive_rms_norm_commands,
    make_adaptive_rms_norm_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "9" * 64


def _program(*, expose_intermediate: bool = False) -> Program:
    bf16_tensor = TensorType(DType.BF16, (1, 50, 1024), device=Device.CUDA)
    f32_tensor = TensorType(DType.F32, (1, 50, 1024), device=Device.CUDA)
    f32_vector = TensorType(DType.F32, (1, 1024), device=Device.CUDA)
    f32_weight = TensorType(DType.F32, (1024,), device=Device.CUDA)
    modulation_type = TensorType(DType.F32, (1, 3072), device=Device.CUDA)
    scalar = TensorType(DType.F32, (1,), device=Device.CUDA)
    ops = [
        Op("constant", (), (Value("one", scalar),), (("value", (1.0,)),)),
        Op(
            "broadcast_in_dim",
            ("one",),
            (Value("ones_weight", f32_weight),),
            (("broadcast_dimensions", (0,)), ("shape", (1024,))),
        ),
        Op(
            "broadcast_in_dim",
            ("one",),
            (Value("ones_tokens", f32_tensor),),
            (("broadcast_dimensions", (0,)), ("shape", (1, 50, 1024))),
        ),
        Op(
            "cast",
            ("hidden",),
            (Value("hidden_f32", f32_tensor),),
            (("dtype", "f32"),),
        ),
        Op(
            "rms_norm",
            ("hidden_f32", "ones_weight"),
            (Value("rms", f32_tensor),),
            (("epsilon", 1.0e-6),),
        ),
    ]
    for name, start in (("scale", 0), ("shift", 1024), ("gate", 2048)):
        ops.extend(
            (
                Op(
                    "slice",
                    ("modulation",),
                    (Value(f"{name}_flat", f32_vector),),
                    (("axis", 1), ("start", start), ("stop", start + 1024)),
                ),
                Op(
                    "broadcast_in_dim",
                    (f"{name}_flat",),
                    (Value(name, f32_tensor),),
                    (
                        ("broadcast_dimensions", (0, 2)),
                        ("shape", (1, 50, 1024)),
                    ),
                ),
            )
        )
    ops.extend(
        (
            Op("add", ("ones_tokens", "scale"), (Value("scale_one", f32_tensor),)),
            Op("mul", ("rms", "scale_one"), (Value("scaled", f32_tensor),)),
            Op("add", ("scaled", "shift"), (Value("shifted", f32_tensor),)),
            Op(
                "cast",
                ("shifted",),
                (Value("normalized", bf16_tensor),),
                (("dtype", "bf16"),),
            ),
            Op(
                "cast",
                ("gate",),
                (Value("gate_bf16", bf16_tensor),),
                (("dtype", "bf16"),),
            ),
        )
    )
    outputs = ["normalized", "gate_bf16"]
    if expose_intermediate:
        outputs.append("scale_flat")
    return Program(
        functions=(
            Function(
                "main",
                (Value("hidden", bf16_tensor), Value("modulation", modulation_type)),
                tuple(outputs),
                Region(tuple(ops)),
            ),
        ),
        entry="main",
    )


def _receipt() -> AdaptiveRmsNormValidationReceipt:
    return AdaptiveRmsNormValidationReceipt(
        AdaptiveRmsNormProblem(CudaArch.SM120),
        IMPLEMENTATION_DIGEST,
        96_000,
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
        0.999999,
        0.0078125,
        0.0,
        False,
    )


class AotAdaptiveRmsNormProviderTests(unittest.TestCase):
    def test_payload_is_fixed_and_fail_closed(self) -> None:
        payload = AdaptiveRmsNormPayload(
            AdaptiveRmsNormProblem(CudaArch.SM120),
            96_000,
            IMPLEMENTATION_DIGEST,
        )
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(AdaptiveRmsNormPayload.from_bytes(encoded), payload)
        fields = list(ADAPTIVE_RMS_NORM_PAYLOAD.unpack(encoded))
        fields[12] += 1
        with self.assertRaisesRegex(FormatError, "exact variant"):
            AdaptiveRmsNormPayload.from_bytes(
                ADAPTIVE_RMS_NORM_PAYLOAD.pack(*fields)
            )
        with self.assertRaisesRegex(FormatError, "fixed size"):
            AdaptiveRmsNormPayload.from_bytes(encoded[:-1])

    def test_receipt_requires_full_outputs_capture_racecheck_and_no_ptx(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_outputs_compared", False, "full-output"),
            ("gate_bit_exact", False, "gate output"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("cosine", 0.9, "correctness gate"),
            ("max_abs", 1.0, "correctness gate"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_exact_region_lowering_replaces_thirteen_ops_and_shared_ones(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_adaptive_rms_norm_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        lowering = lower_adaptive_rms_norm_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(len(lowering.commands[0].fused_execution_indices), 13)
        self.assertEqual(len(lowering.fused_execution_indices), 15)
        self.assertEqual(
            tuple(operand.access for operand in lowering.commands[0].command.operands),
            (
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.WRITE,
                OperandAccess.WRITE,
            ),
        )
        again = lower_adaptive_rms_norm_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_adaptive_rms_norm_partial_lowering(lowering),
            dump_adaptive_rms_norm_partial_lowering(again),
        )

    def test_region_refuses_public_intermediate_and_wrong_target(self) -> None:
        program = _program(expose_intermediate=True)
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        with self.assertRaisesRegex(ValidationError, "public output"):
            lower_adaptive_rms_norm_commands(
                schedule, inventory, memory, _receipt(), target_arch=CudaArch.SM120
            )
        anchor = next(item for item in inventory.ops if item.opcode == "rms_norm")
        with self.assertRaisesRegex(ValidationError, "requires SM120"):
            AdaptiveRmsNormProblem.from_inventory(anchor, target_arch=CudaArch.SM110)


if __name__ == "__main__":
    unittest.main()
