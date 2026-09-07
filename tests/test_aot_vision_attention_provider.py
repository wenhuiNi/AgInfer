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
    VISION_ATTENTION_PAYLOAD,
    VisionAttentionPayload,
    VisionAttentionProblem,
    VisionAttentionValidationReceipt,
    dump_vision_attention_partial_lowering,
    lower_vision_attention_commands,
    make_vision_attention_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "5" * 64


def _program(*, extra_query_consumer: bool = False) -> Program:
    bshd = TensorType(DType.F32, (1, 256, 16, 72), device=Device.CUDA)
    bhsd = TensorType(DType.F32, (1, 16, 256, 72), device=Device.CUDA)
    mask_scalar = TensorType(DType.BOOL, (1,), device=Device.CUDA)
    mask = TensorType(DType.BOOL, (1, 16, 256, 256), device=Device.CUDA)
    flat = TensorType(DType.F32, (1, 256, 1152), device=Device.CUDA)
    ops = [
        Op(
            "transpose",
            (name,),
            (Value(f"{name}_bhsd", bhsd),),
            (("permutation", (0, 2, 1, 3)),),
        )
        for name in ("query", "key", "value")
    ]
    ops.extend(
        (
            Op("constant", (), (Value("mask_scalar", mask_scalar),), (("value", (True,)),)),
            Op(
                "broadcast_in_dim",
                ("mask_scalar",),
                (Value("mask", mask),),
                (
                    ("broadcast_dimensions", (0,)),
                    ("shape", (1, 16, 256, 256)),
                ),
            ),
        )
    )
    if extra_query_consumer:
        ops.append(
            Op(
                "add",
                ("query_bhsd", "query_bhsd"),
                (Value("query_extra", bhsd),),
            )
        )
    ops.extend(
        (
            Op(
                "scaled_dot_product_attention",
                ("query_bhsd", "key_bhsd", "value_bhsd", "mask"),
                (Value("attention", bhsd),),
                (
                    ("kv_group_size", 1),
                    ("mask_fill", "dtype_min"),
                    ("scale", 1.0 / (72.0**0.5)),
                ),
            ),
            Op(
                "transpose",
                ("attention",),
                (Value("attention_bshd", bshd),),
                (("permutation", (0, 2, 1, 3)),),
            ),
            Op(
                "reshape",
                ("attention_bshd",),
                (Value("output", flat),),
                (("shape", (1, 256, 1152)),),
            ),
        )
    )
    return Program(
        functions=(
            Function(
                "main",
                (Value("query", bshd), Value("key", bshd), Value("value", bshd)),
                ("output",),
                Region(tuple(ops)),
            ),
        ),
        entry="main",
    )


def _receipt() -> VisionAttentionValidationReceipt:
    return VisionAttentionValidationReceipt(
        VisionAttentionProblem(CudaArch.SM120),
        IMPLEMENTATION_DIGEST,
        48_608,
        12_080,
        12_080,
        13_000,
        2,
        True,
        True,
        True,
        True,
        True,
        0.9999999,
        0.00001,
        0.000005,
        False,
    )


class AotVisionAttentionProviderTests(unittest.TestCase):
    def test_problem_accepts_only_exact_vision_attention(self) -> None:
        inventory = build_lowering_inventory(_program())
        item = next(op for op in inventory.ops if op.opcode == "scaled_dot_product_attention")
        self.assertEqual(
            VisionAttentionProblem.from_inventory(item, target_arch=CudaArch.SM120),
            VisionAttentionProblem(CudaArch.SM120),
        )
        with self.assertRaisesRegex(ValidationError, "head-dim 72"):
            VisionAttentionProblem.from_inventory(
                dataclasses.replace(item, attributes=item.attributes[:-1]),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "requires SM120"):
            VisionAttentionProblem.from_inventory(item, target_arch=CudaArch.SM110)

    def test_payload_is_fixed_and_fail_closed(self) -> None:
        payload = VisionAttentionPayload(
            VisionAttentionProblem(CudaArch.SM120), 48_608, IMPLEMENTATION_DIGEST
        )
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(VisionAttentionPayload.from_bytes(encoded), payload)
        fields = list(VISION_ATTENTION_PAYLOAD.unpack(encoded))
        fields[11] = 255
        with self.assertRaisesRegex(FormatError, "exact variant"):
            VisionAttentionPayload.from_bytes(VISION_ATTENTION_PAYLOAD.pack(*fields))
        with self.assertRaisesRegex(FormatError, "fixed size"):
            VisionAttentionPayload.from_bytes(encoded[:-1])

    def test_receipt_requires_correctness_capture_false_mask_and_no_ptx(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("false_mask_uniform_verified", False, "false-mask"),
            ("cosine", 0.9, "correctness gate"),
            ("max_abs", 1.0, "correctness gate"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_region_lowering_fuses_all_transposes_mask_and_attention(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_vision_attention_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        lowering = lower_vision_attention_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(lowering.fused_execution_indices, (0, 1, 2, 3, 4, 5))
        self.assertEqual(lowering.elided_execution_indices, (6,))
        self.assertEqual(lowering.commands[0].fused_execution_indices, (0, 1, 2, 4, 5))
        command = lowering.commands[0].command
        self.assertEqual(
            tuple(item.access for item in command.operands),
            (
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.WRITE,
            ),
        )
        self.assertEqual(schedule.values[command.operands[0].value_id].type.shape, (1, 256, 16, 72))
        self.assertEqual(schedule.values[command.operands[3].value_id].type.shape, (1,))
        again = lower_vision_attention_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_vision_attention_partial_lowering(lowering),
            dump_vision_attention_partial_lowering(again),
        )

    def test_region_lowering_rejects_extra_transpose_consumer(self) -> None:
        program = _program(extra_query_consumer=True)
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        with self.assertRaisesRegex(ValidationError, "exclusive exact"):
            lower_vision_attention_commands(
                schedule,
                inventory,
                memory_plan,
                _receipt(),
                target_arch=CudaArch.SM120,
            )


if __name__ == "__main__":
    unittest.main()
