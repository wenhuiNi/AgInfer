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
    FLASHINFER_ATTENTION_PAYLOAD,
    FlashInferAttentionPayload,
    FlashInferAttentionProblem,
    FlashInferAttentionValidationReceipt,
    dump_flashinfer_attention_partial_lowering,
    lower_flashinfer_attention_commands,
    make_flashinfer_attention_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "3" * 64


def _program() -> Program:
    query = TensorType(DType.BF16, (1, 8, 50, 256), device=Device.CUDA)
    key_value = TensorType(DType.BF16, (1, 1, 1018, 256), device=Device.CUDA)
    mask_2d = TensorType(DType.BOOL, (1, 50, 1018), device=Device.CUDA)
    mask_4d = TensorType(DType.BOOL, (1, 8, 50, 1018), device=Device.CUDA)
    output_bshd = TensorType(DType.BF16, (1, 50, 8, 256), device=Device.CUDA)
    output_flat = TensorType(DType.BF16, (1, 50, 2048), device=Device.CUDA)
    return Program(
        functions=(
            Function(
                "main",
                (
                    Value("query", query),
                    Value("key", key_value),
                    Value("value", key_value),
                    Value("mask_2d", mask_2d),
                ),
                ("output",),
                Region(
                    (
                        Op(
                            "broadcast_in_dim",
                            ("mask_2d",),
                            (Value("mask", mask_4d),),
                            (
                                ("broadcast_dimensions", (0, 2, 3)),
                                ("shape", (1, 8, 50, 1018)),
                            ),
                        ),
                        Op(
                            "scaled_dot_product_attention",
                            ("query", "key", "value", "mask"),
                            (Value("attention", query),),
                            (
                                ("kv_group_size", 8),
                                ("mask_fill", "dtype_min"),
                                ("scale", 0.0625),
                            ),
                        ),
                        Op(
                            "transpose",
                            ("attention",),
                            (Value("bshd", output_bshd),),
                            (("permutation", (0, 2, 1, 3)),),
                        ),
                        Op(
                            "reshape",
                            ("bshd",),
                            (Value("output", output_flat),),
                            (("shape", (1, 50, 2048)),),
                        ),
                    )
                ),
            ),
        ),
        entry="main",
    )


def _receipt() -> FlashInferAttentionValidationReceipt:
    return FlashInferAttentionValidationReceipt(
        FlashInferAttentionProblem(CudaArch.SM120),
        IMPLEMENTATION_DIGEST,
        100_000,
        12_080,
        12_080,
        13_000,
        2,
        True,
        True,
        True,
        True,
        0,
        0.9999988,
        0.0001220703125,
        0.00006103515625,
        False,
    )


class FlashInferAttentionProviderTests(unittest.TestCase):
    def test_problem_accepts_only_exact_denoise_attention(self) -> None:
        inventory = build_lowering_inventory(_program())
        item = next(op for op in inventory.ops if op.opcode == "scaled_dot_product_attention")
        self.assertEqual(
            FlashInferAttentionProblem.from_inventory(item, target_arch=CudaArch.SM120),
            FlashInferAttentionProblem(CudaArch.SM120),
        )
        with self.assertRaisesRegex(ValidationError, "denoise envelope"):
            FlashInferAttentionProblem.from_inventory(
                dataclasses.replace(item, attributes=item.attributes[:-1]),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "requires SM120"):
            FlashInferAttentionProblem.from_inventory(item, target_arch=CudaArch.SM110)

    def test_payload_is_fixed_and_fail_closed(self) -> None:
        payload = FlashInferAttentionPayload(
            FlashInferAttentionProblem(CudaArch.SM120)
        )
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(FlashInferAttentionPayload.from_bytes(encoded), payload)
        fields = list(FLASHINFER_ATTENTION_PAYLOAD.unpack(encoded))
        fields[14] = 51
        with self.assertRaisesRegex(FormatError, "exact variant"):
            FlashInferAttentionPayload.from_bytes(
                FLASHINFER_ATTENTION_PAYLOAD.pack(*fields)
            )
        with self.assertRaisesRegex(FormatError, "fixed size"):
            FlashInferAttentionPayload.from_bytes(encoded[:-1])

    def test_receipt_requires_real_correctness_capture_and_no_ptx(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("fully_masked_rows", 1, "no fully-masked"),
            ("cosine", 0.9, "correctness gate"),
            ("max_abs", 1.0, "correctness gate"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_region_lowering_fuses_mask_broadcast_attention_and_transpose(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_flashinfer_attention_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        lowering = lower_flashinfer_attention_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(lowering.fused_execution_indices, (0, 1, 2))
        self.assertEqual(lowering.elided_execution_indices, (3,))
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
        self.assertEqual(
            schedule.values[command.operands[3].value_id].debug_name,
            "entry_input:mask_2d",
        )
        self.assertEqual(
            schedule.values[command.operands[4].value_id].type.shape,
            (1, 50, 8, 256),
        )
        again = lower_flashinfer_attention_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_flashinfer_attention_partial_lowering(lowering),
            dump_flashinfer_attention_partial_lowering(again),
        )


if __name__ == "__main__":
    unittest.main()
