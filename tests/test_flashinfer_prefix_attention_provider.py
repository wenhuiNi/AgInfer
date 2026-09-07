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
    FlashInferPrefixAttentionPayload,
    FlashInferPrefixAttentionProblem,
    FlashInferPrefixAttentionValidationReceipt,
    dump_flashinfer_attention_partial_lowering,
    lower_flashinfer_prefix_attention_commands,
    make_flashinfer_prefix_attention_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "7" * 64
PAD_MASK_DIGEST = "8" * 64


def _program() -> Program:
    pad = TensorType(DType.BOOL, (1, 968), device=Device.CUDA)
    mask_2d = TensorType(DType.BOOL, (1, 968, 968), device=Device.CUDA)
    mask_4d = TensorType(DType.BOOL, (1, 8, 968, 968), device=Device.CUDA)
    query = TensorType(DType.BF16, (1, 8, 968, 256), device=Device.CUDA)
    key_value = TensorType(DType.BF16, (1, 1, 968, 256), device=Device.CUDA)
    output_bshd = TensorType(DType.BF16, (1, 968, 8, 256), device=Device.CUDA)
    output_flat = TensorType(DType.BF16, (1, 968, 2048), device=Device.CUDA)
    return Program(
        functions=(
            Function(
                "main",
                (
                    Value("query", query),
                    Value("key", key_value),
                    Value("value", key_value),
                    Value("pad", pad),
                ),
                ("output",),
                Region(
                    (
                        Op(
                            "broadcast_in_dim",
                            ("pad",),
                            (Value("key_mask", mask_2d),),
                            (
                                ("broadcast_dimensions", (0, 2)),
                                ("shape", (1, 968, 968)),
                            ),
                        ),
                        Op(
                            "broadcast_in_dim",
                            ("pad",),
                            (Value("query_mask", mask_2d),),
                            (
                                ("broadcast_dimensions", (0, 1)),
                                ("shape", (1, 968, 968)),
                            ),
                        ),
                        Op(
                            "logical_and",
                            ("key_mask", "query_mask"),
                            (Value("mask_2d", mask_2d),),
                        ),
                        Op(
                            "broadcast_in_dim",
                            ("mask_2d",),
                            (Value("mask", mask_4d),),
                            (
                                ("broadcast_dimensions", (0, 2, 3)),
                                ("shape", (1, 8, 968, 968)),
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
                            (("shape", (1, 968, 2048)),),
                        ),
                    )
                ),
            ),
        ),
        entry="main",
    )


def _receipt() -> FlashInferPrefixAttentionValidationReceipt:
    return FlashInferPrefixAttentionValidationReceipt(
        problem=FlashInferPrefixAttentionProblem(CudaArch.SM120),
        implementation_sha256=IMPLEMENTATION_DIGEST,
        implementation_bytes=120_000,
        cuda_compiler_version=12_080,
        cuda_runtime_version=12_080,
        cuda_driver_version=13_000,
        pad_mask_sha256=PAD_MASK_DIGEST,
        valid_tokens=558,
        fully_masked_rows=410,
        normal_launches=2,
        normal_repeat_bit_exact=True,
        capture_replay_bit_exact=True,
        capture_matches_normal=True,
        full_output_compared=True,
        finite_mask_rows_compared=True,
        hard_mask_negative_detected=True,
        cosine=0.999999,
        max_abs=0.0001220703125,
        p99_abs=0.00006103515625,
        contains_ptx=False,
        fa3_custom_mask_supported=False,
        fa3_sm120_compiles=False,
    )


class FlashInferPrefixAttentionProviderTests(unittest.TestCase):
    def test_problem_and_payload_are_exact_and_fail_closed(self) -> None:
        inventory = build_lowering_inventory(_program())
        item = next(
            op for op in inventory.ops if op.opcode == "scaled_dot_product_attention"
        )
        problem = FlashInferPrefixAttentionProblem.from_inventory(
            item, target_arch=CudaArch.SM120
        )
        self.assertEqual(problem, FlashInferPrefixAttentionProblem(CudaArch.SM120))
        with self.assertRaisesRegex(ValidationError, "prefix envelope"):
            FlashInferPrefixAttentionProblem.from_inventory(
                dataclasses.replace(item, attributes=item.attributes[:-1]),
                target_arch=CudaArch.SM120,
            )

        payload = FlashInferPrefixAttentionPayload(problem)
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(FlashInferPrefixAttentionPayload.from_bytes(encoded), payload)
        fields = list(FLASHINFER_ATTENTION_PAYLOAD.unpack(encoded))
        fields[16] = 120
        with self.assertRaisesRegex(FormatError, "exact variant"):
            FlashInferPrefixAttentionPayload.from_bytes(
                FLASHINFER_ATTENTION_PAYLOAD.pack(*fields)
            )

    def test_receipt_requires_finite_rows_negative_control_and_fa3_audit(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("fully_masked_rows", 0, "must be positive"),
            ("finite_mask_rows_compared", False, "finite-mask"),
            ("hard_mask_negative_detected", False, "negative control"),
            ("fa3_custom_mask_supported", True, "custom-mask refusal"),
            ("fa3_sm120_compiles", True, "compile blocker"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_region_lowering_fuses_four_mask_ops_attention_and_transpose(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_flashinfer_prefix_attention_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        lowering = lower_flashinfer_prefix_attention_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(lowering.fused_execution_indices, (0, 1, 2, 3, 4, 5))
        self.assertEqual(lowering.elided_execution_indices, (6,))
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
            "entry_input:pad",
        )
        self.assertEqual(
            schedule.values[command.operands[4].value_id].type.shape,
            (1, 968, 8, 256),
        )
        again = lower_flashinfer_prefix_attention_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_flashinfer_attention_partial_lowering(lowering),
            dump_flashinfer_attention_partial_lowering(again),
        )


if __name__ == "__main__":
    unittest.main()
