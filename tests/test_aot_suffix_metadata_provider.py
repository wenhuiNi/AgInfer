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
    SUFFIX_METADATA_PAYLOAD,
    SuffixMetadataPayload,
    SuffixMetadataProblem,
    SuffixMetadataValidationReceipt,
    dump_suffix_metadata_partial_lowering,
    lower_suffix_metadata_commands,
    make_suffix_metadata_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "c" * 64


def _program(
    *,
    suffix_value: bool = True,
    negative_one: int = -1,
    expose_intermediate: bool = False,
    extra_mask_consumer: bool = False,
) -> Program:
    pad = TensorType(DType.BOOL, (1, 968), device=Device.CUDA)
    bool_scalar = TensorType(DType.BOOL, (1,), device=Device.CUDA)
    bool_50 = TensorType(DType.BOOL, (1, 50), device=Device.CUDA)
    prefix_mask = TensorType(DType.BOOL, (1, 50, 968), device=Device.CUDA)
    suffix_mask = TensorType(DType.BOOL, (1, 50, 50), device=Device.CUDA)
    mask_2d = TensorType(DType.BOOL, (1, 50, 1018), device=Device.CUDA)
    mask_4d = TensorType(DType.BOOL, (1, 8, 50, 1018), device=Device.CUDA)
    i32_scalar = TensorType(DType.I32, (1,), device=Device.CUDA)
    i32_pad = TensorType(DType.I32, (1, 968), device=Device.CUDA)
    positions = TensorType(DType.I32, (1, 50), device=Device.CUDA)
    query = TensorType(DType.BF16, (1, 8, 50, 256), device=Device.CUDA)
    key_value = TensorType(DType.BF16, (1, 1, 1018, 256), device=Device.CUDA)
    ops = [
        Op(
            "constant",
            (),
            (Value("suffix_true", bool_scalar),),
            (("value", (suffix_value,)),),
        ),
        Op(
            "constant",
            (),
            (Value("negative_one", i32_scalar),),
            (("value", (negative_one,)),),
        ),
        Op(
            "broadcast_in_dim",
            ("suffix_true",),
            (Value("suffix_pad", bool_50),),
            (("broadcast_dimensions", (0,)), ("shape", (1, 50))),
        ),
        Op(
            "broadcast_in_dim",
            ("prefix_pad",),
            (Value("prefix_mask", prefix_mask),),
            (("broadcast_dimensions", (0, 2)), ("shape", (1, 50, 968))),
        ),
        Op(
            "broadcast_in_dim",
            ("suffix_true",),
            (Value("suffix_mask", suffix_mask),),
            (("broadcast_dimensions", (0,)), ("shape", (1, 50, 50))),
        ),
        Op(
            "concat",
            ("prefix_mask", "suffix_mask"),
            (Value("mask_2d", mask_2d),),
            (("axis", 2),),
        ),
        Op(
            "broadcast_in_dim",
            ("mask_2d",),
            (Value("mask_4d", mask_4d),),
            (("broadcast_dimensions", (0, 2, 3)), ("shape", (1, 8, 50, 1018))),
        ),
        Op("cast", ("prefix_pad",), (Value("pad_i32", i32_pad),), (("dtype", "i32"),)),
        Op(
            "reduce_sum",
            ("pad_i32",),
            (Value("offset", i32_scalar),),
            (("axes", (1,)), ("keepdims", False)),
        ),
        Op(
            "broadcast_in_dim",
            ("offset",),
            (Value("offsets", positions),),
            (("broadcast_dimensions", (0,)), ("shape", (1, 50))),
        ),
        Op("cast", ("suffix_pad",), (Value("suffix_i32", positions),), (("dtype", "i32"),)),
        Op(
            "cumulative_sum",
            ("suffix_i32",),
            (Value("cumulative", positions),),
            (("axis", 1),),
        ),
        Op("add", ("offsets", "cumulative"), (Value("one_based", positions),)),
        Op(
            "broadcast_in_dim",
            ("negative_one",),
            (Value("negative_ones", positions),),
            (("broadcast_dimensions", (0,)), ("shape", (1, 50))),
        ),
        Op("add", ("one_based", "negative_ones"), (Value("positions", positions),)),
        Op(
            "rope_default",
            ("query", "positions"),
            (Value("rotated_query", query),),
            (("frequency_dtype", "bf16"), ("pairing", "split_half"), ("theta", 10000.0)),
        ),
        Op(
            "scaled_dot_product_attention",
            ("rotated_query", "key", "value", "mask_4d"),
            (Value("attention", query),),
            (("kv_group_size", 8), ("mask_fill", "dtype_min"), ("scale", 0.0625)),
        ),
    ]
    outputs = ["attention"]
    if expose_intermediate:
        outputs.append("prefix_mask")
    if extra_mask_consumer:
        ops.append(Op("logical_and", ("mask_2d", "mask_2d"), (Value("extra_mask", mask_2d),)))
        outputs.append("extra_mask")
    return Program(
        functions=(
            Function(
                "main",
                (
                    Value("prefix_pad", pad),
                    Value("query", query),
                    Value("key", key_value),
                    Value("value", key_value),
                ),
                tuple(outputs),
                Region(tuple(ops)),
            ),
        ),
        entry="main",
    )


def _receipt() -> SuffixMetadataValidationReceipt:
    return SuffixMetadataValidationReceipt(
        SuffixMetadataProblem(CudaArch.SM120),
        IMPLEMENTATION_DIGEST,
        132_000,
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
        False,
    )


class AotSuffixMetadataProviderTests(unittest.TestCase):
    def test_payload_is_fixed_and_fail_closed(self) -> None:
        payload = SuffixMetadataPayload(
            SuffixMetadataProblem(CudaArch.SM120), 132_000, IMPLEMENTATION_DIGEST
        )
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(SuffixMetadataPayload.from_bytes(encoded), payload)
        fields = list(SUFFIX_METADATA_PAYLOAD.unpack(encoded))
        fields[14] += 1
        with self.assertRaisesRegex(FormatError, "exact variant"):
            SuffixMetadataPayload.from_bytes(SUFFIX_METADATA_PAYLOAD.pack(*fields))
        with self.assertRaisesRegex(FormatError, "fixed size"):
            SuffixMetadataPayload.from_bytes(encoded[:-1])

    def test_receipt_requires_exact_hole_capture_racecheck_and_no_ptx(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_outputs_exact", False, "exact full outputs"),
            ("hole_mask_negative_detected", False, "hole-mask"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("latency_ms", 0.0, "finite and positive"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_exact_region_lowering_replaces_twelve_ops(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_suffix_metadata_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        lowering = lower_suffix_metadata_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(len(lowering.fused_execution_indices), 12)
        self.assertEqual(
            tuple(operand.access for operand in lowering.commands[0].command.operands),
            (OperandAccess.READ, OperandAccess.WRITE, OperandAccess.WRITE),
        )
        self.assertEqual(
            schedule.values[lowering.commands[0].command.operands[0].value_id].type.shape,
            (1, 968),
        )
        again = lower_suffix_metadata_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_suffix_metadata_partial_lowering(lowering),
            dump_suffix_metadata_partial_lowering(again),
        )

    def test_region_refuses_constant_public_consumer_and_wrong_target(self) -> None:
        for program, message in (
            (_program(suffix_value=False), "mask broadcasts"),
            (_program(negative_one=-2), "negative-one"),
            (_program(expose_intermediate=True), "public"),
            (_program(extra_mask_consumer=True), "exclusive"),
        ):
            inventory = build_lowering_inventory(program)
            schedule = build_execution_schedule(program)
            memory = build_memory_plan(schedule)
            with self.subTest(message=message), self.assertRaisesRegex(
                ValidationError, message
            ):
                lower_suffix_metadata_commands(
                    schedule,
                    inventory,
                    memory,
                    _receipt(),
                    target_arch=CudaArch.SM120,
                )
        inventory = build_lowering_inventory(_program())
        with self.assertRaisesRegex(ValidationError, "target differs"):
            make_suffix_metadata_capabilities(
                inventory, _receipt(), target_arch=CudaArch.SM110
            )


if __name__ == "__main__":
    unittest.main()
