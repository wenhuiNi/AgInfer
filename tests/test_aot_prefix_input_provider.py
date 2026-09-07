from __future__ import annotations

import dataclasses
import math
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value, attributes
from aginfer.lowering import (
    OperandAccess,
    build_execution_schedule,
    build_literal_materialization,
    build_lowering_inventory,
    build_memory_plan,
)
from aginfer.providers import (
    PREFIX_INPUT_PAYLOAD,
    PrefixInputPayload,
    PrefixInputProblem,
    PrefixInputValidationReceipt,
    dump_prefix_input_partial_lowering,
    lower_prefix_input_commands,
    make_prefix_input_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "d" * 64


def _program(*, scale: float = math.sqrt(2048.0), expose_intermediate: bool = False) -> Program:
    def tensor(dtype: DType, shape: tuple[int, ...]) -> TensorType:
        return TensorType(dtype, shape, device=Device.CUDA)

    image = tensor(DType.F32, (1, 256, 2048))
    embedding = tensor(DType.BF16, (257152, 2048))
    tokens = tensor(DType.I32, (1, 200))
    language_bf16 = tensor(DType.BF16, (1, 200, 2048))
    language_f32 = tensor(DType.F32, (1, 200, 2048))
    prefix = tensor(DType.F32, (1, 968, 2048))
    bool_scalar = tensor(DType.BOOL, (1,))
    image_mask = tensor(DType.BOOL, (1, 256))
    token_mask = tensor(DType.BOOL, (1, 200))
    pad = tensor(DType.BOOL, (1, 968))
    i32_scalar = tensor(DType.I32, (1,))
    positions = tensor(DType.I32, (1, 968))
    query = tensor(DType.BF16, (1, 8, 968, 256))
    ops = [
        Op("constant_ref", (), (Value("embedding", embedding),), attributes(namespace="model", name="embedding")),
        Op("gather", ("embedding", "tokens"), (Value("unscaled", language_bf16),)),
        Op("constant", (), (Value("scale_scalar", tensor(DType.BF16, (1,))),), attributes(value=(scale,))),
        Op("broadcast_in_dim", ("scale_scalar",), (Value("scale", language_bf16),), attributes(shape=(1, 200, 2048), broadcast_dimensions=(0,))),
        Op("mul", ("unscaled", "scale"), (Value("language_bf16", language_bf16),)),
        Op("cast", ("language_bf16",), (Value("language_f32", language_f32),), attributes(dtype="f32")),
        Op("concat", ("image0", "image1", "image2", "language_f32"), (Value("prefix", prefix),), attributes(axis=1)),
    ]
    for index in range(3):
        ops.append(Op("broadcast_in_dim", (f"image_mask{index}",), (Value(f"expanded{index}", image_mask),), attributes(shape=(1, 256), broadcast_dimensions=(0,))))
    ops.extend(
        (
            Op("concat", ("expanded0", "expanded1", "expanded2", "token_mask"), (Value("pad", pad),), attributes(axis=1)),
            Op("cast", ("pad",), (Value("pad_i32", positions),), attributes(dtype="i32")),
            Op("cumulative_sum", ("pad_i32",), (Value("cumulative", positions),), attributes(axis=1)),
            Op("constant", (), (Value("negative_scalar", i32_scalar),), attributes(value=(-1,))),
            Op("broadcast_in_dim", ("negative_scalar",), (Value("negative", positions),), attributes(shape=(1, 968), broadcast_dimensions=(0,))),
            Op("add", ("cumulative", "negative"), (Value("positions", positions),)),
            Op("rope_default", ("query", "positions"), (Value("rotated", query),), attributes(frequency_dtype="bf16", pairing="split_half", theta=10000.0)),
        )
    )
    outputs = ["prefix", "pad", "positions", "rotated"]
    if expose_intermediate:
        outputs.append("unscaled")
    return Program(
        functions=(
            Function(
                "main",
                (
                    Value("image0", image), Value("image1", image), Value("image2", image),
                    Value("image_mask0", bool_scalar), Value("image_mask1", bool_scalar), Value("image_mask2", bool_scalar),
                    Value("tokens", tokens), Value("token_mask", token_mask), Value("query", query),
                ),
                tuple(outputs),
                Region(tuple(ops)),
            ),
        ),
        entry="main",
    )


def _receipt() -> PrefixInputValidationReceipt:
    return PrefixInputValidationReceipt(
        PrefixInputProblem(CudaArch.SM120), IMPLEMENTATION_DIGEST, 150_000,
        12_080, 12_080, 13_000, 2, True, True, True, True, True, True,
        True, 0, 0.05, 0.2, False,
    )


class AotPrefixInputProviderTests(unittest.TestCase):
    def test_payload_is_fixed_and_fail_closed(self) -> None:
        payload = PrefixInputPayload(PrefixInputProblem(CudaArch.SM120), 150_000, IMPLEMENTATION_DIGEST)
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(PrefixInputPayload.from_bytes(encoded), payload)
        fields = list(PREFIX_INPUT_PAYLOAD.unpack(encoded))
        fields[14] += 1
        with self.assertRaisesRegex(FormatError, "exact variant"):
            PrefixInputPayload.from_bytes(PREFIX_INPUT_PAYLOAD.pack(*fields))
        with self.assertRaisesRegex(FormatError, "fixed size"):
            PrefixInputPayload.from_bytes(encoded[:-1])

    def test_receipt_requires_exact_controls_capture_speedup_and_no_ptx(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_outputs_exact", False, "exact full outputs"),
            ("bf16_rounding_negative_detected", False, "BF16 rounding"),
            ("mask_hole_negative_detected", False, "mask-hole"),
            ("invalid_token_guarded", False, "invalid-token"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("original_chain_latency_ms", 0.01, "latency evidence"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, message):
                dataclasses.replace(receipt, **{field: value})

    def test_exact_region_fuses_thirteen_ops_and_excludes_its_literals(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        capabilities = make_prefix_input_capabilities(inventory, _receipt(), target_arch=CudaArch.SM120)
        self.assertEqual(len(capabilities), 1)
        lowering = lower_prefix_input_commands(schedule, inventory, memory, _receipt(), target_arch=CudaArch.SM120)
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(len(lowering.fused_execution_indices), 13)
        self.assertEqual(
            tuple(item.access for item in lowering.commands[0].command.operands),
            (OperandAccess.READ,) * 9 + (OperandAccess.WRITE,) * 3,
        )
        materialization = build_literal_materialization(
            schedule, inventory, excluded_execution_indices=set(lowering.fused_execution_indices)
        )
        self.assertFalse(materialization.records)
        again = lower_prefix_input_commands(schedule, inventory, memory, _receipt(), target_arch=CudaArch.SM120)
        self.assertEqual(dump_prefix_input_partial_lowering(lowering), dump_prefix_input_partial_lowering(again))

    def test_region_refuses_wrong_literal_public_intermediate_and_target(self) -> None:
        for program, message in (
            (_program(scale=1.0), "language dataflow"),
            (_program(expose_intermediate=True), "public"),
        ):
            inventory = build_lowering_inventory(program)
            schedule = build_execution_schedule(program)
            with self.subTest(message=message), self.assertRaisesRegex(ValidationError, message):
                lower_prefix_input_commands(
                    schedule, inventory, build_memory_plan(schedule), _receipt(), target_arch=CudaArch.SM120
                )
        with self.assertRaisesRegex(ValidationError, "target differs"):
            make_prefix_input_capabilities(
                build_lowering_inventory(_program()), _receipt(), target_arch=CudaArch.SM110
            )


if __name__ == "__main__":
    unittest.main()
