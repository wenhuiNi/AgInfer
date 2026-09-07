from __future__ import annotations

import dataclasses
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value, attributes
from aginfer.lowering import build_execution_schedule, build_lowering_inventory, build_memory_plan
from aginfer.providers import (
    PATCH_PROJECTION_HEADER,
    CublasLtAlgorithm,
    CublasLtDType,
    CublasLtLinearPayload,
    CublasLtLinearProblem,
    PatchProjectionPayload,
    PatchProjectionProblem,
    PatchProjectionValidationReceipt,
    dump_patch_projection_partial_lowering,
    lower_patch_projection_commands,
    make_patch_projection_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "e" * 64


def _linear_payload() -> CublasLtLinearPayload:
    return CublasLtLinearPayload(
        CublasLtLinearProblem(CudaArch.SM120, CublasLtDType.F32, 256, 1152, 588),
        120803,
        CublasLtAlgorithm(10, 11, 0, 1, 0, 0, 0, 0, 0),
        2304,
        256,
        256,
        256,
        256,
        256,
    )


def _receipt() -> PatchProjectionValidationReceipt:
    return PatchProjectionValidationReceipt(
        PatchProjectionProblem(CudaArch.SM120),
        IMPLEMENTATION_DIGEST,
        160_000,
        12_080,
        12_080,
        13_000,
        _linear_payload(),
        2,
        True,
        True,
        True,
        1.0,
        1e-5,
        1e-6,
        True,
        0,
        0.1,
        0.2,
        False,
    )


def _program(
    *, permutation: tuple[int, ...] = (0, 2, 3, 1), expose_conv: bool = False
) -> Program:
    def tensor(shape: tuple[int, ...]) -> TensorType:
        return TensorType(DType.F32, shape, device=Device.CUDA)

    image = tensor((1, 3, 224, 224))
    weight = tensor((1152, 3, 14, 14))
    bias = tensor((1152,))
    nchw = tensor((1, 1152, 16, 16))
    nhwc = tensor((1, 16, 16, 1152))
    tokens = tensor((1, 256, 1152))
    ops = (
        Op("constant_ref", (), (Value("weight", weight),), attributes(namespace="model", name="patch.weight")),
        Op("constant_ref", (), (Value("bias", bias),), attributes(namespace="model", name="patch.bias")),
        Op("conv2d", ("image", "weight", "bias"), (Value("nchw", nchw),), attributes(pads=(0, 0, 0, 0), strides=(14, 14))),
        Op("transpose", ("nchw",), (Value("nhwc", nhwc),), attributes(permutation=permutation)),
        Op("reshape", ("nhwc",), (Value("tokens", tokens),), attributes(shape=(1, 256, 1152))),
    )
    outputs = ("tokens", "nchw") if expose_conv else ("tokens",)
    return Program(
        functions=(Function("main", (Value("image", image),), outputs, Region(ops)),),
        entry="main",
    )


class PatchProjectionProviderTests(unittest.TestCase):
    def test_payload_roundtrip_and_corruption_fail_closed(self) -> None:
        payload = PatchProjectionPayload(
            PatchProjectionProblem(CudaArch.SM120),
            160_000,
            IMPLEMENTATION_DIGEST,
            _linear_payload(),
        )
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 384)
        self.assertEqual(PatchProjectionPayload.from_bytes(encoded), payload)
        fields = list(PATCH_PROJECTION_HEADER.unpack(encoded[:192]))
        fields[6] += 1
        with self.assertRaisesRegex(FormatError, "exact variant"):
            PatchProjectionPayload.from_bytes(
                PATCH_PROJECTION_HEADER.pack(*fields) + encoded[192:]
            )
        with self.assertRaisesRegex(FormatError, "fixed size"):
            PatchProjectionPayload.from_bytes(encoded[:-1])

    def test_receipt_requires_numeric_capture_negative_speedup_and_no_ptx(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_output_cosine", 0.9, "full-output"),
            ("full_output_max_abs", 0.1, "full-output"),
            ("patch_order_negative_detected", False, "patch-order"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("original_chain_latency_ms", 0.01, "latency evidence"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, message):
                dataclasses.replace(receipt, **{field: value})

    def test_exact_conv_transpose_region_lowers_deterministically(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        capabilities = make_patch_projection_capabilities(
            inventory, _receipt(), target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        lowering = lower_patch_projection_commands(
            schedule, inventory, memory, _receipt(), target_arch=CudaArch.SM120
        )
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(len(lowering.fused_execution_indices), 2)
        self.assertEqual(lowering.commands[0].command.workspace_bytes, 604416)
        self.assertEqual(len(lowering.commands[0].command.operands), 4)
        again = lower_patch_projection_commands(
            schedule, inventory, memory, _receipt(), target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_patch_projection_partial_lowering(lowering),
            dump_patch_projection_partial_lowering(again),
        )

    def test_region_refuses_public_intermediate_and_target(self) -> None:
        program = _program(expose_conv=True)
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        with self.assertRaisesRegex(ValidationError, "layout chain"):
            lower_patch_projection_commands(
                schedule,
                inventory,
                build_memory_plan(schedule),
                _receipt(),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "target differs"):
            make_patch_projection_capabilities(
                build_lowering_inventory(_program()),
                _receipt(),
                target_arch=CudaArch.SM110,
            )


if __name__ == "__main__":
    unittest.main()
