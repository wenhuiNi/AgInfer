from __future__ import annotations

import dataclasses
import hashlib
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value
from aginfer.lowering import (
    OperandAccess,
    build_execution_schedule,
    build_lowering_inventory,
    build_memory_plan,
    resolve_provider_capabilities,
)
from aginfer.providers import (
    CUDA_KERNEL_PAYLOAD,
    AotCastProblem,
    AotCastValidationReceipt,
    CudaKernelDType,
    CudaKernelId,
    CudaKernelPayload,
    dump_aot_cast_partial_lowering,
    lower_aot_cast_commands,
    make_aot_cast_capabilities,
)
from aginfer.schema import CudaArch


MODULE_DIGEST = "1" * 64


def _program(*, colliding_numel: bool = False) -> Program:
    bf16 = TensorType(DType.BF16, (2, 3), device=Device.CUDA)
    f32 = TensorType(DType.F32, (2, 3), device=Device.CUDA)
    boolean = TensorType(DType.BOOL, (6,), device=Device.CUDA)
    i32 = TensorType(DType.I32, (6,), device=Device.CUDA)
    inputs = [Value("x", bf16), Value("mask", boolean)]
    ops = [
        Op("cast", ("x",), (Value("wide", f32),), (("dtype", "f32"),)),
        Op("cast", ("wide",), (Value("roundtrip", bf16),), (("dtype", "bf16"),)),
        Op("cast", ("mask",), (Value("integers", i32),), (("dtype", "i32"),)),
    ]
    outputs = ["roundtrip", "integers"]
    if colliding_numel:
        bf16_flat = TensorType(DType.BF16, (1, 6), device=Device.CUDA)
        f32_flat = TensorType(DType.F32, (1, 6), device=Device.CUDA)
        inputs.append(Value("flat", bf16_flat))
        ops.append(
            Op("cast", ("flat",), (Value("flat_wide", f32_flat),), (("dtype", "f32"),))
        )
        outputs.append("flat_wide")
    return Program(
        functions=(Function("main", tuple(inputs), tuple(outputs), Region(tuple(ops))),),
        entry="main",
    )


def _receipt(inventory=None) -> AotCastValidationReceipt:
    selected = inventory or build_lowering_inventory(_program())
    problems = tuple(
        sorted(
            {
                AotCastProblem.from_inventory(item, target_arch=CudaArch.SM120)
                for item in selected.ops
                if item.opcode == "cast"
            },
            key=lambda item: (int(item.kernel_id), item.numel),
        )
    )
    return AotCastValidationReceipt(
        target_arch=CudaArch.SM120,
        module_bytes=12_296,
        module_sha256=MODULE_DIGEST,
        cuda_driver_version=13_000,
        cuda_runtime_version=12_080,
        problems=problems,
        symbols=tuple(
            item.symbol
            for item in (
                CudaKernelId.CAST_BF16_TO_F32,
                CudaKernelId.CAST_F32_TO_BF16,
                CudaKernelId.CAST_BOOL_TO_I32,
            )
        ),
        normal_launches_per_problem=2,
        normal_repeat_bit_exact=True,
        capture_replay_bit_exact=True,
        full_output_compared=True,
        contains_ptx=False,
    )


class AotCudaProviderTests(unittest.TestCase):
    def test_problem_accepts_only_the_three_implemented_exact_casts(self) -> None:
        inventory = build_lowering_inventory(_program())
        problems = tuple(
            AotCastProblem.from_inventory(item, target_arch=CudaArch.SM120)
            for item in inventory.ops
        )
        self.assertEqual(
            tuple(item.kernel_id for item in problems),
            (
                CudaKernelId.CAST_BF16_TO_F32,
                CudaKernelId.CAST_F32_TO_BF16,
                CudaKernelId.CAST_BOOL_TO_I32,
            ),
        )
        self.assertEqual(tuple(item.numel for item in problems), (6, 6, 6))
        with self.assertRaisesRegex(ValidationError, "dtype attribute"):
            AotCastProblem.from_inventory(
                dataclasses.replace(inventory.ops[0], attributes=(("dtype", "bf16"),)),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "no implemented kernel"):
            bad_output = dataclasses.replace(inventory.ops[0].output_types[0], dtype="i32")
            AotCastProblem.from_inventory(
                dataclasses.replace(
                    inventory.ops[0],
                    output_types=(bad_output,),
                    attributes=(("dtype", "i32"),),
                ),
                target_arch=CudaArch.SM120,
            )

    def test_payload_is_fixed_numeric_and_byte_deterministic(self) -> None:
        problem = AotCastProblem(
            CudaArch.SM120,
            CudaKernelId.CAST_BF16_TO_F32,
            CudaKernelDType.BF16,
            CudaKernelDType.F32,
            51_200,
        )
        payload = CudaKernelPayload.for_problem(
            problem, module_bytes=12_296, module_sha256=MODULE_DIGEST
        )
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 128)
        self.assertEqual(CudaKernelPayload.from_bytes(encoded), payload)
        self.assertEqual(CudaKernelPayload.from_bytes(encoded).to_bytes(), encoded)
        self.assertEqual(payload.grid_x, 200)
        self.assertNotIn(problem.kernel_id.symbol.encode(), encoded)
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "255e264cd13c11cc34572af6f4553daf860449479499d1a30702f3ef3934a82c",
        )

    def test_payload_corruption_and_launch_bounds_fail_closed(self) -> None:
        problem = AotCastProblem(
            CudaArch.SM120,
            CudaKernelId.CAST_BOOL_TO_I32,
            CudaKernelDType.BOOL,
            CudaKernelDType.I32,
            968,
        )
        payload = CudaKernelPayload.for_problem(
            problem, module_bytes=12_296, module_sha256=MODULE_DIGEST
        )
        encoded = payload.to_bytes()
        with self.assertRaisesRegex(FormatError, "fixed size"):
            CudaKernelPayload.from_bytes(encoded[:-1])
        fields = list(CUDA_KERNEL_PAYLOAD.unpack(encoded))
        for index, value, message in (
            (0, b"BADMAGIC", "bad.*magic"),
            (4, 121, "unknown target"),
            (5, 99, "unknown target"),
            (8, 0, "fixed fields"),
            (9, 0, "launch"),
            (14, bytes(32), "canonical SHA"),
            (19, b"x" + bytes(15), "fixed fields"),
        ):
            changed = list(fields)
            changed[index] = value
            with self.subTest(index=index), self.assertRaisesRegex(FormatError, message):
                CudaKernelPayload.from_bytes(CUDA_KERNEL_PAYLOAD.pack(*changed))

    def test_receipt_round_trip_and_evidence_are_strict(self) -> None:
        receipt = _receipt()
        self.assertEqual(AotCastValidationReceipt.from_dict(receipt.to_dict()), receipt)
        for key, value, message in (
            ("extra", True, "unknown or missing"),
            ("normal_launches_per_problem", 1, "two launches"),
            ("normal_repeat_bit_exact", False, "normal repeats"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_output_compared", False, "full-output"),
            ("contains_ptx", True, "without embedded PTX"),
        ):
            record = receipt.to_dict()
            record[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, message):
                AotCastValidationReceipt.from_dict(record)

    def test_equal_numel_different_shapes_get_distinct_capabilities(self) -> None:
        inventory = build_lowering_inventory(_program(colliding_numel=True))
        receipt = _receipt(inventory)
        capabilities = make_aot_cast_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        bf16_to_f32 = [
            item for item in capabilities
            if item.implementation_id == "aginfer.cast.bf16_to_f32.v1"
        ]
        self.assertEqual(len(capabilities), 4)
        self.assertEqual(len(bf16_to_f32), 2)
        self.assertEqual(
            {item.input_types[0].shape for item in bf16_to_f32}, {(2, 3), (1, 6)}
        )

    def test_exact_capabilities_and_partial_commands_preserve_blockers(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        receipt = _receipt(inventory)
        capabilities = make_aot_cast_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        resolution = resolve_provider_capabilities(
            inventory, capabilities, target_arch=CudaArch.SM120, strict=True
        )
        self.assertTrue(resolution.complete)
        self.assertEqual(len(resolution.receipts), 3)
        lowering = lower_aot_cast_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 3)
        self.assertEqual(lowering.unhandled_execution_indices, ())
        self.assertTrue(all(item.command.workspace_bytes == 0 for item in lowering.commands))
        self.assertEqual(
            tuple(item.command.tag.name for item in lowering.commands),
            ("CUDA_KERNEL", "CUDA_KERNEL", "CUDA_KERNEL"),
        )
        self.assertEqual(
            tuple(operand.access for operand in lowering.commands[0].command.operands),
            (OperandAccess.READ, OperandAccess.WRITE),
        )
        again = lower_aot_cast_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_aot_cast_partial_lowering(lowering),
            dump_aot_cast_partial_lowering(again),
        )


if __name__ == "__main__":
    unittest.main()
