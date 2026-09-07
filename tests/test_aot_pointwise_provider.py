from __future__ import annotations

import dataclasses
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value
from aginfer.lowering import (
    OperandAccess,
    TensorSignature,
    build_execution_schedule,
    build_lowering_inventory,
    build_memory_plan,
    resolve_provider_capabilities,
)
from aginfer.providers import (
    CUDA_KERNEL_PAYLOAD,
    AotPointwiseProblem,
    AotPointwiseValidationReceipt,
    CudaKernelId,
    CudaKernelPayload,
    dump_aot_pointwise_partial_lowering,
    lower_aot_pointwise_commands,
    make_aot_pointwise_capabilities,
)
from aginfer.schema import CudaArch


MODULE_DIGEST = "2" * 64
POINTWISE_IDS = (
    CudaKernelId.ADD_F32,
    CudaKernelId.ADD_BF16,
    CudaKernelId.ADD_I32,
    CudaKernelId.MUL_F32,
    CudaKernelId.MUL_BF16,
)


def _program() -> Program:
    bf16 = TensorType(DType.BF16, (2, 3), device=Device.CUDA)
    f32 = TensorType(DType.F32, (2, 3), device=Device.CUDA)
    i32 = TensorType(DType.I32, (6,), device=Device.CUDA)
    return Program(
        functions=(
            Function(
                "main",
                (
                    Value("a", bf16), Value("b", bf16),
                    Value("x", f32), Value("y", f32),
                    Value("i", i32), Value("j", i32),
                ),
                ("add_bf16", "mul_bf16", "add_f32", "mul_f32", "add_i32"),
                Region(
                    (
                        Op("add", ("a", "b"), (Value("add_bf16", bf16),)),
                        Op("mul", ("a", "b"), (Value("mul_bf16", bf16),)),
                        Op("add", ("x", "y"), (Value("add_f32", f32),)),
                        Op("mul", ("x", "y"), (Value("mul_f32", f32),)),
                        Op("add", ("i", "j"), (Value("add_i32", i32),)),
                    )
                ),
            ),
        ),
        entry="main",
    )


def _receipt(inventory=None) -> AotPointwiseValidationReceipt:
    selected = inventory or build_lowering_inventory(_program())
    problems = tuple(
        sorted(
            {
                AotPointwiseProblem.from_inventory(item, target_arch=CudaArch.SM120)
                for item in selected.ops
            },
            key=lambda item: (int(item.kernel_id), item.numel),
        )
    )
    return AotPointwiseValidationReceipt(
        CudaArch.SM120,
        28_376,
        MODULE_DIGEST,
        13_000,
        12_080,
        problems,
        tuple(item.symbol for item in POINTWISE_IDS),
        2,
        True,
        True,
        True,
        False,
    )


class AotPointwiseProviderTests(unittest.TestCase):
    def test_problem_recognizes_only_exact_same_shape_add_mul(self) -> None:
        inventory = build_lowering_inventory(_program())
        problems = tuple(
            AotPointwiseProblem.from_inventory(item, target_arch=CudaArch.SM120)
            for item in inventory.ops
        )
        self.assertEqual({item.kernel_id for item in problems}, set(POINTWISE_IDS))
        bad_rhs = TensorSignature("f32", (1, 6), "row_major", "cuda")
        with self.assertRaisesRegex(ValidationError, "identical"):
            AotPointwiseProblem.from_inventory(
                dataclasses.replace(inventory.ops[2], input_types=(inventory.ops[2].input_types[0], bad_rhs)),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "attributes"):
            AotPointwiseProblem.from_inventory(
                dataclasses.replace(inventory.ops[0], attributes=(("alpha", 1),)),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "no implemented"):
            AotPointwiseProblem.from_inventory(
                dataclasses.replace(inventory.ops[4], opcode="mul"),
                target_arch=CudaArch.SM120,
            )

    def test_binary_payload_round_trip_locks_launch_abi(self) -> None:
        problem = AotPointwiseProblem.from_inventory(
            build_lowering_inventory(_program()).ops[0], target_arch=CudaArch.SM120
        )
        payload = CudaKernelPayload.for_problem(
            problem, module_bytes=28_376, module_sha256=MODULE_DIGEST
        )
        encoded = payload.to_bytes()
        self.assertEqual(CudaKernelPayload.from_bytes(encoded), payload)
        fields = list(CUDA_KERNEL_PAYLOAD.unpack(encoded))
        self.assertEqual(fields[18], 2)
        fields[18] = 1
        with self.assertRaisesRegex(FormatError, "launch ABI"):
            CudaKernelPayload.from_bytes(CUDA_KERNEL_PAYLOAD.pack(*fields))
        fields = list(CUDA_KERNEL_PAYLOAD.unpack(encoded))
        fields[5] = int(CudaKernelId.ADD_F32)
        with self.assertRaisesRegex(FormatError, "kernel ID"):
            CudaKernelPayload.from_bytes(CUDA_KERNEL_PAYLOAD.pack(*fields))

    def test_receipt_round_trip_and_evidence_are_strict(self) -> None:
        receipt = _receipt()
        self.assertEqual(AotPointwiseValidationReceipt.from_dict(receipt.to_dict()), receipt)
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
                AotPointwiseValidationReceipt.from_dict(record)

    def test_capabilities_and_partial_lowering_are_exact(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        receipt = _receipt(inventory)
        capabilities = make_aot_pointwise_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 5)
        resolution = resolve_provider_capabilities(
            inventory, capabilities, target_arch=CudaArch.SM120, strict=True
        )
        self.assertTrue(resolution.complete)
        lowering = lower_aot_pointwise_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 5)
        self.assertEqual(
            tuple(item.access for item in lowering.commands[0].command.operands),
            (OperandAccess.READ, OperandAccess.READ, OperandAccess.WRITE),
        )
        again = lower_aot_pointwise_commands(
            schedule, inventory, memory_plan, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_aot_pointwise_partial_lowering(lowering),
            dump_aot_pointwise_partial_lowering(again),
        )

    def test_receipt_coverage_and_stage_identity_fail_closed(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        receipt = _receipt(inventory)
        missing = dataclasses.replace(receipt, problems=receipt.problems[:-1])
        with self.assertRaisesRegex(ValidationError, "missing=1"):
            make_aot_pointwise_capabilities(
                inventory, missing, target_arch=CudaArch.SM120
            )
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        with self.assertRaisesRegex(ValidationError, "memory plan does not match"):
            lower_aot_pointwise_commands(
                schedule,
                inventory,
                dataclasses.replace(memory_plan, schedule_sha256="1" * 64),
                receipt,
                target_arch=CudaArch.SM120,
            )


if __name__ == "__main__":
    unittest.main()
