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
    resolve_provider_capabilities,
)
from aginfer.providers import (
    CUDA_KERNEL_PAYLOAD,
    AotActivationProblem,
    AotActivationValidationReceipt,
    CudaKernelDType,
    CudaKernelId,
    CudaKernelPayload,
    dump_aot_activation_partial_lowering,
    lower_aot_activation_commands,
    make_aot_activation_capabilities,
)
from aginfer.schema import CudaArch


MODULE_DIGEST = "9" * 64


def _program() -> Program:
    specs = (
        ("denoise", DType.BF16, (1, 50, 4096)),
        ("vision", DType.F32, (1, 256, 4304)),
        ("prefix", DType.BF16, (1, 968, 16384)),
    )
    inputs: list[Value] = []
    outputs: list[str] = []
    ops: list[Op] = []
    for name, dtype, shape in specs:
        tensor = TensorType(dtype, shape, device=Device.CUDA)
        inputs.append(Value(f"{name}_input", tensor))
        outputs.append(f"{name}_output")
        ops.append(
            Op(
                "gelu",
                (f"{name}_input",),
                (Value(f"{name}_output", tensor),),
                (("approximation", "tanh"),),
            )
        )
    silu = TensorType(DType.F32, (1, 1024), device=Device.CUDA)
    inputs.append(Value("time_input", silu))
    outputs.append("time_output")
    ops.append(Op("silu", ("time_input",), (Value("time_output", silu),)))
    return Program(
        functions=(Function("main", tuple(inputs), tuple(outputs), Region(tuple(ops))),),
        entry="main",
    )


def _receipt(problem: AotActivationProblem) -> AotActivationValidationReceipt:
    return AotActivationValidationReceipt(
        problem,
        MODULE_DIGEST,
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
        1.0,
        0.0,
        0.0,
        False,
    )


def _receipts() -> tuple[AotActivationValidationReceipt, ...]:
    inventory = build_lowering_inventory(_program())
    return tuple(
        _receipt(AotActivationProblem.from_inventory(item, target_arch=CudaArch.SM120))
        for item in inventory.ops
    )


class AotActivationProviderTests(unittest.TestCase):
    def test_problem_accepts_only_delivered_gelu_and_silu_envelopes(self) -> None:
        inventory = build_lowering_inventory(_program())
        problems = tuple(
            AotActivationProblem.from_inventory(item, target_arch=CudaArch.SM120)
            for item in inventory.ops
        )
        self.assertEqual(len(set(problems)), 4)
        self.assertEqual(
            {item.kernel_id for item in problems},
            {
                CudaKernelId.GELU_F32,
                CudaKernelId.GELU_BF16,
                CudaKernelId.SILU_F32,
            },
        )
        with self.assertRaisesRegex(ValidationError, "exact envelopes"):
            AotActivationProblem.from_inventory(
                dataclasses.replace(
                    inventory.ops[0], attributes=(("approximation", "none"),)
                ),
                target_arch=CudaArch.SM120,
            )
        changed = dataclasses.replace(
            inventory.ops[1].input_types[0], shape=(1, 256, 4096)
        )
        with self.assertRaisesRegex(ValidationError, "exact envelopes"):
            AotActivationProblem.from_inventory(
                dataclasses.replace(
                    inventory.ops[1], input_types=(changed,), output_types=(changed,)
                ),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "require SM120"):
            AotActivationProblem.from_inventory(
                inventory.ops[0], target_arch=CudaArch.SM110
            )

    def test_shared_payload_round_trip_locks_gelu_problem_and_unary_abi(self) -> None:
        for receipt in _receipts():
            with self.subTest(problem=receipt.problem):
                payload = receipt.payload()
                encoded = payload.to_bytes()
                self.assertEqual(len(encoded), 128)
                self.assertEqual(CudaKernelPayload.from_bytes(encoded), payload)
                fields = list(CUDA_KERNEL_PAYLOAD.unpack(encoded))
                self.assertEqual(fields[18], 1)
                fields[12] += 1
                with self.assertRaisesRegex(FormatError, "exact envelopes"):
                    CudaKernelPayload.from_bytes(CUDA_KERNEL_PAYLOAD.pack(*fields))

    def test_receipt_requires_full_output_capture_racecheck_and_no_ptx(self) -> None:
        receipt = _receipts()[0]
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_output_compared", False, "full-output"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("cosine", 0.9, "correctness gate"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_capabilities_and_partial_lowering_are_exact_and_stable(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        receipts = _receipts()
        capabilities = make_aot_activation_capabilities(
            inventory, receipts, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 4)
        resolution = resolve_provider_capabilities(
            inventory, capabilities, target_arch=CudaArch.SM120, strict=True
        )
        self.assertTrue(resolution.complete)
        lowering = lower_aot_activation_commands(
            schedule, inventory, memory, receipts, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 4)
        self.assertEqual(
            tuple(item.command.operands[0].access for item in lowering.commands),
            (OperandAccess.READ,) * 4,
        )
        self.assertTrue(
            all(
                item.command.operands[1].access == OperandAccess.WRITE
                and item.command.capture_safe
                and item.command.workspace_bytes == 0
                for item in lowering.commands
            )
        )
        again = lower_aot_activation_commands(
            schedule, inventory, memory, receipts, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_aot_activation_partial_lowering(lowering),
            dump_aot_activation_partial_lowering(again),
        )

    def test_duplicate_receipt_and_stage_identity_fail_closed(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        receipts = _receipts()
        with self.assertRaisesRegex(ValidationError, "duplicate problem"):
            make_aot_activation_capabilities(
                inventory, receipts + (receipts[0],), target_arch=CudaArch.SM120
            )
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        with self.assertRaisesRegex(ValidationError, "memory plan does not match"):
            lower_aot_activation_commands(
                schedule,
                inventory,
                dataclasses.replace(memory, schedule_sha256="1" * 64),
                receipts,
                target_arch=CudaArch.SM120,
            )


if __name__ == "__main__":
    unittest.main()
