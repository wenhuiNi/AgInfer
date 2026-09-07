from __future__ import annotations

import dataclasses
import math
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.ir import (
    DType,
    Device,
    Function,
    Op,
    Program,
    Region,
    Tensor,
    TensorType,
    Value,
    execute,
)
from aginfer.lowering import (
    OperandAccess,
    build_execution_schedule,
    build_lowering_inventory,
    build_memory_plan,
    resolve_provider_capabilities,
)
from aginfer.providers import (
    TIME_EMBEDDING_PAYLOAD,
    TimeEmbeddingPayload,
    TimeEmbeddingProblem,
    TimeEmbeddingValidationReceipt,
    dump_time_embedding_partial_lowering,
    lower_time_embedding_commands,
    make_time_embedding_capabilities,
)
from aginfer.schema import CudaArch


MODULE_DIGEST = "b" * 64


def _program(*, dimension: int = 1024, minimum: float = 0.004) -> Program:
    source = TensorType(DType.F32, (1,), device=Device.CUDA)
    output = TensorType(DType.F32, (1, dimension), device=Device.CUDA)
    return Program(
        functions=(
            Function(
                "main",
                (Value("time", source),),
                ("embedding",),
                Region(
                    (
                        Op(
                            "sinusoidal_embedding",
                            ("time",),
                            (Value("embedding", output),),
                            (
                                ("dimension", dimension),
                                ("min_period", minimum),
                                ("max_period", 4.0),
                            ),
                        ),
                    )
                ),
            ),
        ),
        entry="main",
    )


def _receipt() -> TimeEmbeddingValidationReceipt:
    item = build_lowering_inventory(_program()).ops[0]
    return TimeEmbeddingValidationReceipt(
        TimeEmbeddingProblem.from_inventory(item, target_arch=CudaArch.SM120),
        MODULE_DIGEST,
        153_000,
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
        1.0,
        0.0,
        0.0,
        0.002,
        0.006,
        False,
    )


class AotTimeEmbeddingProviderTests(unittest.TestCase):
    def test_reference_formula_uses_geometric_endpoints_and_sin_then_cos(self) -> None:
        small = Program(
            functions=(
                Function(
                    "main",
                    (Value("time", TensorType(DType.F32, (1,))),),
                    ("embedding",),
                    Region(
                        (
                            Op(
                                "sinusoidal_embedding",
                                ("time",),
                                (Value("embedding", TensorType(DType.F32, (1, 4))),),
                                (("dimension", 4), ("min_period", 1.0), ("max_period", 2.0)),
                            ),
                        )
                    ),
                ),
            ),
            entry="main",
        )
        result = execute(small, {"time": Tensor.from_values(DType.F32, (1,), (0.25,))})
        expected = (1.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
        for actual, wanted in zip(result.outputs[0].data, expected, strict=True):
            self.assertAlmostEqual(actual, wanted, places=7)

    def test_problem_and_payload_are_exact_and_stable(self) -> None:
        inventory = build_lowering_inventory(_program())
        problem = TimeEmbeddingProblem.from_inventory(
            inventory.ops[0], target_arch=CudaArch.SM120
        )
        payload = TimeEmbeddingPayload(problem, 153_000, MODULE_DIGEST)
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(TimeEmbeddingPayload.from_bytes(encoded), payload)
        fields = list(TIME_EMBEDDING_PAYLOAD.unpack(encoded))
        fields[7] = 256
        with self.assertRaisesRegex(FormatError, "exact variant"):
            TimeEmbeddingPayload.from_bytes(TIME_EMBEDDING_PAYLOAD.pack(*fields))
        for changed, message in (
            (_program(dimension=512), "exact envelope"),
            (_program(minimum=0.01), "exact envelope"),
        ):
            with self.assertRaisesRegex(ValidationError, message):
                TimeEmbeddingProblem.from_inventory(
                    build_lowering_inventory(changed).ops[0],
                    target_arch=CudaArch.SM120,
                )
        with self.assertRaisesRegex(ValidationError, "requires SM120"):
            TimeEmbeddingProblem.from_inventory(
                inventory.ops[0], target_arch=CudaArch.SM110
            )

    def test_receipt_locks_full_output_f64_semantics_capture_and_latency(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_output_compared", False, "full-output"),
            ("f64_period_negative_detected", False, "F64-period"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("max_abs", 0.01, "correctness gate"),
            ("reference_chain_latency_ms", 0.001, "latency evidence"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_capability_and_command_lowering_are_deterministic(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_time_embedding_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        self.assertTrue(
            resolve_provider_capabilities(
                inventory, capabilities, target_arch=CudaArch.SM120, strict=True
            ).complete
        )
        lowering = lower_time_embedding_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(
            tuple(item.access for item in lowering.commands[0].command.operands),
            (OperandAccess.READ, OperandAccess.WRITE),
        )
        again = lower_time_embedding_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_time_embedding_partial_lowering(lowering),
            dump_time_embedding_partial_lowering(again),
        )
        with self.assertRaisesRegex(ValidationError, "identities"):
            lower_time_embedding_commands(
                schedule,
                inventory,
                dataclasses.replace(memory, schedule_sha256="1" * 64),
                receipt,
                target_arch=CudaArch.SM120,
            )


if __name__ == "__main__":
    unittest.main()
