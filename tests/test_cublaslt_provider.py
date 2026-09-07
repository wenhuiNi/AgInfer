from __future__ import annotations

import dataclasses
import hashlib
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value
from aginfer.lowering import TensorSignature, build_lowering_inventory
from aginfer.lowering import (
    OperandAccess,
    build_execution_schedule,
    build_memory_plan,
    resolve_provider_capabilities,
)
from aginfer.providers import (
    CUBLASLT_LINEAR_PAYLOAD,
    CublasLtAlgorithm,
    CublasLtDType,
    CublasLtLinearPayload,
    CublasLtLinearProblem,
    CublasLtValidationReceipt,
    dump_cublaslt_partial_lowering,
    lower_cublaslt_linear_commands,
    make_cublaslt_capabilities,
)
from aginfer.schema import CudaArch


def _linear_item(*, dtype: DType = DType.BF16, rank_three: bool = True):
    x_shape = (1, 50, 1024) if rank_three else (50, 1024)
    y_shape = x_shape[:-1] + (256,)
    x = TensorType(dtype, x_shape, device=Device.CUDA)
    weight = TensorType(dtype, (256, 1024), device=Device.CUDA)
    bias = TensorType(dtype, (256,), device=Device.CUDA)
    output = TensorType(dtype, y_shape, device=Device.CUDA)
    program = Program(
        functions=(
            Function(
                "main",
                (Value("x", x), Value("weight", weight), Value("bias", bias)),
                ("output",),
                Region(
                    (
                        Op(
                            "linear",
                            ("x", "weight", "bias"),
                            (Value("output", output),),
                        ),
                    )
                ),
            ),
        ),
        entry="main",
    )
    return build_lowering_inventory(program).ops[0]


def _algorithm() -> CublasLtAlgorithm:
    return CublasLtAlgorithm(
        algorithm_id=23,
        tile_id=20,
        split_k=1,
        reduction_scheme=0,
        cta_swizzling=0,
        custom_option=0,
        stages_id=14,
        inner_shape_id=0,
        cluster_shape_id=0,
    )


def _payload(*, dtype: DType = DType.BF16) -> CublasLtLinearPayload:
    return CublasLtLinearPayload(
        problem=CublasLtLinearProblem.from_inventory(
            _linear_item(dtype=dtype), target_arch=CudaArch.SM120
        ),
        cublaslt_version=120805,
        algorithm=_algorithm(),
        workspace_bytes=4 * 1024 * 1024,
        x_alignment=256,
        weight_alignment=256,
        bias_alignment=256,
        output_alignment=256,
        workspace_alignment=256,
    )


def _receipt(payload: CublasLtLinearPayload | None = None) -> CublasLtValidationReceipt:
    selected = payload or _payload()
    return CublasLtValidationReceipt(
        payload=selected,
        cuda_driver_version=13000,
        cuda_runtime_version=12080,
        algo_check=True,
        normal_launches=2,
        normal_repeat_bit_exact=True,
        capture_replay_bit_exact=True,
        sample_count=64,
        max_abs=0.0001 if selected.problem.dtype == CublasLtDType.BF16 else 0.00001,
        sample_tolerance=0.005 if selected.problem.dtype == CublasLtDType.BF16 else 0.0005,
    )


def _two_linear_program() -> Program:
    x = TensorType(DType.BF16, (1, 50, 1024), device=Device.CUDA)
    weight = TensorType(DType.BF16, (256, 1024), device=Device.CUDA)
    bias = TensorType(DType.BF16, (256,), device=Device.CUDA)
    output = TensorType(DType.BF16, (1, 50, 256), device=Device.CUDA)
    return Program(
        functions=(
            Function(
                "main",
                (Value("x", x), Value("weight", weight), Value("bias", bias)),
                ("output",),
                Region(
                    (
                        Op("linear", ("x", "weight", "bias"), (Value("first", output),)),
                        Op("linear", ("x", "weight", "bias"), (Value("second", output),)),
                        Op("relu", ("second",), (Value("output", output),)),
                    )
                ),
            ),
        ),
        entry="main",
    )


def _replace_field(encoded: bytes, index: int, value: object) -> bytes:
    fields = list(CUBLASLT_LINEAR_PAYLOAD.unpack(encoded))
    fields[index] = value
    return CUBLASLT_LINEAR_PAYLOAD.pack(*fields)


class CublasLtProviderTests(unittest.TestCase):
    def test_tf32_is_explicit_and_f32_only(self):
        original=_payload(dtype=DType.F32)
        p=dataclasses.replace(original,compute_mode=2)
        self.assertEqual(CublasLtLinearPayload.from_bytes(p.to_bytes()),p)
        self.assertEqual(CUBLASLT_LINEAR_PAYLOAD.unpack(original.to_bytes())[2],0)
        self.assertEqual(CUBLASLT_LINEAR_PAYLOAD.unpack(p.to_bytes())[2],1)
        for kwargs in ({'compute_mode':True},{'compute_mode':3},
                       {'problem':_payload().problem}):
            with self.assertRaises(ValidationError):dataclasses.replace(p,**kwargs)
        blob=bytearray(p.to_bytes());blob[10]=0
        with self.assertRaises(FormatError):CublasLtLinearPayload.from_bytes(blob)

    def test_linear_problem_flattens_leading_dimensions_and_accepts_f32(self) -> None:
        bf16 = CublasLtLinearProblem.from_inventory(
            _linear_item(), target_arch=CudaArch.SM120
        )
        f32 = CublasLtLinearProblem.from_inventory(
            _linear_item(dtype=DType.F32, rank_three=False),
            target_arch=CudaArch.SM89,
        )
        self.assertEqual((bf16.dtype, bf16.m, bf16.n, bf16.k), (CublasLtDType.BF16, 50, 256, 1024))
        self.assertEqual((f32.dtype, f32.m, f32.n, f32.k), (CublasLtDType.F32, 50, 256, 1024))

    def test_problem_refuses_non_exact_dtype_layout_shape_and_site(self) -> None:
        item = _linear_item()
        with self.assertRaisesRegex(ValidationError, "required GEMM linear"):
            CublasLtLinearProblem.from_inventory(
                dataclasses.replace(item, opcode="relu"), target_arch=CudaArch.SM120
            )
        bad_dtype = TensorSignature("f16", (1, 50, 1024), "row_major", "cuda")
        with self.assertRaisesRegex(ValidationError, "one exact tensor dtype"):
            CublasLtLinearProblem.from_inventory(
                dataclasses.replace(item, input_types=(bad_dtype,) + item.input_types[1:]),
                target_arch=CudaArch.SM120,
            )
        bad_layout = dataclasses.replace(item.input_types[0], layout="channels_last")
        with self.assertRaisesRegex(ValidationError, "CUDA row-major"):
            CublasLtLinearProblem.from_inventory(
                dataclasses.replace(item, input_types=(bad_layout,) + item.input_types[1:]),
                target_arch=CudaArch.SM120,
            )
        bad_output = dataclasses.replace(item.output_types[0], shape=(1, 50, 255))
        with self.assertRaisesRegex(ValidationError, "do not close"):
            CublasLtLinearProblem.from_inventory(
                dataclasses.replace(item, output_types=(bad_output,)),
                target_arch=CudaArch.SM120,
            )
        symbolic = dataclasses.replace(item.input_types[0], shape=(1, "S", 1024))
        with self.assertRaisesRegex(ValidationError, "positive static"):
            CublasLtLinearProblem.from_inventory(
                dataclasses.replace(item, input_types=(symbolic,) + item.input_types[1:]),
                target_arch=CudaArch.SM120,
            )

    def test_payload_round_trip_is_fixed_and_byte_deterministic(self) -> None:
        self.assertEqual(CUBLASLT_LINEAR_PAYLOAD.size, 192)
        first = _payload().to_bytes()
        second = _payload().to_bytes()
        self.assertEqual(first, second)
        self.assertEqual(CublasLtLinearPayload.from_bytes(first), _payload())
        self.assertEqual(CublasLtLinearPayload.from_bytes(first).to_bytes(), first)
        self.assertEqual(
            hashlib.sha256(first).hexdigest(),
            "cb5fcad5c85e598f2805c2b42419a0141d1210c5a1c52553c8ea42afb2e4ce29",
        )

    def test_payload_fixed_fields_and_bounds_fail_closed(self) -> None:
        encoded = _payload().to_bytes()
        with self.assertRaisesRegex(FormatError, "fixed size"):
            CublasLtLinearPayload.from_bytes(encoded[:-1])
        with self.assertRaisesRegex(FormatError, "bad.*magic"):
            CublasLtLinearPayload.from_bytes(_replace_field(encoded, 0, b"BADMAGIC"))
        with self.assertRaisesRegex(FormatError, "unsupported.*schema"):
            CublasLtLinearPayload.from_bytes(_replace_field(encoded, 1, 2))
        with self.assertRaisesRegex(FormatError, "unknown target"):
            CublasLtLinearPayload.from_bytes(_replace_field(encoded, 4, 121))
        with self.assertRaisesRegex(FormatError, "unknown target"):
            CublasLtLinearPayload.from_bytes(_replace_field(encoded, 6, 99))
        for index, value in ((9, 0), (12, 1), (16, 1), (20, 1), (39, b"x" + b"\0" * 7)):
            with self.subTest(index=index), self.assertRaisesRegex(
                FormatError, "fixed layout|reserved"
            ):
                CublasLtLinearPayload.from_bytes(_replace_field(encoded, index, value))
        for index, value, message in (
            (5, 0, "library version"),
            (17, 0, "problem M"),
            (25, -1, "algorithm ID"),
            (27, -1, "split-K"),
            (34, 3, "alignment"),
        ):
            with self.subTest(index=index), self.assertRaisesRegex(FormatError, message):
                CublasLtLinearPayload.from_bytes(_replace_field(encoded, index, value))

    def test_algorithm_workspace_and_alignment_constructors_validate_types(self) -> None:
        with self.assertRaisesRegex(ValidationError, "algorithm ID"):
            dataclasses.replace(_algorithm(), algorithm_id=True)
        with self.assertRaisesRegex(ValidationError, "split-K"):
            dataclasses.replace(_algorithm(), split_k=-1)
        with self.assertRaisesRegex(ValidationError, "workspace bytes"):
            dataclasses.replace(_payload(), workspace_bytes=-1)
        with self.assertRaisesRegex(ValidationError, "alignment"):
            dataclasses.replace(_payload(), workspace_alignment=0)

    def test_validation_receipt_is_strict_self_contained_and_digest_bound(self) -> None:
        receipt = _receipt()
        record = receipt.to_dict()
        self.assertEqual(CublasLtValidationReceipt.from_dict(record), receipt)
        self.assertEqual(record["payload_sha256"], receipt.payload_sha256)
        self.assertEqual(len(record["payload_hex"]), 384)

        for key, value, message in (
            ("extra", True, "unknown or missing"),
            ("algo_check", False, "AlgoCheck"),
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("sample_count", 63, "64 CPU"),
            ("max_abs", 1.0, "correctness gate"),
            ("sample_tolerance", 1.0, "unlocked"),
        ):
            invalid = dict(record)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, message):
                CublasLtValidationReceipt.from_dict(invalid)
        missing = dict(record)
        missing.pop("algo_check")
        with self.assertRaisesRegex(ValidationError, "unknown or missing"):
            CublasLtValidationReceipt.from_dict(missing)
        bad_digest = dict(record)
        bad_digest["payload_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValidationError, "digest does not match"):
            CublasLtValidationReceipt.from_dict(bad_digest)
        noncanonical = dict(record)
        noncanonical["payload_hex"] = str(record["payload_hex"]).upper()
        with self.assertRaisesRegex(ValidationError, "not canonical"):
            CublasLtValidationReceipt.from_dict(noncanonical)

    def test_exact_capability_and_partial_commands_preserve_all_blockers(self) -> None:
        program = _two_linear_program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_cublaslt_capabilities(
            inventory, (receipt,), target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        capability = capabilities[0]
        self.assertEqual(capability.implementation_digest, receipt.payload_sha256)
        self.assertEqual(capability.implementation_id, "cublaslt.linear.bias.v1")
        self.assertIn("cublaslt=120805", capability.provider_version)

        resolution = resolve_provider_capabilities(
            inventory, capabilities, target_arch=CudaArch.SM120, strict=False
        )
        self.assertEqual(len(resolution.receipts), 2)
        self.assertEqual(sum(item.executions for item in resolution.receipts), 2)
        self.assertEqual([(item.site, item.kind) for item in resolution.issues], [("main:2", "unresolved")])

        lowering = lower_cublaslt_linear_commands(
            schedule,
            inventory,
            memory_plan,
            (receipt,),
            target_arch=CudaArch.SM120,
        )
        self.assertFalse(lowering.complete)
        self.assertEqual(len(lowering.capabilities), 1)
        self.assertEqual(len(lowering.commands), 2)
        self.assertEqual(lowering.unhandled_execution_indices, (2,))
        self.assertEqual(lowering.elided_execution_indices, ())
        self.assertEqual(lowering.workspace_bytes, 4 * 1024 * 1024)
        first = lowering.commands[0]
        self.assertEqual(first.execution_index, 0)
        self.assertEqual(first.command.capability_digest, capability.digest)
        self.assertEqual(first.command.payload, receipt.payload.to_bytes())
        self.assertEqual(
            tuple(operand.access for operand in first.command.operands),
            (OperandAccess.READ, OperandAccess.READ, OperandAccess.READ, OperandAccess.WRITE),
        )
        self.assertEqual(
            tuple(operand.value_id for operand in first.command.operands),
            schedule.ops[0].inputs + schedule.ops[0].outputs,
        )
        second = lower_cublaslt_linear_commands(
            schedule,
            inventory,
            memory_plan,
            (receipt,),
            target_arch=CudaArch.SM120,
        )
        self.assertEqual(
            dump_cublaslt_partial_lowering(lowering),
            dump_cublaslt_partial_lowering(second),
        )

    def test_receipt_coverage_and_cross_stage_identity_fail_closed(self) -> None:
        program = _two_linear_program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory_plan = build_memory_plan(schedule)
        receipt = _receipt()
        with self.assertRaisesRegex(ValidationError, "missing=1"):
            make_cublaslt_capabilities(inventory, (), target_arch=CudaArch.SM120)
        with self.assertRaisesRegex(ValidationError, "duplicate exact problem"):
            make_cublaslt_capabilities(
                inventory, (receipt, receipt), target_arch=CudaArch.SM120
            )
        with self.assertRaisesRegex(ValidationError, "target does not match"):
            make_cublaslt_capabilities(
                inventory, (receipt,), target_arch=CudaArch.SM89
            )
        with self.assertRaisesRegex(ValidationError, "memory plan does not match"):
            lower_cublaslt_linear_commands(
                schedule,
                inventory,
                dataclasses.replace(memory_plan, schedule_sha256="1" * 64),
                (receipt,),
                target_arch=CudaArch.SM120,
            )


if __name__ == "__main__":
    unittest.main()
