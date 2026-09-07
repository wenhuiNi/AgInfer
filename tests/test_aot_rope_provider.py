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
    ROPE_PAYLOAD,
    RopePayload,
    RopeProblem,
    RopeValidationReceipt,
    RopeVariant,
    dump_rope_partial_lowering,
    lower_rope_commands,
    make_rope_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "8" * 64


def _program(*, expose_transpose: bool = False) -> Program:
    inputs: list[Value] = []
    ops: list[Op] = []
    outputs: list[str] = []
    for index, variant in enumerate(RopeVariant):
        problem = RopeProblem(CudaArch.SM120, variant)
        source = TensorType(DType.BF16, problem.source_shape, device=Device.CUDA)
        result = TensorType(DType.BF16, problem.output_shape, device=Device.CUDA)
        positions = TensorType(DType.I32, (1, problem.sequence), device=Device.CUDA)
        source_name = f"source_{index}"
        position_name = f"positions_{index}"
        transposed_name = f"transposed_{index}"
        output_name = f"output_{index}"
        inputs.extend((Value(source_name, source), Value(position_name, positions)))
        ops.extend(
            (
                Op(
                    "transpose",
                    (source_name,),
                    (Value(transposed_name, result),),
                    (("permutation", (0, 2, 1, 3)),),
                ),
                Op(
                    "rope_default",
                    (transposed_name, position_name),
                    (Value(output_name, result),),
                    (
                        ("frequency_dtype", "bf16"),
                        ("pairing", "split_half"),
                        ("theta", 10000.0),
                    ),
                ),
            )
        )
        if expose_transpose and index == 0:
            outputs.append(transposed_name)
        outputs.append(output_name)
    return Program(
        functions=(Function("main", tuple(inputs), tuple(outputs), Region(tuple(ops))),),
        entry="main",
    )


def _receipt(variant: RopeVariant) -> RopeValidationReceipt:
    return RopeValidationReceipt(
        RopeProblem(CudaArch.SM120, variant),
        IMPLEMENTATION_DIGEST,
        80_000,
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
        0.999999,
        0.0078125,
        0.0,
        False,
    )


def _receipts() -> tuple[RopeValidationReceipt, ...]:
    return tuple(_receipt(variant) for variant in RopeVariant)


class AotRopeProviderTests(unittest.TestCase):
    def test_problem_accepts_only_four_exact_split_half_envelopes(self) -> None:
        inventory = build_lowering_inventory(_program())
        items = tuple(item for item in inventory.ops if item.opcode == "rope_default")
        self.assertEqual(
            {RopeProblem.from_inventory(item, target_arch=CudaArch.SM120).variant for item in items},
            set(RopeVariant),
        )
        with self.assertRaisesRegex(ValidationError, "exact envelopes"):
            RopeProblem.from_inventory(
                dataclasses.replace(
                    items[0],
                    attributes=(
                        ("frequency_dtype", "bf16"),
                        ("pairing", "interleaved"),
                        ("theta", 10000.0),
                    ),
                ),
                target_arch=CudaArch.SM120,
            )
        with self.assertRaisesRegex(ValidationError, "require SM120"):
            RopeProblem.from_inventory(items[0], target_arch=CudaArch.SM110)

    def test_payload_is_fixed_and_fail_closed_for_all_variants(self) -> None:
        for variant in RopeVariant:
            with self.subTest(variant=variant):
                payload = RopePayload(
                    RopeProblem(CudaArch.SM120, variant),
                    80_000,
                    IMPLEMENTATION_DIGEST,
                )
                encoded = payload.to_bytes()
                self.assertEqual(len(encoded), 192)
                self.assertEqual(RopePayload.from_bytes(encoded), payload)
                fields = list(ROPE_PAYLOAD.unpack(encoded))
                fields[12] += 1
                with self.assertRaisesRegex(FormatError, "exact variants"):
                    RopePayload.from_bytes(ROPE_PAYLOAD.pack(*fields))
                with self.assertRaisesRegex(FormatError, "fixed size"):
                    RopePayload.from_bytes(encoded[:-1])

    def test_receipt_requires_split_control_capture_racecheck_and_no_ptx(self) -> None:
        receipt = _receipt(RopeVariant.BF16_SEQUENCE968_HEADS8)
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("split_half_negative_verified", False, "negative control"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("cosine", 0.9, "correctness gate"),
            ("max_abs", 1.0, "correctness gate"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_exact_region_lowering_fuses_only_exclusive_transpose(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        receipts = _receipts()
        capabilities = make_rope_capabilities(
            inventory, receipts, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 4)
        lowering = lower_rope_commands(
            schedule, inventory, memory, receipts, target_arch=CudaArch.SM120
        )
        self.assertTrue(lowering.complete)
        self.assertEqual(len(lowering.commands), 4)
        self.assertEqual(len(lowering.fused_execution_indices), 6)
        self.assertEqual(len(lowering.elided_execution_indices), 2)
        for item in lowering.commands:
            variant = RopeVariant(ROPE_PAYLOAD.unpack(item.command.payload)[4])
            expected = 1 if variant in {
                RopeVariant.BF16_SEQUENCE968_HEADS1,
                RopeVariant.BF16_SEQUENCE50_HEADS1,
            } else 2
            self.assertEqual(len(item.fused_execution_indices), expected)
            self.assertEqual(
                tuple(operand.access for operand in item.command.operands),
                (OperandAccess.READ, OperandAccess.READ, OperandAccess.WRITE),
            )
        again = lower_rope_commands(
            schedule, inventory, memory, receipts, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_rope_partial_lowering(lowering), dump_rope_partial_lowering(again)
        )

    def test_region_refuses_entry_visible_transpose(self) -> None:
        program = _program(expose_transpose=True)
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        with self.assertRaisesRegex(ValidationError, "exclusive exact"):
            lower_rope_commands(
                schedule,
                inventory,
                memory,
                _receipts(),
                target_arch=CudaArch.SM120,
            )


if __name__ == "__main__":
    unittest.main()
