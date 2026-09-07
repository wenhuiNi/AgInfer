from __future__ import annotations

import dataclasses
import struct
import unittest

from aginfer.errors import ValidationError
from aginfer.ir import (
    DType,
    Device,
    Function,
    Op,
    Program,
    Region,
    TensorType,
    Value,
    attributes,
)
from aginfer.lowering import (
    AllocationRegion,
    LiteralMaterialization,
    build_execution_schedule,
    build_literal_materialization,
    build_lowering_inventory,
    build_memory_plan,
    dump_literal_materialization,
)


def _literal_program() -> Program:
    bool_scalar = TensorType(DType.BOOL, (1,), device=Device.CUDA)
    bool_matrix = TensorType(DType.BOOL, (2, 2), device=Device.CUDA)
    i32_vector = TensorType(DType.I32, (2,), device=Device.CUDA)
    i32_matrix = TensorType(DType.I32, (2, 3), device=Device.CUDA)
    f32_scalar = TensorType(DType.F32, (1,), device=Device.CUDA)
    f32_vector = TensorType(DType.F32, (2,), device=Device.CUDA)
    bf16_scalar = TensorType(DType.BF16, (1,), device=Device.CUDA)
    bf16_matrix = TensorType(DType.BF16, (2, 2), device=Device.CUDA)
    return Program(
        functions=(
            Function(
                "main",
                (Value("unused", f32_scalar),),
                ("bool_out", "i32_out", "f32_out", "bf16_out", "bf16_dup"),
                Region(
                    (
                        Op(
                            "constant",
                            (),
                            (Value("bool_value", bool_scalar),),
                            attributes(value=(True,)),
                        ),
                        Op(
                            "broadcast_in_dim",
                            ("bool_value",),
                            (Value("bool_out", bool_matrix),),
                            attributes(shape=(2, 2), broadcast_dimensions=(0,)),
                        ),
                        Op(
                            "constant",
                            (),
                            (Value("i32_value", i32_vector),),
                            attributes(value=(-7, 9)),
                        ),
                        Op(
                            "broadcast_in_dim",
                            ("i32_value",),
                            (Value("i32_out", i32_matrix),),
                            attributes(shape=(2, 3), broadcast_dimensions=(0,)),
                        ),
                        Op(
                            "constant",
                            (),
                            (Value("f32_value", f32_scalar),),
                            attributes(value=(-0.0,)),
                        ),
                        Op(
                            "broadcast_in_dim",
                            ("f32_value",),
                            (Value("f32_out", f32_vector),),
                            attributes(shape=(2,), broadcast_dimensions=(0,)),
                        ),
                        Op(
                            "constant",
                            (),
                            (Value("bf16_value", bf16_scalar),),
                            attributes(value=(1.5,)),
                        ),
                        Op(
                            "broadcast_in_dim",
                            ("bf16_value",),
                            (Value("bf16_out", bf16_matrix),),
                            attributes(shape=(2, 2), broadcast_dimensions=(0,)),
                        ),
                        Op(
                            "broadcast_in_dim",
                            ("bf16_value",),
                            (Value("bf16_dup", bf16_matrix),),
                            attributes(shape=(2, 2), broadcast_dimensions=(0,)),
                        ),
                    )
                ),
            ),
        ),
        entry="main",
    )


def _build() -> tuple[object, object, LiteralMaterialization]:
    program = _literal_program()
    inventory = build_lowering_inventory(program)
    schedule = build_execution_schedule(program)
    return schedule, inventory, build_literal_materialization(schedule, inventory)


class LiteralMaterializationTests(unittest.TestCase):
    def test_supported_literal_broadcasts_are_byte_exact_and_deduplicated(self) -> None:
        _, _, materialization = _build()
        self.assertEqual(materialization.materialized_ops, (0, 1, 2, 3, 4))
        self.assertEqual(len(materialization.records), 5)
        self.assertEqual(len(materialization.blobs), 4)

        payloads = {
            record.site: materialization.data[
                record.blob_offset : record.blob_offset + record.byte_size
            ]
            for record in materialization.records
        }
        self.assertEqual(payloads["main:1"], b"\x01" * 4)
        self.assertEqual(
            payloads["main:3"],
            struct.pack("<iiiiii", -7, -7, -7, 9, 9, 9),
        )
        self.assertEqual(payloads["main:5"], struct.pack("<f", -0.0) * 2)
        self.assertEqual(payloads["main:7"], struct.pack("<H", 0x3FC0) * 4)
        self.assertEqual(payloads["main:8"], payloads["main:7"])
        self.assertEqual(
            materialization.records[-1].blob_id,
            materialization.records[-2].blob_id,
        )
        self.assertTrue(all(blob.offset % materialization.alignment == 0 for blob in materialization.blobs))

    def test_exclusions_and_nonliteral_sources_are_not_materialized(self) -> None:
        schedule, inventory, _ = _build()
        materialization = build_literal_materialization(
            schedule, inventory, excluded_execution_indices={0, 4}
        )
        self.assertEqual(materialization.materialized_ops, (1, 2, 3))

        scalar = TensorType(DType.F32, (1,), device=Device.CUDA)
        vector = TensorType(DType.F32, (4,), device=Device.CUDA)
        program = Program(
            functions=(
                Function(
                    "main",
                    (Value("runtime", scalar),),
                    ("runtime_out", "weight_out"),
                    Region(
                        (
                            Op(
                                "broadcast_in_dim",
                                ("runtime",),
                                (Value("runtime_out", vector),),
                                attributes(shape=(4,), broadcast_dimensions=(0,)),
                            ),
                            Op(
                                "constant_ref",
                                (),
                                (Value("weight", scalar),),
                                attributes(namespace="model", name="weight"),
                            ),
                            Op(
                                "broadcast_in_dim",
                                ("weight",),
                                (Value("weight_out", vector),),
                                attributes(shape=(4,), broadcast_dimensions=(0,)),
                            ),
                        )
                    ),
                ),
            ),
            entry="main",
        )
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        self.assertFalse(build_literal_materialization(schedule, inventory).records)

    def test_memory_plan_places_materialized_values_in_the_constant_blob(self) -> None:
        schedule, _, materialization = _build()
        plan = build_memory_plan(schedule, literal_materialization=materialization)
        self.assertEqual(plan.elided_ops, materialization.materialized_ops)
        self.assertEqual(plan.literal_materialization_sha256, materialization.digest)
        for record in materialization.records:
            allocation = plan.allocations[record.output_value_id]
            self.assertEqual(allocation.region, AllocationRegion.CONSTANT)
            self.assertEqual(allocation.offset, record.blob_offset)
            self.assertEqual(allocation.byte_size, record.byte_size)
        self.assertGreaterEqual(plan.constant_bytes, len(materialization.data))

    def test_corruption_identity_mismatch_and_bounds_fail_closed(self) -> None:
        schedule, inventory, materialization = _build()
        corrupted = bytearray(materialization.data)
        corrupted[0] ^= 1
        with self.assertRaisesRegex(ValidationError, "digest differs"):
            dataclasses.replace(materialization, data=bytes(corrupted))

        mismatched = dataclasses.replace(materialization, schedule_sha256="f" * 64)
        with self.assertRaisesRegex(ValidationError, "identities do not match"):
            build_memory_plan(schedule, literal_materialization=mismatched)

        bad_record = dataclasses.replace(materialization.records[0], site="main:999")
        forged = dataclasses.replace(
            materialization, records=(bad_record,) + materialization.records[1:]
        )
        with self.assertRaisesRegex(ValidationError, "differs from its scheduled op"):
            build_memory_plan(schedule, literal_materialization=forged)

        with self.assertRaisesRegex(ValidationError, "bounded data size"):
            build_literal_materialization(schedule, inventory, max_materialized_bytes=3)
        with self.assertRaisesRegex(ValidationError, "scheduled execution"):
            build_literal_materialization(
                schedule, inventory, excluded_execution_indices={len(schedule.ops)}
            )

    def test_dump_is_stable_and_payload_free(self) -> None:
        _, _, first = _build()
        _, _, second = _build()
        first_dump = dump_literal_materialization(first)
        self.assertEqual(first_dump, dump_literal_materialization(second))
        self.assertNotIn("\\u0001", first_dump)
        self.assertIn('"data_sha256"', first_dump)


if __name__ == "__main__":
    unittest.main()
