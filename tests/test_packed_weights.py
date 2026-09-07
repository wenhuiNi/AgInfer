from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from aginfer.constant_store import ConstantStore
from aginfer.errors import ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value, attributes
from aginfer.lowering import (
    CommandOperand,
    CommandPlacement,
    CommandTag,
    OperandAccess,
    ProviderCommand,
    assemble_command_stream,
    build_execution_schedule,
    build_lowering_inventory,
    build_memory_plan,
)
from aginfer.packed_weights import pack_command_weights
from aginfer.schema import CudaArch
from aginfer.source_manifest import build_source_manifest
from tests.helpers import write_safetensors


def _lower(program: Program):
    inventory = build_lowering_inventory(program)
    schedule = build_execution_schedule(program)
    memory = build_memory_plan(schedule)
    output_id = schedule.entry_outputs[0][1]
    constant_id = next(
        item.value_id for item in schedule.values if item.constant_identity is not None
    )
    command = ProviderCommand(
        CommandTag.CUDA_KERNEL,
        7,
        1,
        0,
        hashlib.sha256(b"add").hexdigest(),
        (
            CommandOperand(schedule.entry_inputs[0][1], OperandAccess.READ),
            CommandOperand(constant_id, OperandAccess.READ),
            CommandOperand(output_id, OperandAccess.WRITE),
        ),
        b"add",
        capture_safe=True,
    )
    stream = assemble_command_stream(
        schedule,
        memory,
        (CommandPlacement(0, (0,), command),),
        target_arch=CudaArch.SM120,
    )
    return inventory, schedule, memory, stream, constant_id


class PackedWeightTests(unittest.TestCase):
    def test_literal_pack_is_streamed_aligned_and_deterministic(self) -> None:
        tensor = TensorType(DType.F32, (2,), device=Device.CUDA)
        program = Program(
            (
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("out",),
                    Region(
                        (
                            Op(
                                "constant",
                                (),
                                (Value("bias", tensor),),
                                attributes(value=(1.0, -2.0)),
                            ),
                            Op("add", ("x", "bias"), (Value("out", tensor),)),
                        )
                    ),
                ),
            ),
            "main",
        )
        inventory, schedule, memory, stream, constant_id = _lower(program)
        with tempfile.TemporaryDirectory() as directory:
            first = pack_command_weights(
                Path(directory) / "first.bin",
                schedule,
                memory,
                inventory,
                stream,
            )
            second = pack_command_weights(
                Path(directory) / "second.bin",
                schedule,
                memory,
                inventory,
                stream,
            )
            expected = struct.pack("<ff", 1.0, -2.0) + bytes(248)
            self.assertEqual(first.path.read_bytes(), expected)
            self.assertEqual(second.path.read_bytes(), expected)
            self.assertEqual(first.sha256, hashlib.sha256(expected).hexdigest())
            self.assertEqual(first.byte_size, 256)
            self.assertEqual(first.spans[0].value_id, constant_id)
            self.assertEqual(first.spans[0].byte_size, 8)

    def test_safetensors_source_is_read_by_bounded_chunks(self) -> None:
        tensor = TensorType(DType.BF16, (4,), device=Device.CUDA)
        program = Program(
            (
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("out",),
                    Region(
                        (
                            Op(
                                "constant_ref",
                                (),
                                (Value("bias", tensor),),
                                attributes(namespace="model", name="bias"),
                            ),
                            Op("add", ("x", "bias"), (Value("out", tensor),)),
                        )
                    ),
                ),
            ),
            "main",
        )
        inventory, schedule, memory, stream, _ = _lower(program)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            payload = bytes(range(8))
            write_safetensors(root / "model.safetensors", {"bias": ("BF16", [4], payload)})
            store = ConstantStore(build_source_manifest(root), root=root)
            packed = pack_command_weights(
                Path(directory) / "weights.bin",
                schedule,
                memory,
                inventory,
                stream,
                constants=store,
                chunk_size=3,
            )
            self.assertEqual(packed.path.read_bytes(), payload + bytes(248))

    def test_missing_source_and_changed_contract_fail_closed(self) -> None:
        tensor = TensorType(DType.F32, (1,), device=Device.CUDA)
        program = Program(
            (
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("out",),
                    Region(
                        (
                            Op(
                                "constant_ref",
                                (),
                                (Value("bias", tensor),),
                                attributes(namespace="model", name="bias"),
                            ),
                            Op("add", ("x", "bias"), (Value("out", tensor),)),
                        )
                    ),
                ),
            ),
            "main",
        )
        inventory, schedule, memory, stream, _ = _lower(program)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValidationError, "ConstantStore"):
                pack_command_weights(
                    Path(directory) / "weights.bin",
                    schedule,
                    memory,
                    inventory,
                    stream,
                )


if __name__ == "__main__":
    unittest.main()
