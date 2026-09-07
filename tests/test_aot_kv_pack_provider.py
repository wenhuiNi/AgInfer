from __future__ import annotations

import dataclasses
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.ir import (
    DType,
    Device,
    Function,
    Op,
    Program,
    Region,
    State,
    StateAccess,
    TensorType,
    Value,
)
from aginfer.lowering import (
    OperandAccess,
    build_execution_schedule,
    build_lowering_inventory,
    build_memory_plan,
)
from aginfer.providers import (
    KV_PACK_PAYLOAD,
    KvPackPayload,
    KvPackProblem,
    KvPackValidationReceipt,
    dump_kv_pack_partial_lowering,
    lower_kv_pack_commands,
    make_kv_pack_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "a" * 64


def _program(*, expose_intermediate: bool = False) -> Program:
    prefix = TensorType(DType.BF16, (1, 1, 968, 256), device=Device.CUDA)
    current_bhsd = TensorType(DType.BF16, (1, 1, 50, 256), device=Device.CUDA)
    current_bshd = TensorType(DType.BF16, (1, 50, 1, 256), device=Device.CUDA)
    packed = TensorType(DType.BF16, (1, 1, 1018, 256), device=Device.CUDA)
    query = TensorType(DType.BF16, (1, 8, 50, 256), device=Device.CUDA)
    mask = TensorType(DType.BOOL, (1, 8, 50, 1018), device=Device.CUDA)
    positions = TensorType(DType.I32, (1, 50), device=Device.CUDA)
    outputs = ["attention"]
    if expose_intermediate:
        outputs.append("current_v_bhsd")
    return Program(
        functions=(
            Function(
                "main",
                (
                    Value("query", query),
                    Value("current_k_input", current_bhsd),
                    Value("current_v", current_bshd),
                    Value("positions", positions),
                    Value("mask", mask),
                ),
                tuple(outputs),
                Region(
                    (
                        Op(
                            "state_read",
                            (),
                            (Value("prefix_k", prefix),),
                            (("state", "cache_k"),),
                        ),
                        Op(
                            "state_read",
                            (),
                            (Value("prefix_v", prefix),),
                            (("state", "cache_v"),),
                        ),
                        Op(
                            "rope_default",
                            ("current_k_input", "positions"),
                            (Value("current_k", current_bhsd),),
                            (
                                ("frequency_dtype", "bf16"),
                                ("pairing", "split_half"),
                                ("theta", 10000.0),
                            ),
                        ),
                        Op(
                            "transpose",
                            ("current_v",),
                            (Value("current_v_bhsd", current_bhsd),),
                            (("permutation", (0, 2, 1, 3)),),
                        ),
                        Op(
                            "concat",
                            ("prefix_k", "current_k"),
                            (Value("packed_k", packed),),
                            (("axis", 2),),
                        ),
                        Op(
                            "concat",
                            ("prefix_v", "current_v_bhsd"),
                            (Value("packed_v", packed),),
                            (("axis", 2),),
                        ),
                        Op(
                            "scaled_dot_product_attention",
                            ("query", "packed_k", "packed_v", "mask"),
                            (Value("attention", query),),
                            (
                                ("kv_group_size", 8),
                                ("mask_fill", "dtype_min"),
                                ("scale", 0.0625),
                            ),
                        ),
                    )
                ),
            ),
        ),
        entry="main",
        states=(
            State("cache_k", prefix, StateAccess.READ_ONLY),
            State("cache_v", prefix, StateAccess.READ_ONLY),
        ),
    )


def _receipt() -> KvPackValidationReceipt:
    return KvPackValidationReceipt(
        KvPackProblem(CudaArch.SM120),
        IMPLEMENTATION_DIGEST,
        128_000,
        12_080,
        12_080,
        13_000,
        2,
        True,
        True,
        True,
        True,
        0,
        0.01,
        False,
    )


class AotKvPackProviderTests(unittest.TestCase):
    def test_payload_is_fixed_and_fail_closed(self) -> None:
        payload = KvPackPayload(
            KvPackProblem(CudaArch.SM120), 128_000, IMPLEMENTATION_DIGEST
        )
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(KvPackPayload.from_bytes(encoded), payload)
        fields = list(KV_PACK_PAYLOAD.unpack(encoded))
        fields[12] += 1
        with self.assertRaisesRegex(FormatError, "exact variant"):
            KvPackPayload.from_bytes(KV_PACK_PAYLOAD.pack(*fields))
        with self.assertRaisesRegex(FormatError, "fixed size"):
            KvPackPayload.from_bytes(encoded[:-1])

    def test_receipt_requires_bit_exact_capture_racecheck_and_no_ptx(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_outputs_bit_exact", False, "full outputs"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("latency_ms", 0.0, "finite and positive"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_region_lowering_pairs_kv_and_elides_singleton_v_transpose(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_kv_pack_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        lowering = lower_kv_pack_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(len(lowering.fused_execution_indices), 2)
        self.assertEqual(len(lowering.elided_execution_indices), 3)
        self.assertEqual(len(lowering.unhandled_execution_indices), 2)
        self.assertEqual(
            tuple(operand.access for operand in lowering.commands[0].command.operands),
            (
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.WRITE,
                OperandAccess.WRITE,
            ),
        )
        again = lower_kv_pack_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_kv_pack_partial_lowering(lowering),
            dump_kv_pack_partial_lowering(again),
        )

    def test_region_refuses_public_intermediate_and_wrong_target(self) -> None:
        program = _program(expose_intermediate=True)
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        with self.assertRaisesRegex(ValidationError, "public output"):
            lower_kv_pack_commands(
                schedule, inventory, memory, _receipt(), target_arch=CudaArch.SM120
            )
        anchor = next(item for item in inventory.ops if item.opcode == "concat")
        with self.assertRaisesRegex(ValidationError, "requires SM120"):
            KvPackProblem.from_inventory(anchor, target_arch=CudaArch.SM110)


if __name__ == "__main__":
    unittest.main()
