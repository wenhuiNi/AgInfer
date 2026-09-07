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
    PREFIX_KV_STORE_PAYLOAD,
    PrefixKvStorePayload,
    PrefixKvStoreProblem,
    PrefixKvStoreValidationReceipt,
    dump_prefix_kv_store_partial_lowering,
    lower_prefix_kv_store_commands,
    make_prefix_kv_store_capabilities,
)
from aginfer.schema import CudaArch


IMPLEMENTATION_DIGEST = "b" * 64


def _program(*, duplicate_target: bool = False, expose_value: bool = False) -> Program:
    query = TensorType(DType.BF16, (1, 8, 968, 256), device=Device.CUDA)
    key_value = TensorType(DType.BF16, (1, 1, 968, 256), device=Device.CUDA)
    value_bshd = TensorType(DType.BF16, (1, 968, 1, 256), device=Device.CUDA)
    positions = TensorType(DType.I32, (1, 968), device=Device.CUDA)
    mask = TensorType(DType.BOOL, (1, 8, 968, 968), device=Device.CUDA)
    outputs = ["attention"]
    if expose_value:
        outputs.append("value")
    return Program(
        functions=(
            Function(
                "main",
                (
                    Value("query", query),
                    Value("key_input", key_value),
                    Value("value_input", value_bshd),
                    Value("positions", positions),
                    Value("mask", mask),
                ),
                tuple(outputs),
                Region(
                    (
                        Op(
                            "rope_default",
                            ("key_input", "positions"),
                            (Value("key", key_value),),
                            (
                                ("frequency_dtype", "bf16"),
                                ("pairing", "split_half"),
                                ("theta", 10000.0),
                            ),
                        ),
                        Op(
                            "transpose",
                            ("value_input",),
                            (Value("value", key_value),),
                            (("permutation", (0, 2, 1, 3)),),
                        ),
                        Op(
                            "state_write",
                            ("key",),
                            (),
                            (("state", "cache_key"),),
                        ),
                        Op(
                            "state_write",
                            ("value",),
                            (),
                            (("state", "cache_key" if duplicate_target else "cache_value"),),
                        ),
                        Op(
                            "scaled_dot_product_attention",
                            ("query", "key", "value", "mask"),
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
            State("cache_key", key_value, StateAccess.READ_WRITE),
            State("cache_value", key_value, StateAccess.READ_WRITE),
        ),
    )


def _receipt() -> PrefixKvStoreValidationReceipt:
    return PrefixKvStoreValidationReceipt(
        PrefixKvStoreProblem(CudaArch.SM120),
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


class AotPrefixKvStoreProviderTests(unittest.TestCase):
    def test_payload_is_fixed_and_fail_closed(self) -> None:
        payload = PrefixKvStorePayload(
            PrefixKvStoreProblem(CudaArch.SM120), 128_000, IMPLEMENTATION_DIGEST
        )
        encoded = payload.to_bytes()
        self.assertEqual(len(encoded), 192)
        self.assertEqual(PrefixKvStorePayload.from_bytes(encoded), payload)
        fields = list(PREFIX_KV_STORE_PAYLOAD.unpack(encoded))
        fields[12] += 1
        with self.assertRaisesRegex(FormatError, "exact variant"):
            PrefixKvStorePayload.from_bytes(PREFIX_KV_STORE_PAYLOAD.pack(*fields))
        with self.assertRaisesRegex(FormatError, "fixed size"):
            PrefixKvStorePayload.from_bytes(encoded[:-1])

    def test_receipt_requires_bit_exact_capture_racecheck_and_no_ptx(self) -> None:
        receipt = _receipt()
        for field, value, message in (
            ("normal_launches", 1, "two normal"),
            ("capture_replay_bit_exact", False, "capture replay"),
            ("full_states_bit_exact", False, "full states"),
            ("racecheck_hazards", 1, "zero racecheck"),
            ("latency_ms", 0.0, "finite and positive"),
            ("contains_ptx", True, "must not contain PTX"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValidationError, message
            ):
                dataclasses.replace(receipt, **{field: value})

    def test_region_pairs_state_writes_and_uses_singleton_view_input(self) -> None:
        program = _program()
        inventory = build_lowering_inventory(program)
        schedule = build_execution_schedule(program)
        memory = build_memory_plan(schedule)
        receipt = _receipt()
        capabilities = make_prefix_kv_store_capabilities(
            inventory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(capabilities), 1)
        lowering = lower_prefix_kv_store_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(len(lowering.commands), 1)
        self.assertEqual(lowering.fused_execution_indices, (2, 3))
        self.assertEqual(lowering.elided_execution_indices, (1,))
        self.assertEqual(len(lowering.unhandled_execution_indices), 2)
        command = lowering.commands[0].command
        self.assertEqual(
            tuple(operand.access for operand in command.operands),
            (
                OperandAccess.READ,
                OperandAccess.READ,
                OperandAccess.WRITE,
                OperandAccess.WRITE,
            ),
        )
        self.assertEqual(
            schedule.values[command.operands[1].value_id].type.shape,
            (1, 968, 1, 256),
        )
        self.assertEqual(
            {command.operands[2].value_id, command.operands[3].value_id},
            {value_id for _, value_id in schedule.states},
        )
        again = lower_prefix_kv_store_commands(
            schedule, inventory, memory, receipt, target_arch=CudaArch.SM120
        )
        self.assertEqual(
            dump_prefix_kv_store_partial_lowering(lowering),
            dump_prefix_kv_store_partial_lowering(again),
        )

    def test_region_refuses_visible_value_duplicate_state_and_wrong_target(self) -> None:
        for program, message in (
            (_program(expose_value=True), "not a proven view"),
            (_program(duplicate_target=True), "must be distinct"),
        ):
            inventory = build_lowering_inventory(program)
            schedule = build_execution_schedule(program)
            memory = build_memory_plan(schedule)
            with self.subTest(message=message), self.assertRaisesRegex(
                ValidationError, message
            ):
                lower_prefix_kv_store_commands(
                    schedule,
                    inventory,
                    memory,
                    _receipt(),
                    target_arch=CudaArch.SM120,
                )
        inventory = build_lowering_inventory(_program())
        with self.assertRaisesRegex(ValidationError, "target differs"):
            make_prefix_kv_store_capabilities(
                inventory, _receipt(), target_arch=CudaArch.SM110
            )


if __name__ == "__main__":
    unittest.main()
