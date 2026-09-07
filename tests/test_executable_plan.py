from __future__ import annotations

import dataclasses
import hashlib
import struct
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.executable import (
    EXECUTABLE_PLAN_HEADER,
    EXECUTABLE_PORT_RECORD,
    EXECUTABLE_PROVIDER_RECORD,
    EXECUTABLE_STATE_RECORD,
    EXECUTABLE_VALUE_RECORD,
    ExecutablePortKind,
    ExecutableValueRegion,
    PackedWeightSpan,
    parse_executable_plan,
)
from tests.executable_fixtures import executable_fixture


def _rehash(data: bytes) -> bytes:
    fields = list(EXECUTABLE_PLAN_HEADER.unpack_from(data))
    values_offset = fields[16]
    fields[26] = hashlib.sha256(data[values_offset:]).digest()
    return EXECUTABLE_PLAN_HEADER.pack(*fields) + data[EXECUTABLE_PLAN_HEADER.size :]


class ExecutablePlanTests(unittest.TestCase):
    def test_fixed_layout_round_trip_and_explicit_storage(self) -> None:
        self.assertEqual(EXECUTABLE_PLAN_HEADER.size, 320)
        self.assertEqual(EXECUTABLE_VALUE_RECORD.size, 160)
        self.assertEqual(EXECUTABLE_PORT_RECORD.size, 32)
        self.assertEqual(EXECUTABLE_STATE_RECORD.size, 32)
        self.assertEqual(EXECUTABLE_PROVIDER_RECORD.size, 64)
        first = executable_fixture()
        second = executable_fixture()
        self.assertEqual(first.data, second.data)
        self.assertEqual(parse_executable_plan(first.data).data, first.data)
        self.assertEqual(first.target_arch.name_string, "sm120")
        self.assertEqual(len(first.ports), 2)
        self.assertEqual(
            tuple(item.kind for item in first.ports),
            (ExecutablePortKind.INPUT, ExecutablePortKind.OUTPUT),
        )
        self.assertEqual(len(first.states), 1)
        self.assertEqual(len(first.providers), 2)
        self.assertEqual(len(first.command_stream.commands), 2)
        regions = {item.region for item in first.values}
        self.assertTrue(
            {
                ExecutableValueRegion.EXTERNAL_INPUT,
                ExecutableValueRegion.EXTERNAL_OUTPUT,
                ExecutableValueRegion.WEIGHTS,
                ExecutableValueRegion.STATE,
                ExecutableValueRegion.ALIAS,
            }.issubset(regions)
        )

    def test_header_corruption_arch_weight_and_body_checks_fail_closed(self) -> None:
        plan = executable_fixture()
        fields = list(EXECUTABLE_PLAN_HEADER.unpack_from(plan.data))
        fields[1] = 3
        with self.assertRaisesRegex(FormatError, "unsupported executable"):
            parse_executable_plan(
                EXECUTABLE_PLAN_HEADER.pack(*fields)
                + plan.data[EXECUTABLE_PLAN_HEADER.size :]
            )
        with self.assertRaisesRegex(FormatError, "architecture differs"):
            from aginfer.schema import CudaArch

            parse_executable_plan(plan.data, expected_arch=CudaArch.SM89)
        with self.assertRaisesRegex(FormatError, "weight size differs"):
            parse_executable_plan(plan.data, expected_weights_bytes=257)
        corrupted = bytearray(plan.data)
        corrupted[-1] ^= 1
        with self.assertRaisesRegex(FormatError, "body checksum"):
            parse_executable_plan(corrupted)
        with self.assertRaisesRegex(FormatError, "fixed header"):
            parse_executable_plan(plan.data[:100])

    def test_value_port_provider_and_command_cross_checks_fail_closed(self) -> None:
        plan = executable_fixture()
        fields = EXECUTABLE_PLAN_HEADER.unpack_from(plan.data)

        bad_value = bytearray(plan.data)
        record = list(EXECUTABLE_VALUE_RECORD.unpack_from(bad_value, fields[16]))
        record[1] = 99
        EXECUTABLE_VALUE_RECORD.pack_into(bad_value, fields[16], *record)
        with self.assertRaisesRegex(FormatError, "unknown enum"):
            parse_executable_plan(_rehash(bytes(bad_value)))

        bad_port = bytearray(plan.data)
        port = list(EXECUTABLE_PORT_RECORD.unpack_from(bad_port, fields[17]))
        port[2] = len(plan.values)
        EXECUTABLE_PORT_RECORD.pack_into(bad_port, fields[17], *port)
        with self.assertRaisesRegex(FormatError, "port 0"):
            parse_executable_plan(_rehash(bytes(bad_port)))

        bad_provider = bytearray(plan.data)
        provider = list(
            EXECUTABLE_PROVIDER_RECORD.unpack_from(bad_provider, fields[19])
        )
        provider[4] += 1
        EXECUTABLE_PROVIDER_RECORD.pack_into(
            bad_provider, fields[19], *provider
        )
        with self.assertRaisesRegex(FormatError, "provider table differs"):
            parse_executable_plan(_rehash(bytes(bad_provider)))

        bad_command = bytearray(plan.data)
        bad_command[-1] ^= 1
        with self.assertRaisesRegex(FormatError, "body checksum"):
            parse_executable_plan(bytes(bad_command))

    def test_constructor_rejects_missing_misaligned_or_wrong_weight_span(self) -> None:
        plan = executable_fixture()
        weight = next(
            item for item in plan.values if item.region == ExecutableValueRegion.WEIGHTS
        )
        with self.assertRaisesRegex(ValidationError, "uint64"):
            PackedWeightSpan(weight.value_id, -1, 4, "1" * 64)
        with self.assertRaisesRegex(ValidationError, "canonical nonzero"):
            PackedWeightSpan(weight.value_id, 0, 4, "0" * 64)
        with self.assertRaisesRegex(ValidationError, "canonical nonzero"):
            dataclasses.replace(
                PackedWeightSpan(
                    weight.value_id, 0, 4, hashlib.sha256(b"x").hexdigest()
                ),
                sha256=hashlib.sha256(b"x").hexdigest().upper(),
            )

    def test_64_bit_weight_range_is_preserved(self) -> None:
        plan = executable_fixture()
        fields = list(EXECUTABLE_PLAN_HEADER.unpack_from(plan.data))
        fields[15] = 2**32 + 4096
        widened = EXECUTABLE_PLAN_HEADER.pack(*fields) + plan.data[EXECUTABLE_PLAN_HEADER.size :]
        parsed = parse_executable_plan(widened)
        self.assertEqual(parsed.weights_bytes, 2**32 + 4096)


if __name__ == "__main__":
    unittest.main()
