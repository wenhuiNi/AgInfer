from __future__ import annotations

import dataclasses
import hashlib
import unittest

from aginfer.errors import FormatError, ValidationError
from aginfer.lowering import (
    COMMAND_OPERAND,
    COMMAND_RECORD,
    COMMAND_STREAM_HEADER,
    CommandOperand,
    CommandStream,
    CommandTag,
    OperandAccess,
    ProviderCommand,
)
from aginfer.schema import CudaArch


def _digest(label: bytes) -> str:
    return hashlib.sha256(label).hexdigest()


def _command(
    *,
    tag: CommandTag = CommandTag.MEMORY_COPY,
    provider_id: int = 7,
    operands: tuple[CommandOperand, ...] | None = None,
    payload: bytes = b"copy-v1",
    workspace_offset: int = 0,
    workspace_bytes: int = 0,
    capture_safe: bool = True,
) -> ProviderCommand:
    return ProviderCommand(
        tag=tag,
        provider_id=provider_id,
        abi_major=1,
        abi_minor=2,
        capability_digest=_digest(f"capability-{provider_id}".encode()),
        operands=operands
        or (
            CommandOperand(0, OperandAccess.READ, 4),
            CommandOperand(1, OperandAccess.WRITE),
        ),
        payload=payload,
        workspace_offset=workspace_offset,
        workspace_bytes=workspace_bytes,
        capture_safe=capture_safe,
    )


def _stream() -> CommandStream:
    return CommandStream(
        target_arch=CudaArch.SM120,
        memory_plan_sha256=_digest(b"memory-plan"),
        value_count=4,
        arena_bytes=4096,
        state_bytes=256,
        workspace_bytes=1024,
        commands=(
            _command(),
            _command(
                tag=CommandTag.CUBLASLT_MATMUL,
                provider_id=8,
                operands=(
                    CommandOperand(1, OperandAccess.READ),
                    CommandOperand(2, OperandAccess.READ, 128),
                    CommandOperand(3, OperandAccess.WRITE),
                ),
                payload=b"fixed-cublaslt-descriptor",
                workspace_offset=256,
                workspace_bytes=512,
                capture_safe=False,
            ),
        ),
    )


def _replace_header(encoded: bytes, index: int, value: object) -> bytes:
    fields = list(COMMAND_STREAM_HEADER.unpack_from(encoded))
    fields[index] = value
    return COMMAND_STREAM_HEADER.pack(*fields) + encoded[COMMAND_STREAM_HEADER.size :]


def _rehash(encoded: bytes) -> bytes:
    fields = list(COMMAND_STREAM_HEADER.unpack_from(encoded))
    commands_offset = fields[13]
    assert isinstance(commands_offset, int)
    fields[19] = hashlib.sha256(encoded[commands_offset:]).digest()
    return COMMAND_STREAM_HEADER.pack(*fields) + encoded[COMMAND_STREAM_HEADER.size :]


class CommandStreamTests(unittest.TestCase):
    def test_fixed_layout_round_trip_and_encoding_are_deterministic(self) -> None:
        self.assertEqual(COMMAND_STREAM_HEADER.size, 192)
        self.assertEqual(COMMAND_RECORD.size, 96)
        self.assertEqual(COMMAND_OPERAND.size, 16)

        first = _stream().to_bytes()
        second = _stream().to_bytes()
        self.assertEqual(first, second)
        decoded = CommandStream.from_bytes(first)
        self.assertEqual(decoded, _stream())
        self.assertEqual(decoded.to_bytes(), first)

        fields = COMMAND_STREAM_HEADER.unpack_from(first)
        self.assertEqual(fields[13], COMMAND_STREAM_HEADER.size)
        self.assertEqual(fields[14], fields[13] + 2 * COMMAND_RECORD.size)
        self.assertEqual(fields[15], fields[14] + 5 * COMMAND_OPERAND.size)
        self.assertEqual(fields[17], len(first))
        self.assertEqual(fields[19], hashlib.sha256(first[fields[13] :]).digest())

    def test_corruption_truncation_and_noncanonical_header_fail_closed(self) -> None:
        encoded = _stream().to_bytes()
        corrupted = bytearray(encoded)
        corrupted[-1] ^= 1
        with self.assertRaisesRegex(FormatError, "body hash"):
            CommandStream.from_bytes(corrupted)
        with self.assertRaisesRegex(FormatError, "fixed header"):
            CommandStream.from_bytes(encoded[:100])
        with self.assertRaisesRegex(FormatError, "non-canonical bounds"):
            CommandStream.from_bytes(encoded[:-1])
        with self.assertRaisesRegex(FormatError, "unsupported.*schema"):
            CommandStream.from_bytes(_replace_header(encoded, 1, 2))
        with self.assertRaisesRegex(FormatError, "flags.*reserved"):
            CommandStream.from_bytes(_replace_header(encoded, 7, 1))
        with self.assertRaisesRegex(FormatError, "flags.*reserved"):
            CommandStream.from_bytes(_replace_header(encoded, 20, b"x" + b"\0" * 19))
        with self.assertRaisesRegex(FormatError, "non-canonical bounds"):
            CommandStream.from_bytes(_replace_header(encoded, 14, 193))

    def test_unknown_tag_access_and_command_flags_fail_after_valid_hash(self) -> None:
        encoded = _stream().to_bytes()
        fields = COMMAND_STREAM_HEADER.unpack_from(encoded)

        bad_tag = bytearray(encoded)
        COMMAND_RECORD.pack_into(
            bad_tag,
            fields[13],
            99,
            *COMMAND_RECORD.unpack_from(encoded, fields[13])[1:],
        )
        with self.assertRaisesRegex(FormatError, "command 0 is invalid"):
            CommandStream.from_bytes(_rehash(bytes(bad_tag)))

        bad_access = bytearray(encoded)
        operand = list(COMMAND_OPERAND.unpack_from(encoded, fields[14]))
        operand[1] = 99
        COMMAND_OPERAND.pack_into(bad_access, fields[14], *operand)
        with self.assertRaisesRegex(FormatError, "operand 0 is invalid"):
            CommandStream.from_bytes(_rehash(bytes(bad_access)))

        bad_flags = bytearray(encoded)
        command = list(COMMAND_RECORD.unpack_from(encoded, fields[13]))
        command[4] = 2
        COMMAND_RECORD.pack_into(bad_flags, fields[13], *command)
        with self.assertRaisesRegex(FormatError, "unknown flags"):
            CommandStream.from_bytes(_rehash(bytes(bad_flags)))

    def test_record_value_workspace_and_payload_bounds_fail_closed(self) -> None:
        encoded = _stream().to_bytes()
        fields = COMMAND_STREAM_HEADER.unpack_from(encoded)

        bad_value = bytearray(encoded)
        operand = list(COMMAND_OPERAND.unpack_from(encoded, fields[14]))
        operand[0] = 4
        COMMAND_OPERAND.pack_into(bad_value, fields[14], *operand)
        with self.assertRaisesRegex(FormatError, "value_id exceeds"):
            CommandStream.from_bytes(_rehash(bytes(bad_value)))

        bad_workspace = bytearray(encoded)
        command = list(COMMAND_RECORD.unpack_from(encoded, fields[13]))
        command[9] = 1024
        command[10] = 1
        COMMAND_RECORD.pack_into(bad_workspace, fields[13], *command)
        with self.assertRaisesRegex(FormatError, "workspace exceeds"):
            CommandStream.from_bytes(_rehash(bytes(bad_workspace)))

        bad_payload = bytearray(encoded)
        command = list(COMMAND_RECORD.unpack_from(encoded, fields[13]))
        command[7] = fields[16] + 1
        COMMAND_RECORD.pack_into(bad_payload, fields[13], *command)
        with self.assertRaisesRegex(FormatError, "payload bounds"):
            CommandStream.from_bytes(_rehash(bytes(bad_payload)))

        bad_operand_span = bytearray(encoded)
        command = list(COMMAND_RECORD.unpack_from(encoded, fields[13]))
        command[5] = 1
        COMMAND_RECORD.pack_into(bad_operand_span, fields[13], *command)
        with self.assertRaisesRegex(FormatError, "operand bounds"):
            CommandStream.from_bytes(_rehash(bytes(bad_operand_span)))

    def test_constructor_rejects_invalid_commands_and_stream_bounds(self) -> None:
        base = _command()
        with self.assertRaisesRegex(ValidationError, "provider_id"):
            dataclasses.replace(base, provider_id=0)
        with self.assertRaisesRegex(ValidationError, "nonzero SHA-256"):
            dataclasses.replace(base, capability_digest="0" * 64)
        with self.assertRaisesRegex(ValidationError, "nonzero SHA-256"):
            dataclasses.replace(base, capability_digest=_digest(b"x").upper())
        with self.assertRaisesRegex(ValidationError, "write operand"):
            dataclasses.replace(
                base, operands=(CommandOperand(0, OperandAccess.READ),)
            )
        with self.assertRaisesRegex(ValidationError, "offset zero"):
            dataclasses.replace(base, workspace_offset=1, workspace_bytes=0)
        with self.assertRaisesRegex(ValidationError, "bounded bytes"):
            dataclasses.replace(base, payload=bytearray())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "value_id exceeds"):
            dataclasses.replace(_stream(), value_count=1)
        with self.assertRaisesRegex(ValidationError, "workspace exceeds"):
            dataclasses.replace(_stream(), workspace_bytes=767)
        with self.assertRaisesRegex(ValidationError, "boolean"):
            dataclasses.replace(base, capture_safe=1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
