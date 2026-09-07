from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import IntEnum

from ..errors import FormatError, ValidationError
from ..schema import CudaArch


COMMAND_STREAM_MAGIC = b"AICMD1\0\0"
COMMAND_STREAM_SCHEMA_MAJOR = 1
COMMAND_STREAM_SCHEMA_MINOR = 0
COMMAND_STREAM_HEADER = struct.Struct("<8sHHIIIIII" + "Q" * 9 + "32s32s20s")
COMMAND_RECORD = struct.Struct("<IIIIIIIIQQQ32s8s")
COMMAND_OPERAND = struct.Struct("<IIQ")
MAX_COMMANDS = 1_000_000
MAX_OPERANDS = 8_000_000
MAX_COMMAND_PAYLOAD = 64 * 1024 * 1024

assert COMMAND_STREAM_HEADER.size == 192
assert COMMAND_RECORD.size == 96
assert COMMAND_OPERAND.size == 16


class CommandTag(IntEnum):
    CUDA_KERNEL = 1
    CUBLASLT_MATMUL = 2
    ATTENTION = 3
    MEMORY_COPY = 4


class OperandAccess(IntEnum):
    READ = 1
    WRITE = 2
    READ_WRITE = 3


@dataclass(frozen=True, slots=True)
class CommandOperand:
    value_id: int
    access: OperandAccess
    byte_offset: int = 0

    def __post_init__(self) -> None:
        _uint(self.value_id, 32, "command operand value_id")
        if not isinstance(self.access, OperandAccess):
            raise ValidationError("command operand access is unknown")
        _uint(self.byte_offset, 64, "command operand byte_offset")


@dataclass(frozen=True, slots=True)
class ProviderCommand:
    tag: CommandTag
    provider_id: int
    abi_major: int
    abi_minor: int
    capability_digest: str
    operands: tuple[CommandOperand, ...]
    payload: bytes
    workspace_offset: int = 0
    workspace_bytes: int = 0
    capture_safe: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tag, CommandTag):
            raise ValidationError("provider command tag is unknown")
        _positive_uint(self.provider_id, 32, "provider command provider_id")
        _positive_uint(self.abi_major, 32, "provider command abi_major")
        _uint(self.abi_minor, 32, "provider command abi_minor")
        _digest(self.capability_digest, "provider command capability_digest")
        if (
            not isinstance(self.operands, tuple)
            or not self.operands
            or len(self.operands) > MAX_OPERANDS
            or not all(isinstance(item, CommandOperand) for item in self.operands)
        ):
            raise ValidationError("provider command needs a bounded operand tuple")
        if not any(
            item.access in {OperandAccess.WRITE, OperandAccess.READ_WRITE}
            for item in self.operands
        ):
            raise ValidationError("provider command needs at least one write operand")
        if not isinstance(self.payload, bytes) or len(self.payload) > MAX_COMMAND_PAYLOAD:
            raise ValidationError("provider command payload must be bounded bytes")
        _uint(self.workspace_offset, 64, "provider command workspace_offset")
        _uint(self.workspace_bytes, 64, "provider command workspace_bytes")
        if self.workspace_bytes == 0 and self.workspace_offset != 0:
            raise ValidationError("zero-size provider workspace must use offset zero")
        if not isinstance(self.capture_safe, bool):
            raise ValidationError("provider command capture_safe must be boolean")


@dataclass(frozen=True, slots=True)
class CommandStream:
    target_arch: CudaArch
    memory_plan_sha256: str
    value_count: int
    arena_bytes: int
    state_bytes: int
    workspace_bytes: int
    commands: tuple[ProviderCommand, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target_arch, CudaArch):
            raise ValidationError("command stream target_arch must be a CudaArch")
        _digest(self.memory_plan_sha256, "command stream memory_plan_sha256")
        _positive_uint(self.value_count, 32, "command stream value_count")
        _uint(self.arena_bytes, 64, "command stream arena_bytes")
        _uint(self.state_bytes, 64, "command stream state_bytes")
        _uint(self.workspace_bytes, 64, "command stream workspace_bytes")
        if (
            not isinstance(self.commands, tuple)
            or not self.commands
            or len(self.commands) > MAX_COMMANDS
            or not all(isinstance(item, ProviderCommand) for item in self.commands)
        ):
            raise ValidationError("command stream needs a bounded command tuple")
        operand_count = sum(len(command.operands) for command in self.commands)
        if operand_count > MAX_OPERANDS:
            raise ValidationError("command stream operand count exceeds its limit")
        for command in self.commands:
            for operand in command.operands:
                if operand.value_id >= self.value_count:
                    raise ValidationError("command operand value_id exceeds the stream value table")
            if (
                command.workspace_offset > self.workspace_bytes
                or command.workspace_bytes
                > self.workspace_bytes - command.workspace_offset
            ):
                raise ValidationError("provider command workspace exceeds the stream workspace")

    def to_bytes(self) -> bytes:
        command_data = bytearray()
        operand_data = bytearray()
        payload_data = bytearray()
        first_operand = 0
        for command in self.commands:
            payload_offset = len(payload_data)
            payload_data.extend(command.payload)
            command_data.extend(
                COMMAND_RECORD.pack(
                    int(command.tag),
                    command.provider_id,
                    command.abi_major,
                    command.abi_minor,
                    1 if command.capture_safe else 0,
                    first_operand,
                    len(command.operands),
                    len(command.payload),
                    payload_offset,
                    command.workspace_offset,
                    command.workspace_bytes,
                    bytes.fromhex(command.capability_digest),
                    b"\0" * 8,
                )
            )
            for operand in command.operands:
                operand_data.extend(
                    COMMAND_OPERAND.pack(
                        operand.value_id,
                        int(operand.access),
                        operand.byte_offset,
                    )
                )
            first_operand += len(command.operands)

        commands_offset = COMMAND_STREAM_HEADER.size
        operands_offset = commands_offset + len(command_data)
        payload_offset = operands_offset + len(operand_data)
        section_size = payload_offset + len(payload_data)
        if section_size > 2**64 - 1:
            raise ValidationError("command stream section size overflows uint64")
        body = bytes(command_data + operand_data + payload_data)
        header = COMMAND_STREAM_HEADER.pack(
            COMMAND_STREAM_MAGIC,
            COMMAND_STREAM_SCHEMA_MAJOR,
            COMMAND_STREAM_SCHEMA_MINOR,
            COMMAND_STREAM_HEADER.size,
            int(self.target_arch),
            len(self.commands),
            first_operand,
            0,
            0,
            self.value_count,
            self.arena_bytes,
            self.state_bytes,
            self.workspace_bytes,
            commands_offset,
            operands_offset,
            payload_offset,
            len(payload_data),
            section_size,
            bytes.fromhex(self.memory_plan_sha256),
            hashlib.sha256(body).digest(),
            b"\0" * 20,
        )
        return header + body

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "CommandStream":
        view = memoryview(data)
        if len(view) < COMMAND_STREAM_HEADER.size:
            raise FormatError("command stream is smaller than its fixed header")
        fields = COMMAND_STREAM_HEADER.unpack_from(view)
        (
            magic,
            schema_major,
            schema_minor,
            header_size,
            raw_arch,
            command_count,
            operand_count,
            flags,
            reserved_u32,
            value_count,
            arena_bytes,
            state_bytes,
            workspace_bytes,
            commands_offset,
            operands_offset,
            payload_offset,
            payload_bytes,
            section_size,
            memory_digest,
            body_digest,
            reserved,
        ) = fields
        if magic != COMMAND_STREAM_MAGIC:
            raise FormatError("bad command stream magic")
        if (
            schema_major != COMMAND_STREAM_SCHEMA_MAJOR
            or schema_minor > COMMAND_STREAM_SCHEMA_MINOR
        ):
            raise FormatError(
                f"unsupported command stream schema {schema_major}.{schema_minor}"
            )
        if header_size != COMMAND_STREAM_HEADER.size or flags != 0 or reserved_u32 != 0 or any(reserved):
            raise FormatError("invalid command stream header, flags, or reserved bytes")
        if command_count == 0 or command_count > MAX_COMMANDS:
            raise FormatError("command stream command count is outside its bounds")
        if operand_count == 0 or operand_count > MAX_OPERANDS:
            raise FormatError("command stream operand count is outside its bounds")
        expected_operands_offset = COMMAND_STREAM_HEADER.size + command_count * COMMAND_RECORD.size
        expected_payload_offset = expected_operands_offset + operand_count * COMMAND_OPERAND.size
        if (
            commands_offset != COMMAND_STREAM_HEADER.size
            or operands_offset != expected_operands_offset
            or payload_offset != expected_payload_offset
            or payload_offset > len(view)
            or payload_bytes != len(view) - payload_offset
            or section_size != len(view)
        ):
            raise FormatError("command stream tables have invalid or non-canonical bounds")
        if not any(memory_digest) or hashlib.sha256(view[commands_offset:]).digest() != body_digest:
            raise FormatError("command stream memory-plan identity or body hash is invalid")
        try:
            target_arch = CudaArch(raw_arch)
        except ValueError as exc:
            raise FormatError(f"command stream target architecture is unknown: {raw_arch}") from exc

        operands: list[CommandOperand] = []
        for index in range(operand_count):
            value_id, raw_access, byte_offset = COMMAND_OPERAND.unpack_from(
                view, operands_offset + index * COMMAND_OPERAND.size
            )
            try:
                access = OperandAccess(raw_access)
                operands.append(CommandOperand(value_id, access, byte_offset))
            except (ValueError, ValidationError) as exc:
                raise FormatError(f"command stream operand {index} is invalid: {exc}") from exc

        commands: list[ProviderCommand] = []
        next_operand = 0
        next_payload = 0
        for index in range(command_count):
            record = COMMAND_RECORD.unpack_from(
                view, commands_offset + index * COMMAND_RECORD.size
            )
            (
                raw_tag,
                provider_id,
                abi_major,
                abi_minor,
                command_flags,
                first_operand,
                selected_operand_count,
                selected_payload_size,
                selected_payload_offset,
                workspace_offset,
                selected_workspace_bytes,
                capability_digest,
                command_reserved,
            ) = record
            if command_flags & ~1 or any(command_reserved):
                raise FormatError(f"command stream command {index} has unknown flags or reserved bytes")
            if (
                first_operand != next_operand
                or selected_operand_count == 0
                or selected_operand_count > operand_count - first_operand
            ):
                raise FormatError(f"command stream command {index} has invalid operand bounds")
            if (
                selected_payload_offset != next_payload
                or selected_payload_size > payload_bytes - selected_payload_offset
            ):
                raise FormatError(f"command stream command {index} has invalid payload bounds")
            try:
                tag = CommandTag(raw_tag)
                commands.append(
                    ProviderCommand(
                        tag=tag,
                        provider_id=provider_id,
                        abi_major=abi_major,
                        abi_minor=abi_minor,
                        capability_digest=capability_digest.hex(),
                        operands=tuple(
                            operands[first_operand : first_operand + selected_operand_count]
                        ),
                        payload=bytes(
                            view[
                                payload_offset
                                + selected_payload_offset : payload_offset
                                + selected_payload_offset
                                + selected_payload_size
                            ]
                        ),
                        workspace_offset=workspace_offset,
                        workspace_bytes=selected_workspace_bytes,
                        capture_safe=bool(command_flags & 1),
                    )
                )
            except (ValueError, ValidationError) as exc:
                raise FormatError(f"command stream command {index} is invalid: {exc}") from exc
            next_operand += selected_operand_count
            next_payload += selected_payload_size
        if next_operand != operand_count or next_payload != payload_bytes:
            raise FormatError("command stream command spans do not consume their complete tables")
        try:
            return cls(
                target_arch=target_arch,
                memory_plan_sha256=memory_digest.hex(),
                value_count=value_count,
                arena_bytes=arena_bytes,
                state_bytes=state_bytes,
                workspace_bytes=workspace_bytes,
                commands=tuple(commands),
            )
        except ValidationError as exc:
            raise FormatError(f"command stream header contract is invalid: {exc}") from exc


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationError(f"{label} must be a nonzero SHA-256 hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValidationError(f"{label} must be a nonzero SHA-256 hex digest") from exc
    if len(decoded) != 32 or not any(decoded) or value != decoded.hex():
        raise ValidationError(f"{label} must be a nonzero SHA-256 hex digest")


def _positive_uint(value: object, bits: int, label: str) -> None:
    _uint(value, bits, label)
    if value == 0:
        raise ValidationError(f"{label} must be positive")


def _uint(value: object, bits: int, label: str) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > 2**bits - 1
    ):
        raise ValidationError(f"{label} must be a uint{bits}")
