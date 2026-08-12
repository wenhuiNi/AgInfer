from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from .errors import CompatibilityError, FormatError, ValidationError
from .schema import ALIGNMENT, RUNTIME_ABI, SCHEMA_MAJOR, SCHEMA_MINOR, CudaArch, Platform, validate_target

MAGIC = b"AIMAOT1\0"
ENDIAN_TAG = 0x01020304
HEADER_STRUCT = struct.Struct("<8sHHIIIII" + "Q" * 9 + "32s32s32s32s24s")
VARIANT_STRUCT = struct.Struct("<II" + "Q" * 6 + "32s32s32s40s")
HEADER_SIZE = HEADER_STRUCT.size
VARIANT_SIZE = VARIANT_STRUCT.size
FILE_HASH_OFFSET = 136
FILE_HASH_SIZE = 32

assert HEADER_SIZE == 256
assert VARIANT_SIZE == 192


def _sha256(data: bytes | memoryview) -> bytes:
    return hashlib.sha256(data).digest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def _zeroed_file_hash(data: bytes | bytearray | memoryview) -> bytes:
    digest = hashlib.sha256()
    digest.update(data[:FILE_HASH_OFFSET])
    digest.update(b"\0" * FILE_HASH_SIZE)
    digest.update(data[FILE_HASH_OFFSET + FILE_HASH_SIZE :])
    return digest.digest()


def _reject_ptx(blob: bytes, label: str) -> None:
    sample = blob[: 1024 * 1024]
    markers = (b".version ", b".target sm_", b".entry ", b".visible .entry")
    if any(marker in sample for marker in markers):
        raise ValidationError(f"{label} appears to contain PTX; AIM accepts CUBIN/SASS only")


def _validate_cubin(blob: bytes, arch: CudaArch, label: str, error_type: type[Exception] = ValidationError) -> None:
    if len(blob) < 64 or blob[:7] != b"\x7fELF\x02\x01\x01" or struct.unpack_from("<H", blob, 18)[0] != 190:
        raise error_type(f"{label} is not a 64-bit little-endian NVIDIA CUDA ELF CUBIN")
    flags = struct.unpack_from("<I", blob, 48)[0]
    targets = {flags & 0xFF, (flags >> 16) & 0xFF}
    if int(arch) not in targets:
        raise error_type(f"{label} ELF flags do not target exact {arch.name_string}")


@dataclass(frozen=True)
class VariantPayload:
    arch: CudaArch
    kernels: bytes
    weights: bytes
    plan: bytes


@dataclass(frozen=True)
class Section:
    offset: int
    size: int
    sha256: bytes


@dataclass(frozen=True)
class VariantInfo:
    arch: CudaArch
    kernels: Section
    weights: Section
    plan: Section


@dataclass(frozen=True)
class AimInfo:
    path: Path
    schema_major: int
    schema_minor: int
    runtime_abi: int
    platform: Platform
    manifest: dict[str, Any]
    graph: dict[str, Any]
    tensors: dict[str, Any]
    variants: tuple[VariantInfo, ...]
    file_size: int
    file_sha256: str

    def select_variant(self, arch: CudaArch) -> VariantInfo:
        for variant in self.variants:
            if variant.arch == arch:
                return variant
        available = ", ".join(item.arch.name_string for item in self.variants)
        raise CompatibilityError(
            f"AIM has no exact {arch.name_string} variant (available: {available or 'none'}); fallback is forbidden"
        )


class AimWriter:
    """Deterministic AIM v1 writer with content-addressed payload deduplication."""

    @staticmethod
    def write(
        path: str | os.PathLike[str],
        *,
        platform: Platform,
        manifest: dict[str, Any],
        graph: dict[str, Any],
        tensors: dict[str, Any],
        variants: Iterable[VariantPayload],
        runtime_abi: int = RUNTIME_ABI,
    ) -> AimInfo:
        variant_list = sorted(list(variants), key=lambda item: int(item.arch))
        validate_target(platform, [item.arch for item in variant_list])
        if runtime_abi <= 0:
            raise ValidationError("runtime ABI must be positive")
        for item in variant_list:
            if not item.kernels:
                raise ValidationError(f"{item.arch.name_string}: empty kernel bundle")
            if not item.weights:
                raise ValidationError(f"{item.arch.name_string}: empty weight blob")
            if not item.plan:
                raise ValidationError(f"{item.arch.name_string}: empty execution plan")
            _reject_ptx(item.kernels, f"{item.arch.name_string} kernel bundle")
            _validate_cubin(item.kernels, item.arch, f"{item.arch.name_string} kernel bundle")

        manifest_bytes = _canonical_json(manifest)
        graph_bytes = _canonical_json(graph)
        tensor_bytes = _canonical_json(tensors)
        output = bytearray(HEADER_SIZE)
        sections: dict[tuple[str, bytes], Section] = {}

        def add(kind: str, data: bytes) -> Section:
            digest = _sha256(data)
            key = (kind, digest)
            if key in sections and output[sections[key].offset : sections[key].offset + sections[key].size] == data:
                return sections[key]
            padding = _align(len(output)) - len(output)
            output.extend(b"\0" * padding)
            section = Section(len(output), len(data), digest)
            output.extend(data)
            sections[key] = section
            return section

        manifest_section = add("manifest", manifest_bytes)
        graph_section = add("graph", graph_bytes)
        tensor_section = add("tensors", tensor_bytes)
        variant_sections: list[tuple[VariantPayload, Section, Section, Section]] = []
        for item in variant_list:
            variant_sections.append(
                (item, add("kernels", item.kernels), add("weights", item.weights), add("plan", item.plan))
            )

        output.extend(b"\0" * (_align(len(output)) - len(output)))
        directory_offset = len(output)
        for item, kernels, weights, plan in variant_sections:
            output.extend(
                VARIANT_STRUCT.pack(
                    int(item.arch),
                    0,
                    kernels.offset,
                    kernels.size,
                    weights.offset,
                    weights.size,
                    plan.offset,
                    plan.size,
                    kernels.sha256,
                    weights.sha256,
                    plan.sha256,
                    b"\0" * 40,
                )
            )
        directory_size = len(output) - directory_offset
        directory_hash = _sha256(output[directory_offset:])
        file_size = len(output)
        header = HEADER_STRUCT.pack(
            MAGIC,
            SCHEMA_MAJOR,
            SCHEMA_MINOR,
            HEADER_SIZE,
            runtime_abi,
            int(platform),
            ENDIAN_TAG,
            len(variant_list),
            file_size,
            manifest_section.offset,
            manifest_section.size,
            graph_section.offset,
            graph_section.size,
            tensor_section.offset,
            tensor_section.size,
            directory_offset,
            directory_size,
            directory_hash,
            b"\0" * 32,
            manifest_section.sha256,
            graph_section.sha256,
            b"\0" * 24,
        )
        output[:HEADER_SIZE] = header
        file_hash = _zeroed_file_hash(output)
        output[FILE_HASH_OFFSET : FILE_HASH_OFFSET + FILE_HASH_SIZE] = file_hash

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("wb") as stream:
                stream.write(output)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return AimReader.read(destination)


class AimReader:
    @staticmethod
    def read(path: str | os.PathLike[str], *, verify_payloads: bool = True) -> AimInfo:
        source = Path(path)
        try:
            stream: BinaryIO
            with source.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size < HEADER_SIZE:
                    raise FormatError("file is smaller than the AIM header")
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    return AimReader._parse(source, data, verify_payloads)
        except OSError as exc:
            raise FormatError(f"cannot read AIM {source}: {exc}") from exc

    @staticmethod
    def _parse(path: Path, data: mmap.mmap, verify_payloads: bool) -> AimInfo:
        fields = HEADER_STRUCT.unpack_from(data, 0)
        (
            magic,
            schema_major,
            schema_minor,
            header_size,
            runtime_abi,
            platform_raw,
            endian_tag,
            variant_count,
            file_size,
            manifest_offset,
            manifest_size,
            graph_offset,
            graph_size,
            tensor_offset,
            tensor_size,
            directory_offset,
            directory_size,
            directory_hash,
            file_hash,
            manifest_hash,
            graph_hash,
            reserved,
        ) = fields
        if magic != MAGIC:
            raise FormatError("bad AIM magic")
        if schema_major != SCHEMA_MAJOR:
            raise FormatError(f"unsupported AIM schema {schema_major}.{schema_minor}")
        if header_size != HEADER_SIZE or endian_tag != ENDIAN_TAG:
            raise FormatError("unsupported AIM header or byte order")
        if file_size != len(data):
            raise FormatError(f"file size mismatch: header={file_size}, actual={len(data)}")
        if any(reserved):
            raise FormatError("non-zero reserved header bytes")
        try:
            platform = Platform(platform_raw)
        except ValueError as exc:
            raise FormatError(f"unknown platform id {platform_raw}") from exc
        AimReader._check_region(len(data), directory_offset, directory_size, "variant directory")
        if directory_size != variant_count * VARIANT_SIZE:
            raise FormatError("variant directory size does not match variant count")
        if _sha256(data[directory_offset : directory_offset + directory_size]) != directory_hash:
            raise FormatError("variant directory checksum mismatch")
        if _zeroed_file_hash(data) != file_hash:
            raise FormatError("file checksum mismatch")

        manifest = AimReader._json_section(data, manifest_offset, manifest_size, manifest_hash, "manifest")
        graph = AimReader._json_section(data, graph_offset, graph_size, graph_hash, "graph")
        tensor_data = AimReader._slice(data, tensor_offset, tensor_size, "tensor table")
        try:
            tensors = json.loads(tensor_data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatError(f"invalid tensor table JSON: {exc}") from exc

        variants: list[VariantInfo] = []
        seen: set[CudaArch] = set()
        for index in range(variant_count):
            values = VARIANT_STRUCT.unpack_from(data, directory_offset + index * VARIANT_SIZE)
            arch_raw, flags = values[:2]
            offsets = values[2:8]
            hashes = values[8:11]
            variant_reserved = values[11]
            if flags != 0 or any(variant_reserved):
                raise FormatError(f"variant {index} has unsupported flags or reserved data")
            try:
                arch = CudaArch(arch_raw)
            except ValueError as exc:
                raise FormatError(f"variant {index} has unsupported arch sm{arch_raw}") from exc
            if arch in seen:
                raise FormatError(f"duplicate {arch.name_string} variant")
            seen.add(arch)
            sections_list: list[Section] = []
            for section_index, label in enumerate(("kernels", "weights", "plan")):
                offset, size = offsets[section_index * 2 : section_index * 2 + 2]
                if offset % ALIGNMENT != 0:
                    raise FormatError(f"{arch.name_string} {label} is not {ALIGNMENT}-byte aligned")
                content = AimReader._slice(data, offset, size, f"{arch.name_string} {label}")
                if verify_payloads and _sha256(content) != hashes[section_index]:
                    raise FormatError(f"{arch.name_string} {label} checksum mismatch")
                if label == "kernels":
                    _reject_ptx(bytes(content), f"{arch.name_string} kernel bundle")
                    _validate_cubin(content, arch, f"{arch.name_string} kernel bundle", FormatError)
                sections_list.append(Section(offset, size, hashes[section_index]))
            variants.append(VariantInfo(arch, *sections_list))
        try:
            validate_target(platform, [item.arch for item in variants])
        except ValueError as exc:
            raise FormatError(str(exc)) from exc
        return AimInfo(
            path=path,
            schema_major=schema_major,
            schema_minor=schema_minor,
            runtime_abi=runtime_abi,
            platform=platform,
            manifest=manifest,
            graph=graph,
            tensors=tensors,
            variants=tuple(variants),
            file_size=file_size,
            file_sha256=file_hash.hex(),
        )

    @staticmethod
    def _check_region(total: int, offset: int, size: int, label: str) -> None:
        if offset < HEADER_SIZE or size <= 0 or offset > total or size > total - offset:
            raise FormatError(f"invalid {label} region: offset={offset}, size={size}")

    @staticmethod
    def _slice(data: mmap.mmap, offset: int, size: int, label: str) -> bytes:
        AimReader._check_region(len(data), offset, size, label)
        return data[offset : offset + size]

    @staticmethod
    def _json_section(data: mmap.mmap, offset: int, size: int, digest: bytes, label: str) -> dict[str, Any]:
        content = AimReader._slice(data, offset, size, label)
        if _sha256(content) != digest:
            raise FormatError(f"{label} checksum mismatch")
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatError(f"invalid {label} JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise FormatError(f"{label} must be a JSON object")
        return value
