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

MAGIC = b"AIMAOT2\0"
ENDIAN_TAG = 0x01020304
HEADER_STRUCT = struct.Struct("<8sHHIIIII" + "Q" * 11 + "32s" * 5 + "40s")
VARIANT_STRUCT = struct.Struct("<II" + "Q" * 6 + "32s32s32s40s")
COMPATIBILITY_HEADER_STRUCT = struct.Struct("<8sHH" + "I" * 7 + "QQ8s")
PROVIDER_REQUIREMENT_STRUCT = struct.Struct("<IIII16s")
HEADER_SIZE = HEADER_STRUCT.size
VARIANT_SIZE = VARIANT_STRUCT.size
FILE_HASH_OFFSET = 152
FILE_HASH_SIZE = 32
COMPATIBILITY_MAGIC = b"AIMCMP1\0"
COMPATIBILITY_SCHEMA_MAJOR = 1
COMPATIBILITY_SCHEMA_MINOR = 0
MAX_PROVIDER_REQUIREMENTS = 256

assert HEADER_SIZE == 320
assert VARIANT_SIZE == 192
assert COMPATIBILITY_HEADER_STRUCT.size == 64
assert PROVIDER_REQUIREMENT_STRUCT.size == 32


def _sha256(data: bytes | memoryview) -> bytes:
    return hashlib.sha256(data).digest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def _zeroed_file_hash(data: bytes | bytearray | memoryview) -> bytes:
    digest = hashlib.sha256()
    view = memoryview(data)
    try:
        digest.update(view[:FILE_HASH_OFFSET])
        digest.update(b"\0" * FILE_HASH_SIZE)
        digest.update(view[FILE_HASH_OFFSET + FILE_HASH_SIZE :])
    finally:
        view.release()
    return digest.digest()


def _sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> tuple[int, bytes]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ValidationError(f"cannot read AIM payload {path}: {exc}") from exc
    return size, digest.digest()


def _zeroed_stream_hash(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> bytes:
    digest = hashlib.sha256()
    position = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = bytearray(stream.read(chunk_size))
                if not chunk:
                    break
                begin = max(FILE_HASH_OFFSET - position, 0)
                end = min(FILE_HASH_OFFSET + FILE_HASH_SIZE - position, len(chunk))
                if begin < end:
                    chunk[begin:end] = bytes(end - begin)
                digest.update(chunk)
                position += len(chunk)
    except OSError as exc:
        raise ValidationError(f"cannot hash AIM output {path}: {exc}") from exc
    return digest.digest()


def _reject_ptx(blob: bytes | memoryview, label: str) -> None:
    sample = bytes(blob[: 1024 * 1024])
    markers = (b".version ", b".target sm_", b".entry ", b".visible .entry")
    if any(marker in sample for marker in markers):
        raise ValidationError(f"{label} appears to contain PTX; AIM accepts CUBIN/SASS only")


def _validate_cubin(blob: bytes, arch: CudaArch, label: str, error_type: type[Exception] = ValidationError) -> None:
    if len(blob) < 64 or blob[:7] != b"\x7fELF\x02\x01\x01" or struct.unpack_from("<H", blob, 18)[0] != 190:
        raise error_type(f"{label} is not a 64-bit little-endian NVIDIA CUDA ELF CUBIN")
    flags = struct.unpack_from("<I", blob, 48)[0]
    # NVIDIA ELF encodes pre-Blackwell targets in the low/upper bytes and
    # Blackwell targets in bits 8..15 (for example sm120 -> 0x06007802).
    targets = {flags & 0xFF, (flags >> 8) & 0xFF, (flags >> 16) & 0xFF}
    if int(arch) not in targets:
        raise error_type(f"{label} ELF flags do not target exact {arch.name_string}")


@dataclass(frozen=True)
class VariantPayload:
    arch: CudaArch
    kernels: bytes
    weights: bytes
    plan: bytes


@dataclass(frozen=True)
class FileVariantPayload:
    """File-backed payloads for bounded-memory production AIM assembly."""

    arch: CudaArch
    kernels: Path
    weights: Path
    plan: Path


@dataclass(frozen=True, order=True)
class ProviderRequirement:
    provider_id: int
    abi_min: int
    abi_max: int


@dataclass(frozen=True)
class Compatibility:
    cuda_driver_min: int
    cuda_driver_max: int = 0
    cuda_runtime_min: int = 0
    cuda_runtime_max: int = 0
    providers: tuple[ProviderRequirement, ...] = ()

    def to_bytes(self) -> bytes:
        _validate_version_range(self.cuda_driver_min, self.cuda_driver_max, "CUDA Driver")
        if self.cuda_driver_min == 0:
            raise ValidationError("CUDA Driver compatibility must declare a positive minimum version")
        _validate_version_range(self.cuda_runtime_min, self.cuda_runtime_max, "CUDA Runtime")
        if len(self.providers) > MAX_PROVIDER_REQUIREMENTS:
            raise ValidationError(f"compatibility table exceeds {MAX_PROVIDER_REQUIREMENTS} provider requirements")
        ordered = tuple(sorted(self.providers))
        seen: set[int] = set()
        for provider in ordered:
            if provider.provider_id <= 0 or provider.provider_id > 2**32 - 1:
                raise ValidationError("provider ID must be a positive uint32")
            if provider.provider_id in seen:
                raise ValidationError(f"duplicate provider requirement ID {provider.provider_id}")
            seen.add(provider.provider_id)
            _validate_version_range(provider.abi_min, provider.abi_max, f"provider {provider.provider_id} ABI")
            if provider.abi_min == 0:
                raise ValidationError(f"provider {provider.provider_id} ABI minimum must be positive")

        section_size = COMPATIBILITY_HEADER_STRUCT.size + len(ordered) * PROVIDER_REQUIREMENT_STRUCT.size
        output = bytearray(
            COMPATIBILITY_HEADER_STRUCT.pack(
                COMPATIBILITY_MAGIC,
                COMPATIBILITY_SCHEMA_MAJOR,
                COMPATIBILITY_SCHEMA_MINOR,
                COMPATIBILITY_HEADER_STRUCT.size,
                self.cuda_driver_min,
                self.cuda_driver_max,
                self.cuda_runtime_min,
                self.cuda_runtime_max,
                len(ordered),
                0,
                COMPATIBILITY_HEADER_STRUCT.size,
                section_size,
                b"\0" * 8,
            )
        )
        for provider in ordered:
            output.extend(
                PROVIDER_REQUIREMENT_STRUCT.pack(
                    provider.provider_id,
                    provider.abi_min,
                    provider.abi_max,
                    0,
                    b"\0" * 16,
                )
            )
        return bytes(output)

    @staticmethod
    def from_bytes(data: bytes | memoryview) -> "Compatibility":
        if len(data) < COMPATIBILITY_HEADER_STRUCT.size:
            raise FormatError("compatibility table is smaller than its fixed header")
        fields = COMPATIBILITY_HEADER_STRUCT.unpack_from(data)
        (
            magic,
            schema_major,
            schema_minor,
            header_size,
            driver_min,
            driver_max,
            runtime_min,
            runtime_max,
            provider_count,
            flags,
            providers_offset,
            section_size,
            reserved,
        ) = fields
        if magic != COMPATIBILITY_MAGIC:
            raise FormatError("bad AIM compatibility table magic")
        if schema_major != COMPATIBILITY_SCHEMA_MAJOR or schema_minor > COMPATIBILITY_SCHEMA_MINOR:
            raise FormatError(f"unsupported AIM compatibility table schema {schema_major}.{schema_minor}")
        if header_size != COMPATIBILITY_HEADER_STRUCT.size or flags != 0 or any(reserved):
            raise FormatError("invalid AIM compatibility header, flags, or reserved bytes")
        if provider_count > MAX_PROVIDER_REQUIREMENTS:
            raise FormatError("AIM compatibility provider count exceeds its fixed limit")
        expected_size = COMPATIBILITY_HEADER_STRUCT.size + provider_count * PROVIDER_REQUIREMENT_STRUCT.size
        if providers_offset != COMPATIBILITY_HEADER_STRUCT.size or section_size != len(data) or expected_size != len(data):
            raise FormatError("AIM compatibility provider table has invalid bounds")
        try:
            _validate_version_range(driver_min, driver_max, "CUDA Driver")
            _validate_version_range(runtime_min, runtime_max, "CUDA Runtime")
        except ValidationError as exc:
            raise FormatError(str(exc)) from exc
        if driver_min == 0:
            raise FormatError("AIM compatibility table has no CUDA Driver minimum")

        providers: list[ProviderRequirement] = []
        previous_id = 0
        for index in range(provider_count):
            provider_id, abi_min, abi_max, provider_flags, provider_reserved = PROVIDER_REQUIREMENT_STRUCT.unpack_from(
                data, providers_offset + index * PROVIDER_REQUIREMENT_STRUCT.size
            )
            if provider_id <= previous_id or provider_flags != 0 or any(provider_reserved):
                raise FormatError("AIM compatibility provider records are unsorted, duplicated, or have unknown flags")
            try:
                _validate_version_range(abi_min, abi_max, f"provider {provider_id} ABI")
            except ValidationError as exc:
                raise FormatError(str(exc)) from exc
            if abi_min == 0:
                raise FormatError(f"provider {provider_id} ABI minimum must be positive")
            providers.append(ProviderRequirement(provider_id, abi_min, abi_max))
            previous_id = provider_id
        return Compatibility(driver_min, driver_max, runtime_min, runtime_max, tuple(providers))


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
    compatibility: Compatibility
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
    """Deterministic experimental AIM v2 writer with payload deduplication."""

    @staticmethod
    def write(
        path: str | os.PathLike[str],
        *,
        platform: Platform,
        manifest: dict[str, Any],
        graph: dict[str, Any],
        tensors: dict[str, Any],
        compatibility: Compatibility,
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
        compatibility_bytes = compatibility.to_bytes()
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
        compatibility_section = add("compatibility", compatibility_bytes)
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
            compatibility_section.offset,
            compatibility_section.size,
            directory_offset,
            directory_size,
            directory_hash,
            b"\0" * 32,
            manifest_section.sha256,
            graph_section.sha256,
            compatibility_section.sha256,
            b"\0" * 40,
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

    @staticmethod
    def write_streaming(
        path: str | os.PathLike[str],
        *,
        platform: Platform,
        manifest: dict[str, Any],
        graph: dict[str, Any],
        tensors: dict[str, Any],
        compatibility: Compatibility,
        variants: Iterable[FileVariantPayload],
        runtime_abi: int = RUNTIME_ABI,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> AimInfo:
        """Write a deterministic AIM without loading file payloads into memory."""

        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValidationError("AIM streaming chunk_size must be positive")
        variant_list = sorted(list(variants), key=lambda item: int(item.arch))
        if not variant_list or not all(isinstance(item, FileVariantPayload) for item in variant_list):
            raise ValidationError("AIM streaming writer needs file-backed variants")
        validate_target(platform, [item.arch for item in variant_list])
        if runtime_abi <= 0:
            raise ValidationError("runtime ABI must be positive")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        resolved_destination = destination.resolve()
        prepared: list[
            tuple[FileVariantPayload, tuple[int, bytes], tuple[int, bytes], tuple[int, bytes]]
        ] = []
        for item in variant_list:
            paths = tuple(Path(value) for value in (item.kernels, item.weights, item.plan))
            if any(source.resolve() == resolved_destination for source in paths):
                raise ValidationError("AIM output cannot overwrite one of its input payloads")
            identities = tuple(_sha256_file(source, chunk_size=chunk_size) for source in paths)
            if any(size == 0 for size, _ in identities):
                raise ValidationError(f"{item.arch.name_string}: AIM payload is empty")
            try:
                with paths[0].open("rb") as stream:
                    kernel_prefix = stream.read(1024 * 1024)
            except OSError as exc:
                raise ValidationError(f"cannot read kernel bundle {paths[0]}: {exc}") from exc
            _reject_ptx(kernel_prefix, f"{item.arch.name_string} kernel bundle")
            _validate_cubin(kernel_prefix, item.arch, f"{item.arch.name_string} kernel bundle")
            from .executable import parse_executable_plan

            try:
                plan_bytes = paths[2].read_bytes()
            except OSError as exc:
                raise ValidationError(f"cannot read executable plan {paths[2]}: {exc}") from exc
            parse_executable_plan(
                plan_bytes,
                expected_arch=item.arch,
                expected_weights_bytes=identities[1][0],
            )
            prepared.append((item, identities[0], identities[1], identities[2]))

        metadata = (
            ("manifest", _canonical_json(manifest)),
            ("graph", _canonical_json(graph)),
            ("tensors", _canonical_json(tensors)),
            ("compatibility", compatibility.to_bytes()),
        )
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        sections: dict[tuple[str, bytes, int], Section] = {}

        def pad(stream: BinaryIO) -> None:
            padding = _align(stream.tell()) - stream.tell()
            if padding:
                stream.write(bytes(padding))

        def add_bytes(stream: BinaryIO, kind: str, payload: bytes) -> Section:
            digest = _sha256(payload)
            key = (kind, digest, len(payload))
            if key in sections:
                return sections[key]
            pad(stream)
            section = Section(stream.tell(), len(payload), digest)
            stream.write(payload)
            sections[key] = section
            return section

        def add_file(
            stream: BinaryIO,
            kind: str,
            source: Path,
            identity: tuple[int, bytes],
        ) -> Section:
            size, digest = identity
            key = (kind, digest, size)
            if key in sections:
                return sections[key]
            pad(stream)
            section = Section(stream.tell(), size, digest)
            try:
                with source.open("rb") as payload:
                    copied = 0
                    copied_hash = hashlib.sha256()
                    while True:
                        chunk = payload.read(chunk_size)
                        if not chunk:
                            break
                        stream.write(chunk)
                        copied_hash.update(chunk)
                        copied += len(chunk)
            except OSError as exc:
                raise ValidationError(f"cannot stream AIM payload {source}: {exc}") from exc
            if copied != size or copied_hash.digest() != digest:
                raise ValidationError(f"AIM payload changed while streaming: {source}")
            sections[key] = section
            return section

        try:
            with temporary.open("w+b") as output:
                output.write(bytes(HEADER_SIZE))
                manifest_section = add_bytes(output, *metadata[0])
                graph_section = add_bytes(output, *metadata[1])
                tensor_section = add_bytes(output, *metadata[2])
                compatibility_section = add_bytes(output, *metadata[3])
                variant_sections: list[tuple[FileVariantPayload, Section, Section, Section]] = []
                for item, kernel_identity, weight_identity, plan_identity in prepared:
                    variant_sections.append(
                        (
                            item,
                            add_file(output, "kernels", Path(item.kernels), kernel_identity),
                            add_file(output, "weights", Path(item.weights), weight_identity),
                            add_file(output, "plan", Path(item.plan), plan_identity),
                        )
                    )
                pad(output)
                directory_offset = output.tell()
                directory = bytearray()
                for item, kernels, weights, plan in variant_sections:
                    directory.extend(
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
                output.write(directory)
                file_size = output.tell()
                header = HEADER_STRUCT.pack(
                    MAGIC,
                    SCHEMA_MAJOR,
                    SCHEMA_MINOR,
                    HEADER_SIZE,
                    runtime_abi,
                    int(platform),
                    ENDIAN_TAG,
                    len(variant_sections),
                    file_size,
                    manifest_section.offset,
                    manifest_section.size,
                    graph_section.offset,
                    graph_section.size,
                    tensor_section.offset,
                    tensor_section.size,
                    compatibility_section.offset,
                    compatibility_section.size,
                    directory_offset,
                    len(directory),
                    _sha256(directory),
                    b"\0" * 32,
                    manifest_section.sha256,
                    graph_section.sha256,
                    compatibility_section.sha256,
                    b"\0" * 40,
                )
                output.seek(0)
                output.write(header)
                output.flush()
                os.fsync(output.fileno())
            file_digest = _zeroed_stream_hash(temporary, chunk_size=chunk_size)
            with temporary.open("r+b") as output:
                output.seek(FILE_HASH_OFFSET)
                output.write(file_digest)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return AimReader.read(
            destination, verify_payloads=True, verify_executable_plans=True
        )


class AimReader:
    @staticmethod
    def read(
        path: str | os.PathLike[str],
        *,
        verify_payloads: bool = True,
        verify_executable_plans: bool = False,
    ) -> AimInfo:
        source = Path(path)
        try:
            stream: BinaryIO
            with source.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size < HEADER_SIZE:
                    raise FormatError("file is smaller than the AIM header")
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    return AimReader._parse(
                        source, data, verify_payloads, verify_executable_plans
                    )
        except OSError as exc:
            raise FormatError(f"cannot read AIM {source}: {exc}") from exc

    @staticmethod
    def _parse(
        path: Path,
        data: mmap.mmap,
        verify_payloads: bool,
        verify_executable_plans: bool,
    ) -> AimInfo:
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
            compatibility_offset,
            compatibility_size,
            directory_offset,
            directory_size,
            directory_hash,
            file_hash,
            manifest_hash,
            graph_hash,
            compatibility_hash,
            reserved,
        ) = fields
        if magic != MAGIC:
            raise FormatError("bad AIM magic")
        if schema_major != SCHEMA_MAJOR or schema_minor > SCHEMA_MINOR:
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
        if AimReader._sha256_region(data, directory_offset, directory_size) != directory_hash:
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
        compatibility_data = AimReader._slice(
            data, compatibility_offset, compatibility_size, "compatibility table"
        )
        if _sha256(compatibility_data) != compatibility_hash:
            raise FormatError("compatibility table checksum mismatch")
        compatibility = Compatibility.from_bytes(compatibility_data)

        variants: list[VariantInfo] = []
        regions: list[tuple[int, int, str]] = [
            (manifest_offset, manifest_size, "manifest"),
            (graph_offset, graph_size, "graph"),
            (tensor_offset, tensor_size, "tensor table"),
            (compatibility_offset, compatibility_size, "compatibility table"),
            (directory_offset, directory_size, "variant directory"),
        ]
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
                AimReader._check_region(len(data), offset, size, f"{arch.name_string} {label}")
                if verify_payloads and AimReader._sha256_region(data, offset, size) != hashes[section_index]:
                    raise FormatError(f"{arch.name_string} {label} checksum mismatch")
                if label == "kernels":
                    prefix = memoryview(data)[offset : offset + min(size, 1024 * 1024)]
                    try:
                        _reject_ptx(prefix, f"{arch.name_string} kernel bundle")
                        _validate_cubin(prefix, arch, f"{arch.name_string} kernel bundle", FormatError)
                    finally:
                        prefix.release()
                sections_list.append(Section(offset, size, hashes[section_index]))
                regions.append((offset, size, f"{arch.name_string} {label}"))
            if verify_executable_plans:
                from .executable import parse_executable_plan

                plan_offset, plan_size = offsets[4:6]
                try:
                    parse_executable_plan(
                        AimReader._slice(data, plan_offset, plan_size, f"{arch.name_string} plan"),
                        expected_arch=arch,
                        expected_weights_bytes=offsets[3],
                    )
                except FormatError as exc:
                    raise FormatError(f"{arch.name_string} executable plan is invalid: {exc}") from exc
            variants.append(VariantInfo(arch, *sections_list))
        AimReader._check_non_overlapping_regions(regions)
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
            compatibility=compatibility,
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
    def _sha256_region(data: mmap.mmap, offset: int, size: int) -> bytes:
        view = memoryview(data)[offset : offset + size]
        try:
            return hashlib.sha256(view).digest()
        finally:
            view.release()

    @staticmethod
    def _check_non_overlapping_regions(regions: list[tuple[int, int, str]]) -> None:
        ordered = sorted(regions)
        previous: tuple[int, int, str] | None = None
        for current in ordered:
            if previous is not None:
                previous_offset, previous_size, previous_label = previous
                offset, size, label = current
                if offset == previous_offset and size == previous_size:
                    continue
                if offset < previous_offset + previous_size:
                    raise FormatError(
                        f"AIM sections overlap: {previous_label} and {label}"
                    )
            previous = current

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


def _validate_version_range(minimum: int, maximum: int, label: str) -> None:
    for value, suffix in ((minimum, "minimum"), (maximum, "maximum")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**32 - 1:
            raise ValidationError(f"{label} {suffix} must be a uint32")
    if minimum == 0 and maximum != 0:
        raise ValidationError(f"{label} maximum requires a non-zero minimum")
    if maximum != 0 and maximum < minimum:
        raise ValidationError(f"{label} maximum is lower than its minimum")
