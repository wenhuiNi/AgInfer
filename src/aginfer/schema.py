from __future__ import annotations

from enum import Enum, IntEnum

SCHEMA_MAJOR = 2
SCHEMA_MINOR = 0
RUNTIME_ABI = 1
ALIGNMENT = 256


class Platform(IntEnum):
    LINUX_X86_64_GNU = 1
    LINUX_AARCH64_SBSA = 2

    @property
    def triple(self) -> str:
        return {
            Platform.LINUX_X86_64_GNU: "linux-x86_64-gnu",
            Platform.LINUX_AARCH64_SBSA: "linux-aarch64-sbsa",
        }[self]

    @classmethod
    def parse(cls, value: str) -> "Platform":
        for item in cls:
            if item.triple == value:
                return item
        raise ValueError(f"unsupported platform: {value}")


class CudaArch(IntEnum):
    SM89 = 89
    SM110 = 110
    SM120 = 120

    @property
    def name_string(self) -> str:
        return f"sm{int(self)}"

    @classmethod
    def parse(cls, value: str) -> "CudaArch":
        normalized = value.lower().replace("_", "")
        if normalized.startswith("sm"):
            normalized = normalized[2:]
        try:
            return cls(int(normalized))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"unsupported CUDA architecture: {value}") from exc


VALID_ARCHES = {
    Platform.LINUX_X86_64_GNU: frozenset({CudaArch.SM89, CudaArch.SM120}),
    Platform.LINUX_AARCH64_SBSA: frozenset({CudaArch.SM110}),
}


class DType(str, Enum):
    FP32 = "F32"
    FP16 = "F16"
    BF16 = "BF16"
    FP8_E4M3 = "F8_E4M3"


SUPPORTED_CHECKPOINT_DTYPES = frozenset(item.value for item in DType)
REJECTED_DTYPES = frozenset(
    {"I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64", "F64", "BOOL", "F8_E5M2"}
)


def validate_target(platform: Platform, arches: list[CudaArch]) -> None:
    if not arches:
        raise ValueError("at least one --cuda-arch is required")
    if len(set(arches)) != len(arches):
        raise ValueError("duplicate CUDA architecture")
    invalid = set(arches) - VALID_ARCHES[platform]
    if invalid:
        names = ", ".join(sorted(arch.name_string for arch in invalid))
        raise ValueError(f"{names} is not supported for {platform.triple}")
