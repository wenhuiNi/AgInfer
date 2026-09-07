from __future__ import annotations

import tempfile
import unittest
import struct
from pathlib import Path

from aginfer.aim import (
    COMPATIBILITY_HEADER_STRUCT,
    Compatibility,
    AimReader,
    AimWriter,
    FileVariantPayload,
    ProviderRequirement,
    VariantPayload,
)
from aginfer.errors import CompatibilityError, FormatError, ValidationError
from aginfer.schema import CudaArch, Platform
from tests.helpers import fake_cubin, runtime_compatibility
from tests.executable_fixtures import executable_fixture


class AimTests(unittest.TestCase):
    def _write(self, path: Path):
        manifest = {"model_family": "synthetic"}
        variants = [
            VariantPayload(CudaArch.SM89, fake_cubin("sm89"), b"shared weights", b'{"arena_bytes":1,"workspace_bytes":2}'),
            VariantPayload(CudaArch.SM120, fake_cubin("sm120"), b"shared weights", b'{"arena_bytes":1,"workspace_bytes":2}'),
        ]
        return AimWriter.write(
            path, platform=Platform.LINUX_X86_64_GNU, manifest=manifest,
            graph={"opset": 1}, tensors={"count": 0},
            compatibility=runtime_compatibility(), variants=variants
        )

    def test_round_trip_multi_arch_and_deduplicates_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            info = self._write(Path(directory) / "model.aim")
            self.assertEqual([item.arch for item in info.variants], [CudaArch.SM89, CudaArch.SM120])
            self.assertEqual(info.variants[0].weights.offset, info.variants[1].weights.offset)
            self.assertEqual(info.select_variant(CudaArch.SM120).arch, CudaArch.SM120)
            self.assertEqual(info.compatibility.providers[0].provider_id, 1)
            with self.assertRaises(CompatibilityError):
                info.select_variant(CudaArch.SM110)

    def test_detects_single_byte_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.aim"
            info = self._write(path)
            data = bytearray(path.read_bytes())
            data[info.variants[0].kernels.offset] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(FormatError, "file checksum"):
                AimReader.read(path)

    def test_rejects_ptx_and_invalid_platform_arch_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValidationError, "PTX"):
                AimWriter.write(
                    Path(directory) / "ptx.aim", platform=Platform.LINUX_X86_64_GNU,
                    manifest={}, graph={}, tensors={},
                    compatibility=runtime_compatibility(),
                    variants=[VariantPayload(CudaArch.SM89, b".version 8.0\n.target sm_89", b"w", b"p")]
                )
            with self.assertRaisesRegex(ValueError, "not supported"):
                AimWriter.write(
                    Path(directory) / "bad.aim", platform=Platform.LINUX_AARCH64_SBSA,
                    manifest={}, graph={}, tensors={},
                    compatibility=runtime_compatibility(),
                    variants=[VariantPayload(CudaArch.SM89, fake_cubin("sm89"), b"w", b"p")]
                )
            with self.assertRaisesRegex(ValidationError, "exact sm89"):
                AimWriter.write(
                    Path(directory) / "wrong-cubin.aim", platform=Platform.LINUX_X86_64_GNU,
                    manifest={}, graph={}, tensors={},
                    compatibility=runtime_compatibility(),
                    variants=[VariantPayload(CudaArch.SM89, fake_cubin("sm120"), b"w", b"p")]
                )

    def test_compatibility_table_rejects_unknown_flags_and_bad_ranges(self) -> None:
        compatibility = Compatibility(
            cuda_driver_min=12000,
            providers=(ProviderRequirement(7, 3, 4),),
        )
        data = bytearray(compatibility.to_bytes())
        struct.pack_into("<I", data, COMPATIBILITY_HEADER_STRUCT.size + 12, 1)
        with self.assertRaisesRegex(FormatError, "unknown flags"):
            Compatibility.from_bytes(data)
        with self.assertRaisesRegex(FormatError, "invalid bounds"):
            Compatibility.from_bytes(compatibility.to_bytes()[:-1])
        with self.assertRaisesRegex(ValidationError, "lower"):
            Compatibility(cuda_driver_min=13000, cuda_driver_max=12000).to_bytes()

    def test_streaming_writer_accepts_only_checked_executable_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kernels = root / "kernels.cubin"
            weights = root / "weights.bin"
            plan = root / "plan.bin"
            kernels.write_bytes(fake_cubin("sm120"))
            weights.write_bytes(b"w" + bytes(255))
            plan.write_bytes(executable_fixture().data)
            first = root / "first.aim"
            second = root / "second.aim"
            kwargs = {
                "platform": Platform.LINUX_X86_64_GNU,
                "manifest": {"model_family": "synthetic"},
                "graph": {"opset": 1},
                "tensors": {"count": 5},
                "compatibility": runtime_compatibility(),
                "variants": (
                    FileVariantPayload(CudaArch.SM120, kernels, weights, plan),
                ),
                "chunk_size": 17,
            }
            info = AimWriter.write_streaming(first, **kwargs)
            AimWriter.write_streaming(second, **kwargs)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(info.file_size, first.stat().st_size)
            AimReader.read(first, verify_executable_plans=True)

            corrupted_plan = bytearray(plan.read_bytes())
            corrupted_plan[-1] ^= 1
            plan.write_bytes(corrupted_plan)
            with self.assertRaisesRegex(FormatError, "body checksum"):
                AimWriter.write_streaming(root / "bad.aim", **kwargs)


if __name__ == "__main__":
    unittest.main()
