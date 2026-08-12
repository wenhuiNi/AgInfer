from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aginfer.aim import AimReader, AimWriter, VariantPayload
from aginfer.errors import CompatibilityError, FormatError, ValidationError
from aginfer.schema import CudaArch, Platform
from tests.helpers import fake_cubin


class AimTests(unittest.TestCase):
    def _write(self, path: Path):
        manifest = {
            "runtime_abi": 1,
            "toolchain": {
                "cuda_driver_min": 12000,
                "cuda_runtime_min": 12000,
                "cuda_runtime_max": 12999,
                "cublaslt_abi": 12,
                "cudnn_abi": 9,
            },
        }
        variants = [
            VariantPayload(CudaArch.SM89, fake_cubin("sm89"), b"shared weights", b'{"arena_bytes":1,"workspace_bytes":2}'),
            VariantPayload(CudaArch.SM120, fake_cubin("sm120"), b"shared weights", b'{"arena_bytes":1,"workspace_bytes":2}'),
        ]
        return AimWriter.write(
            path, platform=Platform.LINUX_X86_64_GNU, manifest=manifest,
            graph={"opset": 1}, tensors={"count": 0}, variants=variants
        )

    def test_round_trip_multi_arch_and_deduplicates_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            info = self._write(Path(directory) / "model.aim")
            self.assertEqual([item.arch for item in info.variants], [CudaArch.SM89, CudaArch.SM120])
            self.assertEqual(info.variants[0].weights.offset, info.variants[1].weights.offset)
            self.assertEqual(info.select_variant(CudaArch.SM120).arch, CudaArch.SM120)
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
                    variants=[VariantPayload(CudaArch.SM89, b".version 8.0\n.target sm_89", b"w", b"p")]
                )
            with self.assertRaisesRegex(ValueError, "not supported"):
                AimWriter.write(
                    Path(directory) / "bad.aim", platform=Platform.LINUX_AARCH64_SBSA,
                    manifest={}, graph={}, tensors={},
                    variants=[VariantPayload(CudaArch.SM89, fake_cubin("sm89"), b"w", b"p")]
                )
            with self.assertRaisesRegex(ValidationError, "exact sm89"):
                AimWriter.write(
                    Path(directory) / "wrong-cubin.aim", platform=Platform.LINUX_X86_64_GNU,
                    manifest={}, graph={}, tensors={},
                    variants=[VariantPayload(CudaArch.SM89, fake_cubin("sm120"), b"w", b"p")]
                )


if __name__ == "__main__":
    unittest.main()
