from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aginfer.compiler import CompileOptions, compile_model
from aginfer.aim import AimReader
from aginfer.errors import ValidationError
from aginfer.plan import inspect_plan
from aginfer.schema import CudaArch, Platform
from tests.helpers import create_artifacts, create_checkpoint, create_profile


class CompilerTests(unittest.TestCase):
    def test_end_to_end_metadata_compile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "model.aim"
            info = compile_model(
                CompileOptions(
                    source=str(create_checkpoint(root / "checkpoint")), revision=None,
                    adapter="groot_n1_5", platform=Platform.LINUX_X86_64_GNU,
                    cuda_arches=(CudaArch.SM89, CudaArch.SM120), profile_path=create_profile(root / "profile.json"),
                    artifact_dir=create_artifacts(root / "artifacts", ("sm89", "sm120")), output=output, offline=True,
                )
            )
            self.assertEqual(info.manifest["checkpoint_dtypes"], ["F16"])
            self.assertFalse(info.manifest["deployment_features"]["ptx"])
            self.assertEqual(AimReader.read(output).tensors["count"], 1)
            variant = info.variants[0]
            with output.open("rb") as stream:
                stream.seek(variant.plan.offset)
                plan = stream.read(variant.plan.size)
            self.assertEqual(inspect_plan(plan)["launch_count"], 1)

    def test_missing_verified_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = create_artifacts(root / "artifacts")
            (artifacts / "sm89" / "tactics.json").unlink()
            with self.assertRaisesRegex(ValidationError, "tactic database"):
                compile_model(
                    CompileOptions(
                        source=str(create_checkpoint(root / "checkpoint")), revision=None,
                        adapter="groot_n1_5", platform=Platform.LINUX_X86_64_GNU,
                        cuda_arches=(CudaArch.SM89,), profile_path=create_profile(root / "profile.json"),
                        artifact_dir=artifacts, output=root / "model.aim", offline=True,
                    )
                )

if __name__ == "__main__":
    unittest.main()
