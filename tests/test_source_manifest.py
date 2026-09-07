from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from aginfer.cli import main
from aginfer.errors import ValidationError
from aginfer.source_manifest import SourceManifest, build_source_manifest
from tests.helpers import write_safetensors


class SourceManifestTests(unittest.TestCase):
    def test_namespaces_allow_same_tensor_name_for_separate_state_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"pi05"}', encoding="utf-8")
            write_safetensors(root / "model.safetensors", {"shared": ("F16", [1], b"ab")})
            write_safetensors(
                root / "policy_preprocessor_step_0_unnormalize.safetensors",
                {"shared": ("F16", [1], b"cd")},
            )
            write_safetensors(
                root / "policy_postprocessor_step_0_unnormalize.safetensors",
                {"shared": ("F16", [1], b"ef")},
            )

            manifest = build_source_manifest(root)
            self.assertEqual({tensor.namespace for tensor in manifest.tensors}, {"model", "preprocessor", "postprocessor"})
            self.assertEqual([tensor.name for tensor in manifest.tensors], ["shared", "shared", "shared"])
            self.assertIsNone(manifest.revision)

            output = root / "source-manifest.json"
            manifest.write(output)
            self.assertEqual(SourceManifest.read(output).to_dict(), manifest.to_dict())

    def test_validates_sharded_weight_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = "model-00001-of-00002.safetensors"
            second = "model-00002-of-00002.safetensors"
            write_safetensors(root / first, {"a": ("F16", [1], b"ab")})
            write_safetensors(root / second, {"b": ("F16", [1], b"cd")})
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": first, "b": second}}),
                encoding="utf-8",
            )

            manifest = build_source_manifest(root, revision="dev")
            self.assertEqual(manifest.revision, "dev")
            self.assertEqual({tensor.shard_set for tensor in manifest.tensors}, {"model.safetensors.index.json"})

    def test_cli_writes_manifest_and_reports_namespace_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            write_safetensors(root / "model.safetensors", {"a": ("F16", [1], b"ab")})
            output = Path(directory) / "manifest.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main(["source-manifest", str(root), "--offline", "--output", str(output)])
            self.assertEqual(result, 0)
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(stdout.getvalue())["tensor_namespaces"], {"model": 1})

    def test_rejects_unsafe_and_inaccurate_weight_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_safetensors(root / "unrelated.safetensors", {"x": ("F16", [1], b"ab")})
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "../outside.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "unsafe shard path"):
                build_source_manifest(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = "model-00001-of-00001.safetensors"
            write_safetensors(root / shard, {"actual": ("F16", [1], b"ab")})
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"declared": shard}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "invalid weight_map"):
                build_source_manifest(root)


if __name__ == "__main__":
    unittest.main()
