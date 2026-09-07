from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aginfer.constant_store import ConstantCoverage, ConstantStore
from aginfer.errors import ValidationError
from aginfer.source_manifest import SourceManifest, build_source_manifest
from tests.helpers import write_safetensors


class ConstantStoreTests(unittest.TestCase):
    def _source(self, root: Path) -> SourceManifest:
        first = "model-00001-of-00002.safetensors"
        second = "model-00002-of-00002.safetensors"
        write_safetensors(root / first, {"layer.a": ("F16", [2], b"abcd")})
        write_safetensors(root / second, {"layer.b": ("F16", [1], b"ef")})
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer.a": first, "layer.b": second}}),
            encoding="utf-8",
        )
        write_safetensors(root / "policy_preprocessor_state.safetensors", {"shared": ("F16", [1], b"gh")})
        write_safetensors(root / "policy_postprocessor_state.safetensors", {"shared": ("F16", [1], b"ij")})
        return build_source_manifest(root)

    def test_namespace_keys_unify_sharded_and_single_file_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ConstantStore(self._source(root))
            self.assertEqual(store.namespaces, ("model", "postprocessor", "preprocessor"))
            self.assertEqual(
                [(key.namespace, key.name) for key in store.keys],
                [
                    ("model", "layer.a"),
                    ("model", "layer.b"),
                    ("postprocessor", "shared"),
                    ("preprocessor", "shared"),
                ],
            )
            destination = bytearray(3)
            self.assertEqual(store.readinto("model", "layer.a", destination, offset=1), 3)
            self.assertEqual(bytes(destination), b"bcd")
            self.assertEqual(b"".join(store.iter_chunks("model", "layer.b", chunk_size=1)), b"ef")
            with store.mmap_slice("preprocessor", "shared") as view:
                self.assertTrue(view.readonly)
                self.assertEqual(bytes(view), b"gh")

    def test_coverage_requires_every_constant_to_be_consumed_or_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConstantStore(self._source(Path(directory)))
            coverage = ConstantCoverage(store)
            coverage.consume("model", "layer.a", consumer="encoder.block.0.weight")
            coverage.consume("model", "layer.b", consumer="encoder.block.0.bias")
            coverage.ignore("preprocessor", "shared", reason="handled by native preprocessing asset")
            with self.assertRaisesRegex(ValidationError, r"unresolved source constants \(1\)"):
                coverage.require_complete()
            coverage.ignore("postprocessor", "shared", reason="handled by native postprocessing asset")
            coverage.require_complete()
            self.assertEqual(
                [record.disposition for record in coverage.records],
                ["consumed", "consumed", "ignored", "ignored"],
            )
            with self.assertRaisesRegex(ValidationError, "already resolved"):
                coverage.consume("model", "layer.a", consumer="duplicate")
            with self.assertRaisesRegex(ValidationError, "unknown constant"):
                coverage.ignore("model", "missing", reason="not present")

    def test_manifest_and_store_reject_changed_or_inconsistent_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._source(root)
            data = manifest.to_dict()
            data["tensors"][0]["namespace"] = "wrong"
            with self.assertRaisesRegex(ValidationError, "disagrees with asset"):
                SourceManifest.from_dict(data)

            data = manifest.to_dict()
            tensor = data["tensors"][0]
            asset = next(item for item in data["assets"] if item["asset_id"] == tensor["asset_id"])
            tensor["source_offset"] = asset["byte_size"]
            with self.assertRaisesRegex(ValidationError, "exceeds source asset"):
                SourceManifest.from_dict(data)

            first_asset = next(asset for asset in manifest.assets if asset.path.endswith("00001-of-00002.safetensors"))
            path = root / first_asset.path
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaisesRegex(ValidationError, "size changed"):
                ConstantStore(manifest)

    def test_manifest_deserialization_rejects_bool_numeric_and_untyped_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._source(Path(directory))
            data = manifest.to_dict()
            data["version"] = False
            with self.assertRaisesRegex(ValidationError, "version 0"):
                SourceManifest.from_dict(data)

            data = manifest.to_dict()
            data["assets"][0]["required"] = "yes"
            with self.assertRaisesRegex(ValidationError, "required flag"):
                SourceManifest.from_dict(data)

            data = manifest.to_dict()
            data["tensors"][0]["shape"] = [True]
            with self.assertRaisesRegex(ValidationError, "invalid tensor shape"):
                SourceManifest.from_dict(data)

            data = manifest.to_dict()
            data["tensors"][0]["dtype"] = ["F16"]
            with self.assertRaisesRegex(ValidationError, "unsupported source tensor dtype"):
                SourceManifest.from_dict(data)


if __name__ == "__main__":
    unittest.main()
