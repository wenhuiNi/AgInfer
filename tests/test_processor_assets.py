from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from aginfer.errors import ValidationError
from aginfer.cli import main
from aginfer.processor_assets import inspect_source_asset_contract
from aginfer.source_package import open_source_package
from tests.helpers import write_safetensors


class ProcessorAssetTests(unittest.TestCase):
    def _source(self, root: Path, *, state_file: str = "policy_preprocessor_state.safetensors") -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "type": "synthetic_vla",
                    "input_features": {"image": {"type": "VISUAL", "shape": [3, 8, 8]}},
                    "output_features": {"action": {"type": "ACTION", "shape": [7]}},
                    "chunk_size": 4,
                }
            ),
            encoding="utf-8",
        )
        (root / "train_config.json").write_text('{"steps":1000,"batch_size":8}', encoding="utf-8")
        write_safetensors(root / "model.safetensors", {"weight": ("F16", [1], b"ab")})
        write_safetensors(root / "policy_preprocessor_state.safetensors", {"mean": ("F16", [1], b"cd")})
        (root / "policy_preprocessor.json").write_text(
            json.dumps(
                {
                    "name": "policy_preprocessor",
                    "steps": [
                        {
                            "registry_name": "normalizer_processor",
                            "config": {"eps": 1e-8},
                            "state_file": state_file,
                        },
                        {
                            "registry_name": "tokenizer_processor",
                            "config": {"tokenizer_name": "org/tokenizer", "max_length": 16},
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_feature_pipeline_state_and_external_requirement_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = open_source_package(str(self._source(Path(directory))), offline=True)
            contract = inspect_source_asset_contract(source)
            self.assertEqual(contract.source_type, "synthetic_vla")
            self.assertEqual(contract.input_features[0].shape, (3, 8, 8))
            self.assertEqual(contract.output_features[0].kind, "ACTION")
            self.assertEqual(len(contract.pipelines), 1)
            pipeline = contract.pipelines[0]
            self.assertEqual(pipeline.namespace, "preprocessor")
            self.assertEqual(
                pipeline.steps[0].state_asset_id,
                "preprocessor:policy_preprocessor_state.safetensors",
            )
            self.assertEqual(
                [(item.kind, item.locator) for item in contract.external_requirements],
                [("tokenizer", "org/tokenizer")],
            )
            self.assertEqual(contract.config_json, inspect_source_asset_contract(source).config_json)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["source-contract", str(Path(directory)), "--offline"]), 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["source_type"], "synthetic_vla")
            self.assertEqual(report["tensor_count"], 2)
            self.assertEqual(report["external_requirements"][0]["locator"], "org/tokenizer")

    def test_missing_or_wrong_namespace_processor_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._source(Path(directory), state_file="missing.safetensors")
            source = open_source_package(str(root), offline=True)
            with self.assertRaisesRegex(ValidationError, "not declared exactly once"):
                inspect_source_asset_contract(source)

        with tempfile.TemporaryDirectory() as directory:
            root = self._source(Path(directory), state_file="model.safetensors")
            source = open_source_package(str(root), offline=True)
            with self.assertRaisesRegex(ValidationError, "does not match pipeline namespace"):
                inspect_source_asset_contract(source)

    def test_malformed_feature_and_processor_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._source(Path(directory))
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["input_features"]["image"]["dynamic"] = True
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            source = open_source_package(str(root), offline=True)
            with self.assertRaisesRegex(ValidationError, "unsupported fields"):
                inspect_source_asset_contract(source)

        with tempfile.TemporaryDirectory() as directory:
            root = self._source(Path(directory))
            pipeline = json.loads((root / "policy_preprocessor.json").read_text(encoding="utf-8"))
            pipeline["steps"][0]["implementation"] = "python"
            (root / "policy_preprocessor.json").write_text(json.dumps(pipeline), encoding="utf-8")
            source = open_source_package(str(root), offline=True)
            with self.assertRaisesRegex(ValidationError, "unsupported fields"):
                inspect_source_asset_contract(source)


if __name__ == "__main__":
    unittest.main()
