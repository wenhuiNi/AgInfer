from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aginfer.constant_store import ConstantCoverage
from aginfer.errors import ValidationError
from aginfer.frontend import FrontendOutput
from aginfer.ir import DType, Function, Op, Program, Region, TensorType, Value, attributes
from aginfer.source_manifest import AssetRole
from aginfer.source_package import open_source_package
from tests.helpers import write_safetensors
from tests.ir_fixtures import gated_mlp_program


class SourcePackageTests(unittest.TestCase):
    def _source(self, root: Path, *, weight_name: str = "weight") -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text('{"type":"synthetic","width":2}', encoding="utf-8")
        write_safetensors(root / "model.safetensors", {weight_name: ("F16", [2], b"abcd")})
        return root

    def test_open_source_package_exposes_bounded_assets_and_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._source(Path(directory) / "checkpoint")
            source = open_source_package(str(root), offline=True)
            metadata = source.assets.select(role=AssetRole.METADATA.value, namespace="model")
            self.assertEqual([asset.path for asset in metadata], ["config.json"])
            self.assertEqual(source.assets.read_json("model:config.json"), {"type": "synthetic", "width": 2})
            with self.assertRaisesRegex(ValidationError, "bounded read limit"):
                source.assets.read_bytes("model:model.safetensors", max_bytes=1)
            self.assertEqual(source.constants.tensor("model", "weight").shape, [2])

    def test_source_package_rejects_pickle_before_frontend_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._source(Path(directory) / "checkpoint")
            (root / "optimizer.pt").write_bytes(b"not executed")
            with self.assertRaisesRegex(ValidationError, "pickle"):
                open_source_package(str(root), offline=True)

    def test_frontend_output_requires_verified_ir_and_exact_source_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = open_source_package(str(self._source(root / "first")), offline=True)
            coverage = ConstantCoverage(source.constants)
            with self.assertRaisesRegex(ValidationError, "unresolved source constants"):
                FrontendOutput.finalize(gated_mlp_program(), source, coverage)
            coverage.consume("model", "weight", consumer="gated_mlp.gate_weight")
            output = FrontendOutput.finalize(gated_mlp_program(), source, coverage)
            self.assertIs(output.constants, source.constants)
            self.assertEqual(output.coverage[0].detail, "gated_mlp.gate_weight")

            other = open_source_package(str(self._source(root / "second")), offline=True)
            wrong_coverage = ConstantCoverage(other.constants)
            wrong_coverage.consume("model", "weight", consumer="wrong source")
            with self.assertRaisesRegex(ValidationError, "does not belong"):
                FrontendOutput.finalize(gated_mlp_program(), source, wrong_coverage)

    def test_frontend_constant_refs_match_store_type_and_consumed_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = open_source_package(str(self._source(Path(directory))), offline=True)

            def program(shape: tuple[int, ...]) -> Program:
                tensor = TensorType(DType.F16, shape)
                return Program(
                    entry="main",
                    functions=(
                        Function(
                            "main",
                            (Value("trigger", TensorType(DType.F32, (1,))),),
                            ("weight",),
                            Region(
                                (
                                    Op(
                                        "constant_ref",
                                        (),
                                        (Value("weight", tensor),),
                                        attributes(namespace="model", name="weight"),
                                    ),
                                )
                            ),
                        ),
                    ),
                )

            coverage = ConstantCoverage(source.constants)
            coverage.consume("model", "weight", consumer="main.weight")
            FrontendOutput.finalize(program((2,)), source, coverage)
            with self.assertRaisesRegex(ValidationError, "type disagrees"):
                FrontendOutput.finalize(program((1, 2)), source, coverage)

            ignored = ConstantCoverage(source.constants)
            ignored.ignore("model", "weight", reason="not needed")
            with self.assertRaisesRegex(ValidationError, "references an ignored"):
                FrontendOutput.finalize(program((2,)), source, ignored)


if __name__ == "__main__":
    unittest.main()
