from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from aginfer.checkpoint import load_checkpoint, pack_weights
from aginfer.errors import ValidationError
from tests.helpers import create_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_reads_and_packs_safetensors_without_casting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = load_checkpoint(str(create_checkpoint(Path(directory) / "checkpoint")), offline=True)
            self.assertEqual(checkpoint.dtypes, {"F16"})
            blob, table = pack_weights(checkpoint.tensors)
            self.assertEqual(blob, bytes(range(8)))
            self.assertEqual(table[0]["dtype"], "F16")
            self.assertEqual(table[0]["stride"], [2, 1])

    def test_rejects_pickle_even_when_safetensors_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_checkpoint(Path(directory) / "checkpoint")
            (root / "pytorch_model.bin").write_bytes(b"pickle")
            with self.assertRaisesRegex(ValidationError, "pickle"):
                load_checkpoint(str(root))

    def test_rejects_unknown_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"groot"}')
            header = json.dumps({"bad": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}}).encode()
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"x")
            with self.assertRaisesRegex(ValidationError, "unsupported dtype"):
                load_checkpoint(str(root))


if __name__ == "__main__":
    unittest.main()

