from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aginfer.errors import ValidationError
from aginfer.safetensors import SafetensorsReader
from tests.helpers import write_safetensors


class SafetensorsReaderTests(unittest.TestCase):
    def test_bounded_readinto_chunks_and_mmap_slice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            payload = bytes(range(17))
            write_safetensors(path, {"weight": ("F16", [17], payload + payload)})

            reader = SafetensorsReader(path)
            self.assertEqual(reader.tensor_names, ("weight",))
            with reader:
                destination = bytearray(7)
                self.assertEqual(reader.readinto("weight", destination, offset=5), 7)
                self.assertEqual(bytes(destination), (payload + payload)[5:12])
                self.assertEqual(
                    b"".join(reader.iter_chunks("weight", chunk_size=5)),
                    payload + payload,
                )
                with reader.mmap_slice("weight", offset=3, length=4) as view:
                    self.assertTrue(view.readonly)
                    self.assertEqual(bytes(view), (payload + payload)[3:7])

            with self.assertRaisesRegex(ValidationError, "open reader context"):
                reader.readinto("weight", bytearray(1))

    def test_rejects_invalid_ranges_and_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            write_safetensors(path, {"weight": ("F16", [2], b"abcd")})
            with SafetensorsReader(path) as reader:
                with self.assertRaisesRegex(ValidationError, "out of range"):
                    reader.readinto("weight", bytearray(1), offset=5)
                with self.assertRaisesRegex(ValidationError, "writable"):
                    reader.readinto("weight", b"x")
                with self.assertRaisesRegex(ValidationError, "positive"):
                    list(reader.iter_chunks("weight", chunk_size=0))
                with self.assertRaisesRegex(ValidationError, "not found"):
                    reader.tensor("missing")


if __name__ == "__main__":
    unittest.main()
