"""Small synthetic fixtures pin fast load versus Debug/verification builds."""
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from aginfer.aim import AimReader, FILE_HASH_OFFSET, HEADER_STRUCT
from aginfer.compiler.verify import verify_artifact
from aginfer.errors import FormatError


def main():
    probe, fixture, mode = sys.argv[1:]
    assert mode in ("verify", "fast")
    original = Path(fixture).read_bytes()
    info = AimReader.read(fixture)
    header = HEADER_STRUCT.unpack_from(original)
    compatibility, directory = header[15], header[17]
    checks = 0

    def check(path, error=None):
        nonlocal checks
        result = subprocess.run([probe, str(path)], text=True, capture_output=True, check=True)
        lines = result.stdout.splitlines()
        if error is None:
            assert lines[0] == "OK", result.stdout
        else:
            assert lines[0] != "OK" and error in result.stdout, result.stdout
        checks += 1

    def rehash_file(data):
        data[FILE_HASH_OFFSET:FILE_HASH_OFFSET + 32] = bytes(32)
        data[FILE_HASH_OFFSET:FILE_HASH_OFFSET + 32] = hashlib.sha256(data).digest()

    with tempfile.TemporaryDirectory(prefix="aginfer-load-policy-") as temp:
        root = Path(temp)
        check(fixture)
        # Each stored digest is ignored only in fast mode. Repair outer digests
        # to ensure verification actually reaches the individual section gate.
        for name, offset, error, variant_digest in (
            ("file", 152, "file checksum", False),
            ("directory", 120, "metadata checksum", False),
            ("manifest", 184, "metadata checksum", False),
            ("graph", 216, "metadata checksum", False),
            ("compatibility", 248, "compatibility table checksum", False),
            ("kernel", directory + 56, "payload checksum", True),
            ("weight", directory + 88, "payload checksum", True),
            ("plan", directory + 120, "payload checksum", True),
        ):
            data = bytearray(original)
            data[offset] ^= 1
            if variant_digest:
                data[120:152] = hashlib.sha256(data[directory:directory + header[18]]).digest()
            if name != "file":
                rehash_file(data)
            path = root / f"digest-{name}.aim"
            path.write_bytes(data)
            check(path, error if mode == "verify" else None)

        # A structurally valid weight bit flip may load in fast mode, but must
        # always fail the offline verifier, independent of C++ build options.
        data = bytearray(original)
        data[info.variants[0].weights.offset] ^= 1
        path = root / "weight-bit-flip.aim"
        path.write_bytes(data)
        check(path, "file checksum" if mode == "verify" else None)
        try:
            verify_artifact(path)
        except FormatError as error:
            assert "file checksum" in str(error)
        else:
            raise AssertionError("offline verify accepted corrupt weights")

        # Repair hashes in structural negatives: these must fail for their
        # structural reason in BOTH modes, not merely because a hash differs.
        cases = [
            ("magic", 0, b"!", "bad AIM magic"),
            ("abi", 16, struct.pack("<I", 0), "Runtime ABI mismatch"),
            ("reserved", 280, b"\1", "non-zero AIM reserved"),
            ("section-bounds", 40, struct.pack("<Q", len(original) + 1), "section bounds"),
            ("weight-bounds", directory + 24, struct.pack("<Q", 2**64 - 256), "payload bounds"),
            ("compatibility-flags", compatibility + 36, struct.pack("<I", 1), "invalid AIM compatibility header"),
            ("cubin", info.variants[0].kernels.offset, b"!", "exact-architecture"),
        ]
        for name, offset, value, error in cases:
            data = bytearray(original)
            data[offset:offset + len(value)] = value
            if name == "cubin":
                region = info.variants[0].kernels
                data[directory + 56:directory + 88] = hashlib.sha256(data[region.offset:region.offset + region.size]).digest()
            data[120:152] = hashlib.sha256(data[directory:directory + header[18]]).digest()
            data[248:280] = hashlib.sha256(data[compatibility:compatibility + header[16]]).digest()
            rehash_file(data)
            path = root / f"bad-{name}.aim"
            path.write_bytes(data)
            check(path, error)
        path = root / "truncated.aim"
        path.write_bytes(original[:100])
        check(path, "smaller than its fixed header")
    print(f"{mode}: {checks} native load checks and offline corruption rejection passed")


if __name__ == "__main__":
    main()
