from __future__ import annotations

import pathlib
import sys

from aginfer.providers import SuffixMetadataPayload, SuffixMetadataProblem
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_suffix_metadata_payload_fixture.py OUTPUT")
    payload = SuffixMetadataPayload(
        SuffixMetadataProblem(CudaArch.SM120), 132_000, "c" * 64
    )
    pathlib.Path(sys.argv[1]).write_bytes(payload.to_bytes())


if __name__ == "__main__":
    main()
