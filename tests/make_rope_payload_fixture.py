from __future__ import annotations

import pathlib
import sys

from aginfer.providers import RopePayload, RopeProblem, RopeVariant
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_rope_payload_fixture.py OUTPUT")
    payloads = tuple(
        RopePayload(
            RopeProblem(CudaArch.SM120, variant), 80_000, "8" * 64
        ).to_bytes()
        for variant in RopeVariant
    )
    pathlib.Path(sys.argv[1]).write_bytes(b"".join(payloads))


if __name__ == "__main__":
    main()
