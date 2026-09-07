from __future__ import annotations

import pathlib
import sys

from aginfer.providers import RmsNormPayload, RmsNormProblem, RmsNormVariant
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_rms_norm_payload_fixture.py OUTPUT")
    payloads = tuple(
        RmsNormPayload(
            RmsNormProblem(CudaArch.SM120, variant), 64_000, "7" * 64
        ).to_bytes()
        for variant in RmsNormVariant
    )
    pathlib.Path(sys.argv[1]).write_bytes(b"".join(payloads))


if __name__ == "__main__":
    main()
