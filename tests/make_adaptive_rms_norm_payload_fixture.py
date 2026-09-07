from __future__ import annotations

import pathlib
import sys

from aginfer.providers import AdaptiveRmsNormPayload, AdaptiveRmsNormProblem
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_adaptive_rms_norm_payload_fixture.py OUTPUT")
    payload = AdaptiveRmsNormPayload(
        AdaptiveRmsNormProblem(CudaArch.SM120), 96_000, "9" * 64
    )
    pathlib.Path(sys.argv[1]).write_bytes(payload.to_bytes())


if __name__ == "__main__":
    main()
