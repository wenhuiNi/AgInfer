from __future__ import annotations

import pathlib
import sys

from aginfer.providers import PrefixInputPayload, PrefixInputProblem
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_prefix_input_payload_fixture.py OUTPUT")
    payload = PrefixInputPayload(
        PrefixInputProblem(CudaArch.SM120), 150_000, "d" * 64
    )
    pathlib.Path(sys.argv[1]).write_bytes(payload.to_bytes())


if __name__ == "__main__":
    main()
