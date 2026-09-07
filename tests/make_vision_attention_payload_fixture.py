from __future__ import annotations

import pathlib
import sys

from aginfer.providers import VisionAttentionPayload, VisionAttentionProblem
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_vision_attention_payload_fixture.py OUTPUT")
    payload = VisionAttentionPayload(
        VisionAttentionProblem(CudaArch.SM120), 48_608, "5" * 64
    ).to_bytes()
    pathlib.Path(sys.argv[1]).write_bytes(payload)


if __name__ == "__main__":
    main()
