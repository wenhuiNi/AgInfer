from __future__ import annotations

import pathlib
import sys

from aginfer.providers import FlashInferAttentionPayload, FlashInferAttentionProblem
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_flashinfer_attention_payload_fixture.py OUTPUT")
    payload = FlashInferAttentionPayload(
        FlashInferAttentionProblem(CudaArch.SM120)
    ).to_bytes()
    pathlib.Path(sys.argv[1]).write_bytes(payload)


if __name__ == "__main__":
    main()
