from __future__ import annotations

import sys
from pathlib import Path

from aginfer.providers import TimeEmbeddingPayload, TimeEmbeddingProblem
from aginfer.schema import CudaArch


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    payload = TimeEmbeddingPayload(
        TimeEmbeddingProblem(CudaArch.SM120), 153_000, "b" * 64
    )
    Path(sys.argv[1]).write_bytes(payload.to_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
