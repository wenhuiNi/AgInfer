from __future__ import annotations

import sys
from pathlib import Path

from aginfer.providers import ActionSlicePayload, ActionSliceProblem
from aginfer.schema import CudaArch


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    payload = ActionSlicePayload(
        ActionSliceProblem(CudaArch.SM120), 181_000, "c" * 64
    )
    Path(sys.argv[1]).write_bytes(payload.to_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
