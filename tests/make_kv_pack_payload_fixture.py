from __future__ import annotations

import pathlib
import sys

from aginfer.providers import KvPackPayload, KvPackProblem
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_kv_pack_payload_fixture.py OUTPUT")
    payload = KvPackPayload(KvPackProblem(CudaArch.SM120), 128_000, "a" * 64)
    pathlib.Path(sys.argv[1]).write_bytes(payload.to_bytes())


if __name__ == "__main__":
    main()
