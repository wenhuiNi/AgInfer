from __future__ import annotations

import pathlib
import sys

from aginfer.providers import PrefixKvStorePayload, PrefixKvStoreProblem
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_prefix_kv_store_payload_fixture.py OUTPUT")
    payload = PrefixKvStorePayload(
        PrefixKvStoreProblem(CudaArch.SM120), 128_000, "b" * 64
    )
    pathlib.Path(sys.argv[1]).write_bytes(payload.to_bytes())


if __name__ == "__main__":
    main()
