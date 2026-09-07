from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from aginfer.lowering import (
    CommandOperand,
    CommandStream,
    CommandTag,
    OperandAccess,
    ProviderCommand,
)
from aginfer.schema import CudaArch


def digest(label: bytes) -> str:
    return hashlib.sha256(label).hexdigest()


def main() -> None:
    output = Path(sys.argv[1])
    stream = CommandStream(
        target_arch=CudaArch.SM120,
        memory_plan_sha256=digest(b"memory-plan"),
        value_count=4,
        arena_bytes=4096,
        state_bytes=256,
        workspace_bytes=1024,
        commands=(
            ProviderCommand(
                tag=CommandTag.MEMORY_COPY,
                provider_id=7,
                abi_major=1,
                abi_minor=2,
                capability_digest=digest(b"capability-7"),
                operands=(
                    CommandOperand(0, OperandAccess.READ, 4),
                    CommandOperand(1, OperandAccess.WRITE),
                ),
                payload=b"copy-v1",
                capture_safe=True,
            ),
            ProviderCommand(
                tag=CommandTag.CUBLASLT_MATMUL,
                provider_id=8,
                abi_major=1,
                abi_minor=2,
                capability_digest=digest(b"capability-8"),
                operands=(
                    CommandOperand(1, OperandAccess.READ),
                    CommandOperand(2, OperandAccess.READ, 128),
                    CommandOperand(3, OperandAccess.WRITE),
                ),
                payload=b"fixed-cublaslt-descriptor",
                workspace_offset=256,
                workspace_bytes=512,
            ),
        ),
    )
    output.write_bytes(stream.to_bytes())


if __name__ == "__main__":
    main()
