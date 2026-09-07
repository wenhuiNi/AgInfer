from __future__ import annotations

import sys
from pathlib import Path

from tests.executable_fixtures import executable_fixture


def main() -> None:
    Path(sys.argv[1]).write_bytes(executable_fixture().data)


if __name__ == "__main__":
    main()
