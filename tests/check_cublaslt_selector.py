"""CPU-only native selection protocol contract. No model fixture or GPU call."""
import json
import subprocess
import sys
from test_algorithm_selection import fixture
from aginfer.compiler.selection import WORKSPACE_LIMIT, encode_requests

_, report, _ = fixture()
valid = encode_requests(report["requests"])
binary = sys.argv[1]
result = subprocess.run([binary, "--check-input"], input=valid, text=True, capture_output=True)
assert result.returncode == 0, result.stderr
assert json.loads(result.stdout) == {"request_count": 8, "input_valid": True}
for bad in ("", "UNKNOWN 1\n", "AGINFER_LT_SELECT_V1 257\n", valid + "extra", valid[:-8],
            valid.replace("2 3 2 0", "1 3 2 0", 1), valid.replace(str(WORKSPACE_LIMIT), "67108865", 1),
            valid.replace("256 256 256", "3 256 256", 1), "x" * 65537):
    result = subprocess.run([binary, "--check-input"], input=bad, text=True, capture_output=True)
    assert result.returncode == 2 and not result.stdout, (bad[:80], result.stdout, result.stderr)
