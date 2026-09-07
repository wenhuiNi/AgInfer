"""Optional small GPU contract: actual native evaluator, no model or dataset."""
import hashlib
import math
from pathlib import Path
import struct
import sys
import tempfile

from aginfer.compiler.constant_folding import fold_native_constants, validate_folding_report
from aginfer.executable import compile_executable_plan
from aginfer.packed_weights import pack_command_weights
from tests.test_constant_folding import fixture

helper, cubin = map(Path, sys.argv[1:])
module = cubin.read_bytes()
i, s, m, c, placements = fixture(module_bytes=len(module), module_sha256=hashlib.sha256(module).hexdigest())
with tempfile.TemporaryDirectory(prefix="aginfer-constant-eval-test-") as path:
    fm, fc, values, report = fold_native_constants(path, evaluator=helper, cubin_path=cubin,
        schedule=s, memory=m, commands=c, placements=placements, inventory=i, constants=None, literals=None)
    expected = 1 / (1 + math.exp(-1))
    expected = expected / (1 + math.exp(-expected))
    assert len(values) == 1 and len(fc.commands) == 1 and report["removed_commands"] == 2
    for payload in values.values():
        assert all(abs(x[0] - expected) < 2e-7 for x in struct.iter_unpack("<f", payload))
    weights = pack_command_weights(Path(path) / "final.bin", s, fm, i, fc, computed_constants=values)
    plan = compile_executable_plan(s, fm, fc, weights.spans, weights_bytes=weights.byte_size)
    validate_folding_report(report, plan)
print("constant-only native CUDA graph evaluation, repeat parity, finite output and packed boundary passed")
