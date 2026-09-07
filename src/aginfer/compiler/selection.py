"""Offline, correctness-unvalidated cuBLASLt heuristic selection.

The native helper queries descriptors and AlgoCheck, never model data or timers.
The runtime continues to consume only fixed canonical payloads.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .identity import canonical, compiler_identity, digest, file_identity
from .native import FixedAlgorithms
from ..aim import _validate_cubin, _reject_ptx
from ..errors import ValidationError
from ..ir import dump_program
from ..lowering import build_lowering_inventory, dump_lowering_inventory
from ..providers.cublaslt import CublasLtAlgorithm, CublasLtDType, CublasLtLinearPayload, CublasLtLinearProblem
from ..providers.patch_projection import PatchProjectionProblem
from ..providers.rounded_attention import RoundedAttentionPayload
from ..recipes import Pi05SourceFrontend
from ..schema import CudaArch
from ..source_package import open_source_package

POLICY = "heuristic-first-reconstructable.v1"
WORKSPACE_LIMIT = 4 * 1024 * 1024


def validate_workspace_limit(value):
    if type(value) is not int or not 0 <= value <= 64 * 1024 * 1024:
        raise ValidationError("selection workspace limit must be 0..67108864 bytes")
    return value


def linear_request(problem, compute, workspace_limit=WORKSPACE_LIMIT):
    m, n, k = problem.m, problem.n, problem.k
    return {"dtype": int(problem.dtype), "compute": compute, "bias": 1, "batch": 1,
        "trans_a": 1, "trans_b": 0, "layouts": [[k, n, k, 0], [k, m, k, 0], [n, m, n, 0]],
        "workspace_limit": validate_workspace_limit(workspace_limit), "alignments": [256, 256, 256]}


def attention_requests(payload):
    q, k, heads, dim = payload.query_length, payload.key_length, payload.query_heads, payload.head_dim
    vision = payload.variant >= 3
    ld = heads * dim if vision else dim
    requests = []
    for qk in (True, False):
        layouts = [[dim, k, ld, dim if vision else 0],
            [dim if qk else k, q, ld if qk else k, (dim if vision else q * dim) if qk else q * k],
            [k if qk else dim, q, k if qk else heads * dim, q * k if qk else dim]]
        # cuBLAS preferences must also respect each strided batch's address.
        alignments = [math.gcd(16, x[3] * (4 if vision else 2)) for x in layouts]
        requests.append({"dtype": 1 if vision else 2, "compute": 1, "bias": 0,
            "batch": heads, "trans_a": int(qk), "trans_b": 0, "layouts": layouts,
            "workspace_limit": 0, "alignments": alignments})
    return requests


def encode_requests(requests):
    if not isinstance(requests, list) or not 1 <= len(requests) <= 256:
        raise ValidationError("invalid selection request count")
    lines = [f"AGINFER_LT_SELECT_V1 {len(requests)}"]
    for r in requests:
        numbers = [r[k] for k in ("dtype", "compute", "bias", "batch", "trans_a", "trans_b")]
        numbers += [x for layout in r["layouts"] for x in layout]
        numbers += [r["workspace_limit"], *r["alignments"]]
        if len(numbers) != 22 or any(type(x) is not int or x < 0 for x in numbers):
            raise ValidationError("invalid selection request fields")
        lines.append(" ".join(map(str, numbers)))
    return "\n".join(lines) + "\n"


def checked_results(probe, requests):
    fields = {"schema", "arch", "cublaslt_version", "cuda_driver_version", "results"}
    if (not isinstance(probe, dict) or set(probe) != fields
            or probe["schema"] != "aginfer.lt-selection-probe.v1"
            or type(probe["arch"]) is not int or probe["arch"] != 120
            or type(probe["cublaslt_version"]) is not int or probe["cublaslt_version"] != 120803
            or type(probe["cuda_driver_version"]) is not int or probe["cuda_driver_version"] <= 0
            or not isinstance(probe["results"], list) or len(probe["results"]) != len(requests)):
        raise ValidationError("selection helper returned incompatible target/version/count")
    result = []
    for r, item in zip(requests, probe["results"]):
        if (not isinstance(item, dict) or set(item) != {"algorithm", "workspace_bytes", "heuristic_rank", "candidate_count", "algo_check"}
                or item["algo_check"] is not True
                or not isinstance(item["algorithm"], list) or len(item["algorithm"]) != 9
                or any(type(x) is not int for x in item["algorithm"])
                or type(item["workspace_bytes"]) is not int or not 0 <= item["workspace_bytes"] <= r["workspace_limit"]
                or type(item["heuristic_rank"]) is not int or type(item["candidate_count"]) is not int
                or not 0 <= item["heuristic_rank"] < item["candidate_count"] <= 32):
            raise ValidationError("selection helper returned an invalid AlgoCheck result")
        result.append((CublasLtAlgorithm(*item["algorithm"]), item["workspace_bytes"]))
    return result


def selection_requests(algorithms, workspace_limit=WORKSPACE_LIMIT):
    result = [linear_request(x.problem, x.compute_mode, workspace_limit) for x in algorithms.linear]
    result.append(linear_request(algorithms.patch.problem, algorithms.patch.compute_mode, workspace_limit))
    for x in algorithms.attention:
        result.extend(attention_requests(x))
    return result


def validate_selection_report(report, selection, cubin, program_sha256=None):
    fields = {"schema", "policy", "compiler", "selector", "module", "program_sha256", "inventory_sha256",
        "requests", "probe", "fixed_algorithms", "selection_sha256", "numerical_validation", "capture_validation", "timing", "report_sha256"}
    if (not isinstance(report, dict) or set(report) != fields
            or report["schema"] != "aginfer.offline-selection.v1" or report["policy"] != POLICY
            or report["report_sha256"] != digest({k: v for k, v in report.items() if k != "report_sha256"})
            or report["fixed_algorithms"] != selection or report["selection_sha256"] != digest(selection)
            or report["module"] != {"bytes": len(cubin), "sha256": hashlib.sha256(cubin).hexdigest()}
            or report["numerical_validation"] != "not_run" or report["capture_validation"] != "not_run"
            or report["timing"] != "not_run"
            or (program_sha256 is not None and report["program_sha256"] != program_sha256)):
        raise ValidationError("selection report does not bind this candidate/module/program")
    compiler = report["compiler"]
    if (not isinstance(compiler, dict) or not isinstance(compiler.get("files"), dict)
            or not compiler["files"] or compiler.get("sha256") != digest(compiler["files"])):
        raise ValidationError("invalid selector compiler identity")
    selector = report["selector"]
    if (not isinstance(selector, dict) or set(selector) != {"bytes", "sha256"}
            or type(selector["bytes"]) is not int or selector["bytes"] <= 0
            or not isinstance(selector["sha256"], str) or len(selector["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in selector["sha256"])):
        raise ValidationError("invalid selection helper identity")
    for key in ("program_sha256", "inventory_sha256"):
        value = report[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValidationError("invalid selection graph identity")
    algorithms = FixedAlgorithms.from_dict(selection)
    for p in algorithms.linear + (algorithms.patch,):
        compute = 1 if p is algorithms.patch or p.problem.dtype == CublasLtDType.BF16 else 2
        if p.alignments != (256,) * 5 or p.compute_mode != compute:
            raise ValidationError("selection payload differs from precision/alignment policy")
    if not isinstance(report["requests"], list) or not report["requests"] or not isinstance(report["requests"][0], dict):
        raise ValidationError("invalid selection descriptors")
    limit = validate_workspace_limit(report["requests"][0].get("workspace_limit"))
    expected = selection_requests(algorithms, limit)
    if canonical(report["requests"]) != canonical(expected):
        raise ValidationError("selection descriptors differ from fixed payloads")
    results = checked_results(report["probe"], expected)
    index = 0
    for p in algorithms.linear + (algorithms.patch,):
        if (p.algorithm, p.workspace_bytes) != results[index]:
            raise ValidationError("linear payload differs from AlgoCheck selection")
        index += 1
    for p in algorithms.attention:
        for config in (p.qk_algorithm, p.pv_algorithm):
            if (CublasLtAlgorithm(*config), 0) != results[index]:
                raise ValidationError("attention payload differs from AlgoCheck selection")
            index += 1
    return report


def select_source_algorithms(source_path, *, cubin_path, selector_path, output, report_path, workspace_limit=WORKSPACE_LIMIT,
                             fuse_projections=False, resident_kv=False, fuse_ffn=False):
    validate_workspace_limit(workspace_limit)
    output, report_path = Path(output), Path(report_path)
    if output.resolve() == report_path.resolve() or any(x.exists() or not x.parent.is_dir() for x in (output, report_path)):
        raise ValidationError("selection outputs must be distinct new files in existing directories")
    cubin_path = Path(cubin_path)
    if cubin_path.stat().st_size > 256 * 1024 * 1024:
        raise ValidationError("selection CUBIN exceeds size limit")
    cubin = cubin_path.read_bytes()
    _validate_cubin(cubin, CudaArch.SM120, "selection input")
    _reject_ptx(cubin, "selection input")
    source = open_source_package(str(source_path), offline=True)
    program = Pi05SourceFrontend().import_program(source).program
    if fuse_projections or fuse_ffn:
        from .projection_fusion import fuse_projections as transform
        program = transform(program, qkv=fuse_projections, ffn=fuse_ffn).program
    if resident_kv:
        from .resident_kv import make_kv_resident
        program = make_kv_resident(program).program
    inventory = build_lowering_inventory(program)
    problems = sorted({CublasLtLinearProblem.from_inventory(x, target_arch=CudaArch.SM120)
        for x in inventory.ops if x.opcode == "linear"}, key=lambda p: (int(p.dtype), p.m, p.n, p.k))
    patch_problems = set()
    for x in inventory.ops:
        if x.opcode == "conv2d":
            PatchProjectionProblem.from_inventory(x, target_arch=CudaArch.SM120)
            weight = x.input_types[1].shape
            patch_problems.add(CublasLtLinearProblem(CudaArch.SM120, CublasLtDType.F32,
                math.prod(x.output_types[0].shape[2:]), weight[0], math.prod(weight[1:])))
    if not problems or len(patch_problems) != 1:
        raise ValidationError("source does not have a supported linear/patch selection inventory")
    patch_problem = next(iter(patch_problems))
    module_sha = hashlib.sha256(cubin).hexdigest()
    attention = [RoundedAttentionPayload(CudaArch.SM120, variant, len(cubin), module_sha,
        120803, (0,) * 9, (0,) * 9) for variant in (1, 2, 4)]
    computes = [2 if x.dtype == CublasLtDType.F32 else 1 for x in problems] + [1]
    all_linear = problems + [patch_problem]
    requests = [linear_request(p, compute, workspace_limit) for p, compute in zip(all_linear, computes)]
    for p in attention:
        requests.extend(attention_requests(p))
    selector_path = Path(selector_path).resolve()
    selector_identity = file_identity(selector_path)
    try:
        run = subprocess.run([str(selector_path)], input=encode_requests(requests),
            capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValidationError(f"offline selection helper could not run: {exc}") from exc
    if run.returncode != 0:
        raise ValidationError(f"offline selection helper refused: {run.stderr[:2048].strip()}")
    if len(run.stdout) > 1024 * 1024 or file_identity(selector_path) != selector_identity:
        raise ValidationError("selection helper output exceeds limit or binary changed")
    try:
        probe = json.loads(run.stdout)
    except ValueError as exc:
        raise ValidationError("selection helper returned invalid JSON") from exc
    results = checked_results(probe, requests)
    linear = [CublasLtLinearPayload(p, 120803, result[0], result[1], 256, 256, 256, 256, 256, compute)
        for p, compute, result in zip(all_linear, computes, results)]
    offset = len(all_linear)
    def config(index):
        return tuple(probe["results"][index]["algorithm"])
    attention = [RoundedAttentionPayload(CudaArch.SM120, p.variant, len(cubin), module_sha,
        120803, config(offset + i * 2), config(offset + i * 2 + 1)) for i, p in enumerate(attention)]
    selection = {"schema": "aginfer.fixed-algorithms.v1", "linear": [p.to_bytes().hex() for p in linear[:-1]],
        "patch": linear[-1].to_bytes().hex(), "attention": [p.to_bytes().hex() for p in attention]}
    report = {"schema": "aginfer.offline-selection.v1", "policy": POLICY, "compiler": compiler_identity(),
        "selector": selector_identity, "module": {"bytes": len(cubin), "sha256": module_sha},
        "program_sha256": hashlib.sha256(dump_program(program).encode()).hexdigest(),
        "inventory_sha256": hashlib.sha256(dump_lowering_inventory(inventory).encode()).hexdigest(),
        "requests": requests, "probe": probe, "fixed_algorithms": selection, "selection_sha256": digest(selection),
        "numerical_validation": "not_run", "capture_validation": "not_run", "timing": "not_run"}
    report["report_sha256"] = digest(report)
    validate_selection_report(report, selection, cubin, report["program_sha256"])
    # Exclusive creation: never truncate an existing user's selection or receipt.
    with output.open("xb") as f:
        f.write(canonical(selection))
    with report_path.open("xb") as f:
        f.write(canonical(report))
    return {"status": "algorithms_selected_not_numerically_validated", "problems": len(requests),
        "selection": file_identity(output), "report_sha256": report["report_sha256"], "policy": POLICY}
