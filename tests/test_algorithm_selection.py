from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from aginfer.compiler.identity import digest
from aginfer.compiler.native import FixedAlgorithms
from aginfer.compiler.selection import (POLICY, BENCHMARK_POLICY, PREFILL_BENCHMARK_POLICY, WORKSPACE_LIMIT, attention_requests, checked_results,
    benchmark_eligible,
    encode_requests, linear_request, select_source_algorithms, selection_requests, validate_selection_report)
from aginfer.errors import ValidationError
from aginfer.providers.cublaslt import CublasLtAlgorithm, CublasLtDType, CublasLtLinearPayload, CublasLtLinearProblem
from aginfer.providers.rounded_attention import RoundedAttentionPayload
from aginfer.schema import CudaArch
import hashlib


def fixture():
    cubin = b"fixture module"
    module = {"bytes": len(cubin), "sha256": hashlib.sha256(cubin).hexdigest()}
    algorithm = CublasLtAlgorithm(10, 11, 0, 1, 0, 0, 0, 0, 0)
    def linear(m, n, k, compute):
        return CublasLtLinearPayload(CublasLtLinearProblem(CudaArch.SM120, CublasLtDType.F32, m, n, k),
            120803, algorithm, 0, 256, 256, 256, 256, 256, compute)
    attention = [RoundedAttentionPayload(CudaArch.SM120, v, len(cubin), module["sha256"],
        120803, (10, 11, 0, 1, 0, 0, 0, 0, 0), (10, 11, 0, 1, 0, 0, 0, 0, 0)) for v in (1, 2, 4)]
    selection = {"schema": "aginfer.fixed-algorithms.v1", "linear": [linear(1, 3, 2, 2).to_bytes().hex()],
        "patch": linear(256, 1152, 588, 1).to_bytes().hex(), "attention": [p.to_bytes().hex() for p in attention]}
    requests = selection_requests(FixedAlgorithms.from_dict(selection))
    probe = {"schema": "aginfer.lt-selection-probe.v1", "arch": 120, "cublaslt_version": 120803,
        "cuda_driver_version": 13000, "results": [{"algorithm": [10, 11, 0, 1, 0, 0, 0, 0, 0],
            "workspace_bytes": 0, "heuristic_rank": 0, "candidate_count": 1, "algo_check": True} for _ in requests]}
    files = {"compiler.py": {"bytes": 1, "sha256": "1" * 64}}
    report = {"schema": "aginfer.offline-selection.v1", "policy": POLICY,
        "compiler": {"files": files, "sha256": digest(files)}, "selector": files["compiler.py"], "module": module,
        "program_sha256": "2" * 64, "inventory_sha256": "3" * 64, "requests": requests, "probe": probe,
        "fixed_algorithms": selection, "selection_sha256": digest(selection),
        "numerical_validation": "not_run", "capture_validation": "not_run", "timing": "not_run"}
    report["report_sha256"] = digest(report)
    return selection, report, cubin


class AlgorithmSelectionTests(unittest.TestCase):
    def test_prefill_benchmark_envelope_and_version_are_separate(self):
        def request(m, n=32768, k=2048):
            return linear_request(CublasLtLinearProblem(CudaArch.SM120, CublasLtDType.BF16, m, n, k), 1)
        for m in (129, 968, 2048):
            self.assertTrue(benchmark_eligible(request(m, 16384), prefill=True))
            self.assertFalse(benchmark_eligible(request(m)))
        for r in (request(128), request(2049), request(2048), request(968, 32768, 32768)):
            self.assertFalse(benchmark_eligible(r, prefill=True))
        r = request(968)
        for field, value in [('dtype', 1), ('compute', 2), ('batch', 2), ('bias', 0), ('alignments', [16]*3)]:
            bad = deepcopy(r); bad[field] = value
            self.assertFalse(benchmark_eligible(bad, prefill=True))
        _, report, _ = fixture()
        probe = deepcopy(report['probe']); probe['schema'] = 'aginfer.lt-selection-probe.v3'
        item = deepcopy(probe['results'][0]); probe['results'] = [item]
        item.update(candidate_count=2, heuristic_rank=1, benchmark={
            'repeats': 8, 'input': 'synthetic-bf16-v1', 'candidates': [
                {'rank': 0, 'matches_baseline': True, 'times_us': [100, 101, 99]},
                {'rank': 1, 'matches_baseline': True, 'times_us': [90, 91, 89]}]})
        checked_results(probe, [r], prefill=True)
        with self.assertRaises(ValidationError): checked_results(probe, [r], benchmark=True)
        for mutation in ('rank', 'timing', 'missing'):
            bad = deepcopy(probe)
            if mutation == 'rank': bad['results'][0]['heuristic_rank'] = 0
            elif mutation == 'timing': bad['results'][0]['benchmark']['candidates'][1]['times_us'] = [90, 102, 89]
            else: bad['results'][0]['benchmark'] = None
            with self.assertRaises(ValidationError): checked_results(bad, [r], prefill=True)
        with self.assertRaises(ValidationError): checked_results(probe, [request(50)], prefill=True)

    def test_prefill_receipts_and_conflicting_flags(self):
        selection, report, cubin = fixture()
        report['policy'] = report['timing'] = PREFILL_BENCHMARK_POLICY
        report['probe']['schema'] = 'aginfer.lt-selection-probe.v3'
        for item in report['probe']['results']: item['benchmark'] = None
        report['report_sha256'] = digest({k: v for k, v in report.items() if k != 'report_sha256'})
        validate_selection_report(report, selection, cubin)
        # A timed prefix entry must survive the full payload/report validator,
        # not only the standalone probe validator. Other entries remain untimed.
        p = CublasLtLinearPayload.from_bytes(bytes.fromhex(selection['linear'][0]))
        p = replace(p, problem=CublasLtLinearProblem(CudaArch.SM120, CublasLtDType.BF16, 968, 32768, 2048), compute_mode=1)
        selection['linear'][0] = p.to_bytes().hex()
        report['requests'] = selection_requests(FixedAlgorithms.from_dict(selection))
        report['probe']['results'][0]['benchmark'] = {'repeats': 8, 'input': 'synthetic-bf16-v1',
            'candidates': [{'rank': 0, 'matches_baseline': True, 'times_us': [700, 701, 699]}]}
        report['selection_sha256'] = digest(selection)
        report['report_sha256'] = digest({k: v for k, v in report.items() if k != 'report_sha256'})
        validate_selection_report(report, selection, cubin)
        for kwargs in ({'benchmark_prefill_gemm': 1}, {'benchmark_prefill_gemm': True, 'benchmark_small_gemm': True}):
            with self.assertRaisesRegex(ValidationError, 'benchmark mode'):
                select_source_algorithms('absent', cubin_path='absent', selector_path='absent',
                    output='absent', report_path='absent', **kwargs)

    def test_grouped_softmax_receipt_keeps_gemm_descriptors(self):
        selection,report,cubin=fixture()
        p=RoundedAttentionPayload.from_bytes(bytes.fromhex(selection['attention'][0]))
        selection['attention'][0]=replace(p,softmax_warps=4).to_bytes().hex()
        report['selection_sha256']=digest(selection)
        report['report_sha256']=digest({k:v for k,v in report.items() if k!='report_sha256'})
        validate_selection_report(report,selection,cubin)
        self.assertEqual(selection_requests(FixedAlgorithms.from_dict(selection)),report['requests'])

    def test_small_gemm_timing_is_bounded_and_conservative(self):
        _, report, _ = fixture()
        request = linear_request(CublasLtLinearProblem(CudaArch.SM120,CublasLtDType.BF16,50,1024,1024),1)
        self.assertTrue(benchmark_eligible(request))
        for field, value in [('dtype',1),('batch',2),('bias',0),('trans_a',0),('alignments',[16]*3)]:
            other=deepcopy(request);other[field]=value
            self.assertFalse(benchmark_eligible(other))
        probe=deepcopy(report['probe']);probe['schema']='aginfer.lt-selection-probe.v2'
        item=deepcopy(probe['results'][0]);probe['results']=[item]
        item.update(candidate_count=4,heuristic_rank=2,benchmark={'repeats':8,'input':'synthetic-bf16-v1',
            'candidates':[{'rank':0,'matches_baseline':True,'times_us':[10,10,10]},
                          {'rank':1,'matches_baseline':False,'times_us':[]},
                          {'rank':2,'matches_baseline':True,'times_us':[9,9,9]},
                          {'rank':3,'matches_baseline':True,'times_us':[8,11,8]}]})
        checked_results(probe,[request],benchmark=True)
        for field,value in [('heuristic_rank',3),('benchmark',None)]:
            bad=deepcopy(probe);bad['results'][0][field]=value
            with self.assertRaises(ValidationError):checked_results(bad,[request],benchmark=True)
        for times in ([0,0,0],[float('nan')]*3,[True]*3,[9,9], [9.99]*3):
            bad=deepcopy(probe);bad['results'][0]['benchmark']['candidates'][2]['times_us']=times
            with self.assertRaises(ValidationError):checked_results(bad,[request],benchmark=True)
        with self.assertRaises(ValidationError):checked_results(probe,[request])

    def test_benchmark_policy_roundtrip_does_not_claim_model_validation(self):
        selection,report,cubin=fixture()
        report['policy']=report['timing']=BENCHMARK_POLICY
        report['probe']['schema']='aginfer.lt-selection-probe.v2'
        for item in report['probe']['results']:item['benchmark']=None
        report['report_sha256']=digest({k:v for k,v in report.items() if k!='report_sha256'})
        validate_selection_report(report,selection,cubin)
        for field in ('numerical_validation','capture_validation','timing'):
            bad=deepcopy(report);bad[field]='passed'
            bad['report_sha256']=digest({k:v for k,v in bad.items() if k!='report_sha256'})
            with self.assertRaises(ValidationError):validate_selection_report(bad,selection,cubin)

    def test_linear_column_major_transpose_and_precision(self):
        p = CublasLtLinearProblem(CudaArch.SM120, CublasLtDType.F32, 50, 1024, 32)
        r = linear_request(p, 2)
        self.assertEqual(r["layouts"], [[32, 1024, 32, 0], [32, 50, 32, 0], [1024, 50, 1024, 0]])
        self.assertEqual((r["compute"], r["trans_a"], r["bias"]), (2, 1, 1))
        self.assertEqual(r["workspace_limit"], WORKSPACE_LIMIT)

    def test_attention_broadcast_interleave_and_batch_alignment(self):
        selection, _, _ = fixture()
        algorithms = FixedAlgorithms.from_dict(selection)
        denoise = attention_requests(algorithms.attention[0])
        self.assertEqual(denoise[0]["layouts"][0], [256, 1018, 256, 0])
        self.assertEqual(denoise[0]["alignments"][2], 8)
        self.assertEqual(denoise[1]["alignments"][1], 8)
        self.assertEqual(denoise[1]["layouts"][2], [256, 50, 2048, 256])
        vision = attention_requests(algorithms.attention[2])
        self.assertEqual(vision[0]["compute"], 1)
        self.assertEqual(vision[0]["layouts"][0], [72, 256, 1152, 72])
        self.assertEqual(vision[1]["layouts"][2], [72, 256, 1152, 72])
        self.assertTrue(all(x["workspace_limit"] == 0 for x in denoise + vision))

    def test_protocol_is_bounded_canonical_numbers(self):
        _, report, _ = fixture()
        lines = encode_requests(report["requests"]).splitlines()
        self.assertEqual(lines[0], "AGINFER_LT_SELECT_V1 8")
        self.assertTrue(all(len(line.split()) == 22 for line in lines[1:]))
        with self.assertRaises(ValidationError): encode_requests([])
        bad = deepcopy(report["requests"])
        bad[0]["compute"] = True
        with self.assertRaises(ValidationError): encode_requests(bad)

    def test_probe_rejects_wrong_device_version_count_and_false_check(self):
        _, report, _ = fixture()
        self.assertEqual(len(checked_results(report["probe"], report["requests"])), 8)
        for key, value in (("arch", 90), ("arch", True), ("cublaslt_version", 120800), ("results", [])):
            bad = deepcopy(report["probe"]); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError): checked_results(bad, report["requests"])
        for key, value in (("algo_check", False), ("workspace_bytes", WORKSPACE_LIMIT + 1),
                           ("candidate_count", 0), ("algorithm", [0] * 8), ("heuristic_rank", True)):
            bad = deepcopy(report["probe"]); bad["results"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError): checked_results(bad, report["requests"])

    def test_report_binds_payloads_module_and_program(self):
        selection, report, cubin = fixture()
        self.assertIs(validate_selection_report(report, selection, cubin, "2" * 64), report)
        for module, program in ((b"other", "2" * 64), (cubin, "4" * 64)):
            with self.assertRaises(ValidationError): validate_selection_report(report, selection, module, program)

    def test_rehashed_claims_or_mismatched_probe_still_fail(self):
        selection, report, cubin = fixture()
        for case in ("numerical_validation", "request", "algorithm", "selector", "compiler", "program_sha256"):
            bad = deepcopy(report)
            if case == "request": bad["requests"][0]["compute"] = 1
            elif case == "algorithm": bad["probe"]["results"][0]["algorithm"][0] = 11
            elif case == "selector": bad["selector"] = {"bytes": True, "sha256": "1" * 64}
            elif case == "compiler": bad["compiler"]["sha256"] = "1" * 64
            elif case == "program_sha256": bad[case] = "not a digest"
            else: bad[case] = "passed"
            bad["report_sha256"] = digest({k: v for k, v in bad.items() if k != "report_sha256"})
            with self.subTest(case=case), self.assertRaises(ValidationError): validate_selection_report(bad, selection, cubin)

    def test_outputs_never_overwrite_before_any_gpu_or_source_work(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "keep.json"; path.write_bytes(b"keep")
            with self.assertRaisesRegex(ValidationError, "distinct new files"):
                select_source_algorithms("absent", cubin_path="absent", selector_path="absent",
                    output=path, report_path=Path(temp) / "report.json")
            self.assertEqual(path.read_bytes(), b"keep")

    def test_workspace_policy_is_explicit_and_old_budget_reports_still_verify(self):
        selection, report, cubin = fixture()
        report["requests"] = selection_requests(FixedAlgorithms.from_dict(selection), 8 * 1024 * 1024)
        report["report_sha256"] = digest({k: v for k, v in report.items() if k != "report_sha256"})
        validate_selection_report(report, selection, cubin)
        for value in (-1, True, 64 * 1024 * 1024 + 1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                selection_requests(FixedAlgorithms.from_dict(selection), value)


if __name__ == "__main__":
    unittest.main()
