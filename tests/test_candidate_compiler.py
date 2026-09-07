import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from aginfer.cli import main
from aginfer.compiler.identity import canonical, digest, read_json
from aginfer.compiler.native import FixedAlgorithms
from aginfer.compiler.pipeline import compile_source, validate_kernel_record
from aginfer.compiler.verify import decode_command_payload
from aginfer.errors import ValidationError
from aginfer.lowering import build_execution_schedule, build_lowering_inventory, build_memory_plan
from aginfer.lowering.command import CommandTag
from aginfer.providers import LayerNormPayload, LayerNormProblem, lower_layer_norm_commands
from aginfer.providers.build_binding import BuildBinding
from aginfer.providers.flashinfer_attention import FlashInferAttentionProblem, FlashInferPrefixAttentionProblem
from aginfer.providers.rounded_attention import RoundedAttentionPayload
from aginfer.schema import CudaArch
from tests.test_aot_layer_norm_provider import _program, _receipt


class CandidateCompilerTests(unittest.TestCase):
    def binding(self):
        problem = LayerNormProblem(CudaArch.SM120)
        return BuildBinding((problem,), (LayerNormPayload(problem, 64000, "6" * 64),))

    def command(self):
        program = _program()
        schedule = build_execution_schedule(program)
        return lower_layer_norm_commands(schedule, build_lowering_inventory(program),
            build_memory_plan(schedule), self.binding(), target_arch=CudaArch.SM120).commands[0].command

    def test_candidate_capability_is_not_a_measurement(self):
        command = self.command()
        self.assertFalse(command.capture_safe)
        item = build_lowering_inventory(_program()).ops[0]
        capability = self.binding().capability_for(item)
        self.assertEqual(capability.implementation_digest, hashlib.sha256(command.payload).hexdigest())
        self.assertFalse(capability.supports_capture)
        self.assertNotEqual(capability.digest, _receipt().capability_for(item).digest)
        self.assertEqual(decode_command_payload(command), self.binding().payload())

    def test_binding_rejects_mixed_modules_empty_and_provider_mismatch(self):
        binding = self.binding()
        for bad in (
            lambda: BuildBinding((), ()),
            lambda: replace(binding, provider_id=5),
            lambda: replace(binding, payloads=()),
            lambda: BuildBinding(binding.problems * 2, binding.payloads * 2),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                bad()

    def test_attention_variant_cannot_claim_another_boundary(self):
        payload = RoundedAttentionPayload(CudaArch.SM120, 1, 64000, "6" * 64)
        BuildBinding((FlashInferAttentionProblem(CudaArch.SM120),), (payload,), 5)
        with self.assertRaisesRegex(ValidationError, "boundary"):
            BuildBinding((FlashInferPrefixAttentionProblem(CudaArch.SM120),), (payload,), 5)

    def test_payload_verifier_rejects_tag_abi_workspace_and_unknown_provider(self):
        command = self.command()
        for bad in (replace(command, tag=CommandTag.ATTENTION),
                    replace(command, abi_minor=1), replace(command, workspace_bytes=256),
                    replace(command, provider_id=999), replace(command, payload=b"bad")):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                decode_command_payload(bad)

    def test_fixed_algorithms_and_json_fail_closed(self):
        for value in ({}, {"schema": "unknown"}, {"schema": "aginfer.fixed-algorithms.v1", "linear": [], "patch": "", "attention": []}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                FixedAlgorithms.from_dict(value)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metadata.json"
            for text in ('{"key":1,"key":2}', '{"key":NaN}', '{'):
                path.write_text(text)
                with self.assertRaises(ValidationError):
                    read_json(path)

    def test_kernel_record_wrong_module_and_missing_record_are_rejected(self):
        with self.assertRaises(ValidationError):
            validate_kernel_record({}, b"bad")
        record = {"schema": "aginfer.kernel-build.v1", "arch": 120,
            "module": {"bytes": 3, "sha256": "1" * 64}, "sources": {}, "nvcc": {},
            "flags": ["-cubin", "-arch=sm_120", "--std=c++17"], "contains_ptx": False}
        record["build_sha256"] = digest(record)
        with self.assertRaises(ValidationError):
            validate_kernel_record(record, b"bad")

    def test_compile_refuses_overwrite_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "keep.aim"
            output.write_bytes(b"keep")
            with self.assertRaisesRegex(ValidationError, "already exists"):
                compile_source("missing", output=output, cubin_path="missing",
                    kernel_record_path="missing", algorithms_path="missing")
            self.assertEqual(output.read_bytes(), b"keep")

    def test_cli_errors_are_clean_and_version_matches_container_schema(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["verify", "does-not-exist.aim"]), 2)
        self.assertIn("aginfer: error:", errors.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit):
            main(["--version"])
        self.assertIn("schema 2.0", output.getvalue())

    def test_canonical_identity_is_order_independent_and_rejects_nan(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            canonical({"metric": float("nan")})


if __name__ == "__main__":
    unittest.main()
