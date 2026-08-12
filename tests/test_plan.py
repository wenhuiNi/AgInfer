from __future__ import annotations

import unittest

from aginfer.errors import ValidationError
from aginfer.plan import compile_execution_plan, inspect_plan
from aginfer.schema import CudaArch


def _plan() -> dict:
    return {
        "cuda_arch": "sm89",
        "arena_bytes": 256,
        "workspace_bytes": 128,
        "shape_dispatch": [
            {
                "profile": "vector4",
                "inputs": [{"name": "input", "dtype": "F32", "shape": [4]}],
                "outputs": [{"name": "output", "dtype": "F32", "shape": [4]}],
                "launches": [
                    {
                        "kernel": "copy_f32",
                        "grid": [1, 1, 1],
                        "block": [4, 1, 1],
                        "arguments": [
                            {"tensor": "input"},
                            {"tensor": "output"},
                            {"scalar_u32": 4},
                            {"arena_offset": 0},
                            {"weights_offset": 0},
                        ],
                    }
                ],
            }
        ],
        "cuda_graph_templates": [{"profile": "vector4"}],
    }


class PlanTests(unittest.TestCase):
    def test_compiles_static_kernel_abi(self) -> None:
        compiled = compile_execution_plan(_plan(), CudaArch.SM89, weight_size=64)
        summary = inspect_plan(compiled.data)
        self.assertEqual(summary["arch"], 89)
        self.assertEqual(summary["profile_count"], 1)
        self.assertEqual(summary["tensor_count"], 2)
        self.assertEqual(summary["launch_count"], 1)
        self.assertEqual(summary["argument_count"], 5)
        self.assertEqual(compiled.input_names, ("input",))
        self.assertEqual(compiled.output_names, ("output",))

    def test_rejects_unknown_tensor_argument(self) -> None:
        plan = _plan()
        plan["shape_dispatch"][0]["launches"][0]["arguments"][0] = {"tensor": "missing"}
        with self.assertRaisesRegex(ValidationError, "unknown tensor"):
            compile_execution_plan(plan, CudaArch.SM89, weight_size=64)

    def test_rejects_host_tensor_kernel_argument(self) -> None:
        plan = _plan()
        plan["shape_dispatch"][0]["inputs"][0]["location"] = "host"
        with self.assertRaisesRegex(ValidationError, "host tensor"):
            compile_execution_plan(plan, CudaArch.SM89, weight_size=64)

    def test_rejects_oversized_thread_block(self) -> None:
        plan = _plan()
        plan["shape_dispatch"][0]["launches"][0]["block"] = [1024, 2, 1]
        with self.assertRaisesRegex(ValidationError, "more than 1024"):
            compile_execution_plan(plan, CudaArch.SM89, weight_size=64)


if __name__ == "__main__":
    unittest.main()
