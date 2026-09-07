from __future__ import annotations

import dataclasses
import unittest

from aginfer.errors import ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value, attributes
from aginfer.lowering import (
    ProviderCapability,
    TensorSignature,
    build_lowering_inventory,
    dump_provider_capabilities,
    dump_provider_resolution,
    resolve_provider_capabilities,
)
from aginfer.schema import CudaArch


def _program(*, device: Device = Device.CUDA) -> Program:
    tensor = TensorType(DType.F32, (1, 2), device=device)
    weight = TensorType(DType.F32, (2, 2), device=device)
    bias = TensorType(DType.F32, (2,), device=device)
    leaf = Function(
        "transform",
        (Value("x", tensor),),
        ("out",),
        Region(
            (
                Op(
                    "constant_ref",
                    (),
                    (Value("weight", weight),),
                    attributes(namespace="model", name="weight"),
                ),
                Op(
                    "constant_ref",
                    (),
                    (Value("bias", bias),),
                    attributes(namespace="model", name="bias"),
                ),
                Op("linear", ("x", "weight", "bias"), (Value("projected", tensor),)),
                Op("relu", ("projected",), (Value("out", tensor),)),
            )
        ),
    )
    main = Function(
        "main",
        (Value("input", tensor),),
        ("output",),
        Region(
            (
                Op(
                    "call",
                    ("input",),
                    (Value("output", tensor),),
                    attributes(callee="transform", repeat=2),
                ),
            )
        ),
    )
    return Program(functions=(main, leaf), entry="main")


def _capability(item, provider_id: int) -> ProviderCapability:
    return ProviderCapability.exact_for(
        item,
        provider_id=provider_id,
        abi_major=1,
        abi_minor=2,
        provider_version="2026.09",
        implementation_id=f"implementation_{provider_id}",
        implementation_digest=(f"{provider_id:064x}"),
        target_arch=CudaArch.SM120,
        supports_capture=True,
        workspace_bytes=provider_id * 256,
    )


class ProviderCapabilityTests(unittest.TestCase):
    def test_exact_resolution_is_stable_and_preserves_expanded_hits(self) -> None:
        inventory = build_lowering_inventory(_program())
        required = [item for item in inventory.ops if item.status.value == "required"]
        capabilities = tuple(_capability(item, index + 1) for index, item in enumerate(required))
        first = resolve_provider_capabilities(
            inventory, capabilities, target_arch=CudaArch.SM120
        )
        second = resolve_provider_capabilities(
            inventory, reversed(capabilities), target_arch=CudaArch.SM120
        )
        self.assertTrue(first.complete)
        self.assertEqual(dump_provider_resolution(first), dump_provider_resolution(second))
        self.assertEqual(dump_provider_capabilities(capabilities), dump_provider_capabilities(reversed(capabilities)))
        self.assertEqual([item.site for item in first.receipts], ["transform:2", "transform:3"])
        self.assertEqual([item.executions for item in first.receipts], [2, 2])
        self.assertEqual(first.receipts[0].workspace_bytes, 256)
        self.assertTrue(first.receipts[0].supports_capture)
        self.assertEqual(first.to_dict()["summary"]["resolved_executions"], 4)  # type: ignore[index]

    def test_arch_dtype_shape_and_attributes_are_exact_match_inputs(self) -> None:
        inventory = build_lowering_inventory(_program())
        item = next(record for record in inventory.ops if record.opcode == "linear")
        exact = _capability(item, 1)
        wrong_dtype = TensorSignature("bf16", exact.input_types[0].shape, "row_major", "cuda")
        wrong_shape = TensorSignature("f32", (2, 2), "row_major", "cuda")
        candidates = (
            dataclasses.replace(exact, target_arch=CudaArch.SM89),
            dataclasses.replace(exact, provider_id=2, input_types=(wrong_dtype,) + exact.input_types[1:]),
            dataclasses.replace(exact, provider_id=3, input_types=(wrong_shape,) + exact.input_types[1:]),
            dataclasses.replace(exact, provider_id=4, attributes=(("variant", "other"),)),
        )
        resolution = resolve_provider_capabilities(
            inventory, candidates, target_arch=CudaArch.SM120, strict=False
        )
        issue = next(record for record in resolution.issues if record.site == item.site_id)
        self.assertEqual(issue.kind, "unresolved")
        self.assertIn("sm120", issue.detail)
        self.assertIn("f32[1x2]", issue.detail)

    def test_zero_and_multiple_matches_fail_closed(self) -> None:
        inventory = build_lowering_inventory(_program())
        item = next(record for record in inventory.ops if record.opcode == "linear")
        with self.assertRaisesRegex(ValidationError, "provider resolution incomplete"):
            resolve_provider_capabilities(inventory, (), target_arch=CudaArch.SM120)

        ambiguous = resolve_provider_capabilities(
            inventory,
            (_capability(item, 1), _capability(item, 2)),
            target_arch=CudaArch.SM120,
            strict=False,
        )
        issue = next(record for record in ambiguous.issues if record.site == item.site_id)
        self.assertEqual(issue.kind, "ambiguous")
        self.assertEqual(len(issue.candidates), 2)
        with self.assertRaisesRegex(ValidationError, "ambiguous"):
            ambiguous.require_complete()

    def test_inventory_blocker_cannot_be_overridden_by_a_capability(self) -> None:
        inventory = build_lowering_inventory(_program(device=Device.CPU))
        resolution = resolve_provider_capabilities(
            inventory, (), target_arch=CudaArch.SM120, strict=False
        )
        self.assertFalse(resolution.complete)
        self.assertEqual({item.kind for item in resolution.issues}, {"inventory_blocked"})
        self.assertTrue(all("CUDA" in item.detail for item in resolution.issues))

    def test_capability_records_and_registry_are_validated(self) -> None:
        inventory = build_lowering_inventory(_program())
        item = next(record for record in inventory.ops if record.opcode == "linear")
        capability = _capability(item, 1)
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            resolve_provider_capabilities(
                inventory,
                (capability, capability),
                target_arch=CudaArch.SM120,
                strict=False,
            )
        with self.assertRaisesRegex(ValidationError, "positive uint32"):
            dataclasses.replace(capability, provider_id=True)
        with self.assertRaisesRegex(ValidationError, "implementation_digest"):
            dataclasses.replace(capability, implementation_digest="0" * 64)
        meta = next(record for record in inventory.ops if record.opcode == "constant_ref")
        with self.assertRaisesRegex(ValidationError, "required inventory site"):
            ProviderCapability.exact_for(
                meta,
                provider_id=1,
                abi_major=1,
                abi_minor=0,
                provider_version="1",
                implementation_id="invalid",
                implementation_digest="1" * 64,
                target_arch=CudaArch.SM120,
                supports_capture=False,
                workspace_bytes=0,
            )


if __name__ == "__main__":
    unittest.main()
