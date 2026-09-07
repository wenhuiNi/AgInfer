from dataclasses import replace
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from aginfer.compiler.constant_folding import (discover_constant_region, evaluation_plan,
    apply_constants, validate_folding_report, POLICY)
from aginfer.compiler.identity import digest
from aginfer.errors import ValidationError
from aginfer.executable import compile_executable_plan, ExecutableValueRegion
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value, attributes, State, StateAccess
from aginfer.lowering import (build_execution_schedule, build_memory_plan, build_lowering_inventory,
    CommandOperand, CommandPlacement, ProviderCommand, CommandTag, OperandAccess, assemble_command_stream)
from aginfer.lowering.memory import AllocationRegion, replan_memory_for_commands
from aginfer.packed_weights import pack_command_weights
from aginfer.providers import AotActivationProblem, AotPointwiseProblem, CudaKernelPayload
from aginfer.schema import CudaArch


def fixture(dynamic=False, unknown=False, stateful=False, module_bytes=1000, module_sha256="1" * 64):
    t = TensorType(DType.F32, (1, 1024), device=Device.CUDA)
    inputs = (Value("observation", t),)
    ops = []
    if stateful:
        ops.append(Op("state_read", (), (Value("condition", t),), attributes(state="cache")))
    elif dynamic:
        inputs += (Value("condition", t),)
    else:
        ops.append(Op("constant", (), (Value("condition", t),), attributes(value=(1.0,) * 1024)))
    ops.extend((Op("silu", ("condition",), (Value("h", t),)),
                Op("silu", ("h",), (Value("style", t),)),
                Op("add", ("observation", "style"), (Value("out", t),))))
    program = Program((Function("unrelated_family", inputs, ("out",), Region(tuple(ops))),), "unrelated_family",
        states=(State("cache", t, StateAccess.READ_WRITE),) if stateful else ())
    inventory = build_lowering_inventory(program)
    by_site = {x.site_id: x for x in inventory.ops}
    schedule = build_execution_schedule(program)
    memory = build_memory_plan(schedule)
    placements = []
    for op in schedule.ops:
        if op.opcode == "state_read":
            continue
        kind = AotActivationProblem if op.opcode == "silu" else AotPointwiseProblem
        problem = kind.from_inventory(by_site[op.site], target_arch=CudaArch.SM120)
        payload = CudaKernelPayload.for_problem(problem, module_bytes=module_bytes, module_sha256=module_sha256).to_bytes()
        command = ProviderCommand(CommandTag.CUDA_KERNEL, 99 if unknown else 2, 1, 0,
            hashlib.sha256(payload).hexdigest(),
            tuple(CommandOperand(v, OperandAccess.READ) for v in op.inputs) +
            tuple(CommandOperand(v, OperandAccess.WRITE) for v in op.outputs), payload)
        placements.append(CommandPlacement(op.execution_index, (op.execution_index,), command))
    memory = replan_memory_for_commands(schedule, memory, placements)
    stream = assemble_command_stream(schedule, memory, placements, target_arch=CudaArch.SM120)
    return inventory, schedule, memory, stream, tuple(placements)


class ConstantFoldingTests(unittest.TestCase):
    def test_constant_closure_and_compact_boundary(self):
        i, s, m, c, placements = fixture()
        region = discover_constant_region(s, m, placements)
        self.assertEqual((len(region.selected), len(region.remaining), len(region.boundary)), (2, 1, 1))
        es, em, ec = evaluation_plan(s, m, c, region)
        self.assertEqual((es.entry_inputs, es.states), ((), ()))
        self.assertEqual(len(es.entry_outputs), 1)
        self.assertEqual(tuple(x.payload for x in ec.commands), tuple(x.command.payload for x in region.selected))
        values = {region.boundary[0]: struct.pack("<f", 0.5) * 1024}
        fm, fc = apply_constants(s, m, c, region, values)
        self.assertEqual(len(fc.commands), 1)
        self.assertEqual(fm.allocations[region.boundary[0]].region, AllocationRegion.CONSTANT)
        with tempfile.TemporaryDirectory() as path:
            weights = pack_command_weights(Path(path) / "weights.bin", s, fm, i, fc, computed_constants=values)
            plan = compile_executable_plan(s, fm, fc, weights.spans, weights_bytes=weights.byte_size)
            self.assertEqual(plan.values[region.boundary[0]].region, ExecutableValueRegion.WEIGHTS)
            self.assertEqual(weights.path.read_bytes(), values[region.boundary[0]])
            # The evaluation plan has no observation allocation/port.
            ew = pack_command_weights(Path(path) / "eval.bin", es, em, i, ec)
            ep = compile_executable_plan(es, em, ec, ew.spans, weights_bytes=ew.byte_size)
            self.assertEqual(len(ep.ports), 1)

    def test_dynamic_roots_and_unknown_provider_are_not_folded(self):
        for kwargs in ({"dynamic": True}, {"unknown": True}, {"stateful": True}):
            _, s, m, _, placements = fixture(**kwargs)
            with self.assertRaisesRegex(ValidationError, "no foldable"):
                discover_constant_region(s, m, placements)

    def test_bad_evaluation_bytes_fail_closed(self):
        _, s, m, c, placements = fixture()
        region = discover_constant_region(s, m, placements)
        for values in ({}, {region.boundary[0]: b"x"}, {region.boundary[0]: struct.pack("<f", float("nan")) * 1024}):
            with self.assertRaises(ValidationError):
                apply_constants(s, m, c, region, values)

    def test_nonzero_offsets_not_folded(self):
        _, s, m, _, placements = fixture()
        first = placements[0]
        operands = tuple(replace(o, byte_offset=4) if o.access == OperandAccess.READ else o
                         for o in first.command.operands)
        changed = (replace(first, command=replace(first.command, operands=operands)), *placements[1:])
        with self.assertRaisesRegex(ValidationError, "no foldable"):
            discover_constant_region(s, m, changed)

    def test_computed_weight_override_is_not_an_arbitrary_input_patch(self):
        i, s, m, c, placements = fixture()
        region = discover_constant_region(s, m, placements)
        good = {region.boundary[0]: bytes(4096)}
        fm, fc = apply_constants(s, m, c, region, good)
        with tempfile.TemporaryDirectory() as path:
            for bad in ({region.boundary[0]: b"short"}, {s.entry_inputs[0][1]: bytes(4096)}):
                with self.assertRaises(ValidationError):
                    pack_command_weights(Path(path) / "weights.bin", s, fm, i, fc, computed_constants=bad)

    def test_folding_record_binds_exact_packed_values(self):
        i, s, m, c, placements = fixture()
        region = discover_constant_region(s, m, placements)
        values = {region.boundary[0]: bytes(4096)}
        fm, fc = apply_constants(s, m, c, region, values)
        with tempfile.TemporaryDirectory() as path:
            weights = pack_command_weights(Path(path) / "weights.bin", s, fm, i, fc, computed_constants=values)
            plan = compile_executable_plan(s, fm, fc, weights.spans, weights_bytes=weights.byte_size)
        report = {"policy": POLICY, "evaluator": {"bytes": 1, "sha256": "1" * 64},
            "evaluation_weights": {"bytes": 4096, "sha256": "2" * 64},
            "evaluation_plan_sha256": "3" * 64, "source_commands_sha256": "4" * 64,
            "removed_commands": 2, "graph_repeat_bit_exact": True, "output_bytes": 4096,
            "output_sha256": hashlib.sha256(bytes(4096)).hexdigest(),
            "values": [{"value_id": region.boundary[0], "bytes": 4096, "sha256": weights.spans[0].sha256}]}
        report["sha256"] = digest(report)
        validate_folding_report(report, plan)
        for change in ({"removed_commands": 0}, {"output_bytes": 1},
                       {"values": [{"value_id": region.boundary[0], "bytes": 4096, "sha256": "5" * 64}]}):
            bad = {**report, **change}
            bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
            with self.assertRaises(ValidationError):
                validate_folding_report(bad, plan)


if __name__ == "__main__":
    unittest.main()
