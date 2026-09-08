from dataclasses import replace
import unittest

from aginfer.compiler.dead_commands import eliminate_dead_commands
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value
from aginfer.lowering import (build_execution_schedule, build_lowering_inventory,
    build_memory_plan, CommandOperand, CommandPlacement, ProviderCommand,
    CommandTag, OperandAccess, assemble_command_stream)
from aginfer.lowering.memory import replan_memory_for_commands, AllocationRegion
from aginfer.schema import CudaArch


def fixture(export_dead=False, alias=False):
    t = TensorType(DType.F32, (4,), device=Device.CUDA)
    ops = (Op('silu', ('x',), (Value('live', t),)),
           Op('silu', ('x',), (Value('dead0', t),)),
           Op('silu', ('dead0',), (Value('dead1', t),)))
    outputs = ('live', 'dead1') if export_dead else ('live',)
    if alias:
        from aginfer.ir import attributes
        ops += (Op('reshape', ('dead1',), (Value('view', t),), attributes(shape=(4,))),)
        outputs = ('live', 'view')
    p = Program((Function('generic', (Value('x', t),), outputs, Region(ops)),), 'generic')
    s = build_execution_schedule(p); m = build_memory_plan(s)
    ps = tuple(CommandPlacement(op.execution_index, (op.execution_index,),
        ProviderCommand(CommandTag.CUDA_KERNEL, 2, 1, 0, '1' * 64,
            tuple(CommandOperand(v, OperandAccess.READ) for v in op.inputs) +
            tuple(CommandOperand(v, OperandAccess.WRITE) for v in op.outputs), b'fixture'))
        for op in s.ops if op.opcode == 'silu')
    return s, m, ps


class DeadCommandTests(unittest.TestCase):
    def test_transitive_dead_chain_and_complete_coverage(self):
        s, m, ps = fixture()
        m, kept = eliminate_dead_commands(s, m, ps)
        self.assertEqual(len(kept), 1)
        m = replan_memory_for_commands(s, m, kept)
        stream = assemble_command_stream(s, m, kept, target_arch=CudaArch.SM120)
        self.assertEqual(len(stream.commands), 1)
        self.assertEqual(m.allocations[ps[-1].command.operands[-1].value_id].region, AllocationRegion.UNUSED)

    def test_exported_features_and_aliases_are_live(self):
        for kw in ({'export_dead': True}, {'alias': True}):
            s, m, ps = fixture(**kw)
            self.assertEqual(len(eliminate_dead_commands(s, m, ps)[1]), 3)

    def test_unknown_provider_or_opcode_is_retained_with_dependencies(self):
        s, m, ps = fixture()
        changed = (*ps[:-1], replace(ps[-1], command=replace(ps[-1].command, provider_id=99)))
        self.assertEqual(len(eliminate_dead_commands(s, m, changed)[1]), 3)
        s = replace(s, ops=(*s.ops[:-1], replace(s.ops[-1], opcode='future_effect')))
        self.assertEqual(len(eliminate_dead_commands(s, m, ps)[1]), 3)

    def test_partial_write_is_not_dead(self):
        s, m, ps = fixture()
        cmd = ps[-1].command
        changed = (*ps[:-1], replace(ps[-1], command=replace(cmd, operands=(cmd.operands[0],
            replace(cmd.operands[-1], byte_offset=4)))))
        self.assertEqual(len(eliminate_dead_commands(s, m, changed)[1]), 3)

    def test_state_write_retains_its_producers(self):
        s, m, ps = fixture()
        output = ps[-1].command.operands[-1].value_id
        s = replace(s, states=(('cache', output),), ops=(*s.ops[:-1],
            replace(s.ops[-1], opcode='state_write', attributes=(('state', 'cache'),))))
        allocations = tuple(replace(a, region=AllocationRegion.STATE) if a.value_id == output else a
                            for a in m.allocations)
        self.assertEqual(len(eliminate_dead_commands(s, replace(m, allocations=allocations), ps)[1]), 3)


if __name__ == '__main__':
    unittest.main()
