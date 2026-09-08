from dataclasses import replace
import unittest

from test_aot_adaptive_rms_norm_provider import _program
from aginfer.ir import Op, Value
from aginfer.lowering import build_execution_schedule, build_lowering_inventory, build_memory_plan, placements_from_partial_lowering
from aginfer.lowering.memory import AllocationRegion, replan_memory_for_commands
from aginfer.lowering.schedule import ValueStorage
from aginfer.providers import AdaptiveRmsNormPayload, AdaptiveRmsNormProblem, lower_adaptive_rms_norm_commands
from aginfer.providers.build_binding import BuildBinding
from aginfer.providers.rounded_mul_add import lower_rounded_mul_add
from aginfer.providers.compact_gate import CompactGatePayload, compact_gate_commands
from aginfer.compiler.verify import decode_command_payload, _payload_arch
from aginfer.errors import FormatError
from aginfer.schema import CudaArch


def fixture(expose=False,shared=False,reverse=False,chain=False):
    program=_program(); f=program.functions[0]; t=f.inputs[0].type
    ops=list(f.body.ops)
    ops.extend([Op('mul',('gate_bf16','hidden') if reverse else ('hidden','gate_bf16'),(Value('product',t),)),
                Op('add',('product','hidden'),(Value('out',t),))])
    outputs=('normalized','out')+(('gate_bf16',) if expose else ())
    if shared:
        ops.append(Op('add',('gate_bf16','hidden'),(Value('extra',t),)));outputs+=('extra',)
    if chain:
        rename=lambda name: 'out' if name=='hidden' else 'second_'+name
        f=replace(f,inputs=(*f.inputs,Value('second_modulation',f.inputs[1].type)))
        for op in _program().functions[0].body.ops:
            ops.append(replace(op,inputs=tuple(rename(v) for v in op.inputs),
                outputs=tuple(replace(v,value_id=rename(v.value_id)) for v in op.outputs)))
        ops.extend((Op('mul',('second_normalized','second_gate_bf16'),(Value('product2',t),)),
                    Op('add',('product2','out'),(Value('out2',t),))))
        outputs+=('second_normalized','out2')
    f=replace(f,body=replace(f.body,ops=tuple(ops)),outputs=outputs)
    program=replace(program,functions=(f,))
    schedule=build_execution_schedule(program);inventory=build_lowering_inventory(program)
    problem=AdaptiveRmsNormProblem(CudaArch.SM120)
    binding=BuildBinding((problem,),(AdaptiveRmsNormPayload(problem,64000,'6'*64),))
    norm=lower_adaptive_rms_norm_commands(schedule,inventory,build_memory_plan(schedule),binding,target_arch=CudaArch.SM120)
    residual=lower_rounded_mul_add(schedule,64000,'6'*64,excluded=norm.fused_execution_indices)
    return schedule,{'adaptive':norm,'rounded_mul_add':residual}


class CompactGateTests(unittest.TestCase):
    def test_residual_next_norm_fusion_preserves_both_outputs_and_coverage(self):
        s,lowered=fixture(chain=True)
        before={i for result in lowered.values() for c in placements_from_partial_lowering(s,result) for i in c.covered_execution_indices}
        compact_gate_commands(s,lowered,64000,'6'*64)
        commands=[c for result in lowered.values() for c in result.commands]
        fused=[c for c in commands if c.command.payload.startswith(b'AIGRN1')]
        self.assertEqual(len(fused),1)
        self.assertEqual(len(fused[0].command.operands),6)
        self.assertEqual(decode_command_payload(fused[0].command).kind,'residual_norm')
        self.assertEqual(before,{i for c in commands for i in c.fused_execution_indices})
        placements=tuple(p for result in lowered.values() for p in placements_from_partial_lowering(s,result))
        memory=replan_memory_for_commands(s,build_memory_plan(s),placements)
        for o in fused[0].command.operands[-2:]:
            self.assertNotEqual(memory.allocations[o.value_id].region,AllocationRegion.UNUSED)

    def test_both_boundaries_and_unused_gate_allocation(self):
        for reverse in (False,True):
            s,lowered=fixture(reverse=reverse)
            gate=lowered['adaptive'].commands[0].command.operands[3].value_id
            modulation=lowered['adaptive'].commands[0].command.operands[1].value_id
            compact_gate_commands(s,lowered,64000,'6'*64)
            nc=lowered['adaptive'].commands[0].command;rc=lowered['rounded_mul_add'].commands[0].command
            self.assertEqual(decode_command_payload(nc).kind,'norm')
            self.assertEqual(decode_command_payload(rc).kind,'residual')
            self.assertEqual(rc.operands[1].value_id,modulation)
            self.assertNotIn(gate,[o.value_id for c in (nc,rc) for o in c.operands])
            placements=tuple(p for result in lowered.values() for p in placements_from_partial_lowering(s,result))
            memory=replan_memory_for_commands(s,build_memory_plan(s),placements)
            self.assertEqual(memory.allocations[gate].region,AllocationRegion.UNUSED)

    def test_shared_and_exported_gates_are_retained(self):
        for kwargs in ({'expose':True},{'shared':True}):
            s,lowered=fixture(**kwargs);before=lowered['adaptive'].commands
            compact_gate_commands(s,lowered,64000,'6'*64)
            self.assertEqual(lowered['adaptive'].commands,before)

    def test_mutable_or_aliased_modulation_is_not_read_late(self):
        for change in ({'storage':ValueStorage.STATE},{'alias_of':0}):
            s,lowered=fixture();before=lowered['adaptive'].commands
            v=before[0].command.operands[1].value_id
            values=list(s.values);values[v]=replace(values[v],**change);s=replace(s,values=tuple(values))
            compact_gate_commands(s,lowered,64000,'6'*64)
            self.assertEqual(lowered['adaptive'].commands,before)

    def test_fixed_payload_and_reserved_fields(self):
        for kind in ('norm','residual'):
            p=CompactGatePayload(kind,64000,'6'*64)
            self.assertEqual(CompactGatePayload.from_bytes(p.to_bytes()),p)
            self.assertEqual(_payload_arch(CompactGatePayload.from_bytes(p.to_bytes())),CudaArch.SM120)
            for i in (0,8,10,12,16,64,127):
                bad=bytearray(p.to_bytes());bad[i]^=1
                with self.assertRaises(FormatError):CompactGatePayload.from_bytes(bad)


if __name__=='__main__':unittest.main()
