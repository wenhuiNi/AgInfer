from dataclasses import replace
from pathlib import Path
import sys
import unittest

from aginfer.compiler.verify import decode_command_payload
from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value
from aginfer.lowering import build_execution_schedule, build_memory_plan, placements_from_partial_lowering, assemble_command_stream
from aginfer.lowering.memory import AllocationRegion, replan_memory_for_commands
from aginfer.providers.rounded_mul_add import RoundedMulAddPayload, RoundedMulAddProblem, lower_rounded_mul_add
from aginfer.schema import CudaArch


def program(reverse=False, exported=False, shared=False, dtype=DType.BF16, shape=(1,50,1024), two=False):
    t=TensorType(dtype,shape,device=Device.CUDA)
    ops=[Op('mul',('a','b'),(Value('product',t),))]
    r='c'
    if two:
        ops.append(Op('mul',('b','c'),(Value('product2',t),)));r='product2'
    ops.append(Op('add',(r,'product') if reverse else ('product',r),(Value('out',t),)))
    outputs=('out','product') if exported else ('out',)
    if shared:
        ops.append(Op('mul',('product','c'),(Value('extra',t),)));outputs+=('extra',)
    return Program((Function('generic_residual',tuple(Value(v,t) for v in ('a','b','c')),outputs,Region(tuple(ops))),),'generic_residual')


class RoundedMulAddTests(unittest.TestCase):
    def test_fusion_liveness_and_order(self):
        for shape in ((37,), (1,50,1024)):
            for reverse in (True,False):
                s=build_execution_schedule(program(reverse=reverse,shape=shape))
                result=lower_rounded_mul_add(s,64000,'6'*64)
                self.assertEqual(len(result.commands),1)
                c=result.commands[0]
                self.assertEqual(c.fused_execution_indices,(0,1));self.assertEqual(len(c.command.operands),4)
                self.assertEqual(decode_command_payload(c.command).module_bytes,64000)
                placements=placements_from_partial_lowering(s,result)
                memory=replan_memory_for_commands(s,build_memory_plan(s),placements)
                self.assertEqual(memory.allocations[s.ops[0].outputs[0]].region,AllocationRegion.UNUSED)
                self.assertEqual(len(assemble_command_stream(s,memory,placements,target_arch=CudaArch.SM120).commands),1)
                self.assertFalse(c.command.capture_safe)

    def test_refusals_and_unique_ownership(self):
        for kwargs in ({'exported':True},{'shared':True},{'dtype':DType.F32},{'shape':(1<<27,)}):
            self.assertFalse(lower_rounded_mul_add(build_execution_schedule(program(**kwargs)),1,'1'*64).commands)
        s=build_execution_schedule(program())
        self.assertFalse(lower_rounded_mul_add(s,1,'1'*64,excluded=(0,)).commands)
        self.assertFalse(lower_rounded_mul_add(s,1,'1'*64,excluded=(1,)).commands)
        self.assertEqual(len(lower_rounded_mul_add(build_execution_schedule(program(two=True)),1,'1'*64).commands),1)

    def test_payload_roundtrip_refuses_corruption(self):
        p=RoundedMulAddPayload(RoundedMulAddProblem(CudaArch.SM120,51200),64000,'6'*64)
        self.assertEqual(RoundedMulAddPayload.from_bytes(p.to_bytes()),p)
        for index in (0,8,10,12,23,64,127):
            data=bytearray(p.to_bytes());data[index]^=128
            with self.assertRaises(FormatError):RoundedMulAddPayload.from_bytes(data)
        for n in (0,-1,True,(1<<26)+1):
            with self.assertRaises(ValidationError):RoundedMulAddProblem(CudaArch.SM120,n)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--fixture':
        Path(sys.argv[2]).write_bytes(RoundedMulAddPayload(RoundedMulAddProblem(CudaArch.SM120,51200),64000,'6'*64).to_bytes())
    else:unittest.main()
