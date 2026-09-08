from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import unittest

from aginfer import providers as p
from aginfer.compiler.constant_casts import fold_constant_casts, validate_cast_report
from aginfer.constant_conversion import widen_bf16, cast_chunks
from aginfer.compiler.projection_fusion import DerivedTensor
from aginfer.errors import ValidationError
from aginfer.executable import compile_executable_plan, ExecutableValueRegion
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value, attributes
from aginfer.lowering import (build_execution_schedule, build_lowering_inventory, build_memory_plan,
    replan_memory_for_commands, CommandPlacement, CommandOperand, ProviderCommand, CommandTag,
    OperandAccess, assemble_command_stream)
from aginfer.packed_weights import pack_command_weights
from aginfer.schema import CudaArch


class Store:
    def tensor(self, namespace, name):
        return DerivedTensor('BF16', (4,), 8)

    def iter_chunks(self, namespace, name, *, chunk_size):
        data = struct.pack('<4H', 0, 0x8000, 0x3f80, 1)
        for i in range(0, len(data), 3):
            yield data[i:i+3]


def fixture(dynamic=False, exported=False):
    b = TensorType(DType.BF16, (4,), device=Device.CUDA)
    f = TensorType(DType.F32, (4,), device=Device.CUDA)
    ops = [] if dynamic else [Op('constant_ref', (), (Value('w', b),), attributes(namespace='model', name='w'))]
    ops += [Op('cast', ('w',), (Value(f'f{i}', f),), attributes(dtype='f32')) for i in range(3)]
    ops += [Op('add', ('f0', 'f1'), (Value('sum', f),)), Op('add', ('sum', 'f2'), (Value('out', f),))]
    program = Program((Function('generic', (Value('w', b),) if dynamic else (Value('unused', f),),
        ('out', 'f0') if exported else ('out',), Region(tuple(ops))),), 'generic')
    inventory = build_lowering_inventory(program); by_site = {o.site_id:o for o in inventory.ops}
    s = build_execution_schedule(program); m = build_memory_plan(s); ps = []
    for op in s.ops:
        kind = p.AotCastProblem if op.opcode == 'cast' else p.AotPointwiseProblem
        problem = kind.from_inventory(by_site[op.site], target_arch=CudaArch.SM120)
        payload = p.CudaKernelPayload.for_problem(problem, module_bytes=1000, module_sha256='1'*64)
        c = ProviderCommand(CommandTag.CUDA_KERNEL, 2, 1, 0, '2'*64,
            tuple(CommandOperand(v, OperandAccess.READ) for v in op.inputs) +
            tuple(CommandOperand(v, OperandAccess.WRITE) for v in op.outputs), payload.to_bytes())
        ps.append(CommandPlacement(op.execution_index, (op.execution_index,), c))
    m = replan_memory_for_commands(s, m, ps)
    c = assemble_command_stream(s, m, ps, target_arch=CudaArch.SM120)
    return inventory, s, m, c, tuple(ps)


class ConstantCastTests(unittest.TestCase):
    def test_all_bf16_bit_patterns_widen_without_host_fp_arithmetic(self):
        data = struct.pack('<65536H', *range(65536))
        self.assertEqual(widen_bf16(data), struct.pack('<65536I', *(x << 16 for x in range(65536))))
        with self.assertRaises(ValidationError):
            widen_bf16(b'x')

    def test_three_uses_share_one_packed_constant(self):
        i,s,m,c,ps = fixture()
        fm, fc, derived, report = fold_constant_casts(s,m,c,ps)
        self.assertEqual((len(fc.commands),len(derived),report['shared_aliases']), (2,1,2))
        with tempfile.TemporaryDirectory() as folder:
            packed = pack_command_weights(Path(folder)/'weights.bin',s,fm,i,fc,
                constants=Store(),widened_constants=derived,chunk_size=5)
            plan = compile_executable_plan(s,fm,fc,packed.spans,weights_bytes=packed.byte_size)
            self.assertEqual(len(packed.spans),1)
            self.assertEqual(packed.path.read_bytes()[:16], struct.pack('<4I',0,0x80000000,0x3f800000,0x10000))
            root = next(iter(derived))
            self.assertEqual(plan.values[root].region,ExecutableValueRegion.WEIGHTS)
            self.assertEqual(sum(v.alias_of == root for v in plan.values),2)
            validate_cast_report(report,plan)
            from aginfer.compiler.identity import digest
            bad = {**report,'derived_bytes':0}
            bad['sha256'] = digest({k:v for k,v in bad.items() if k != 'sha256'})
            with self.assertRaises(ValidationError): validate_cast_report(bad,plan)

    def test_dynamic_casts_and_exported_outputs_remain_runtime(self):
        _,s,m,c,ps = fixture(dynamic=True)
        self.assertEqual(fold_constant_casts(s,m,c,ps)[1], c)
        _,s,m,c,ps = fixture(exported=True)
        _,fc,derived,report = fold_constant_casts(s,m,c,ps)
        self.assertEqual(report['removed_commands'],2)
        self.assertEqual(len(fc.commands),3)

    def test_truncated_or_changed_store_is_rejected(self):
        _,s,m,c,ps = fixture()
        _,_,derived,_ = fold_constant_casts(s,m,c,ps)
        target,source = next(iter(derived.items()))
        class Short(Store):
            def iter_chunks(self,*args,**kwargs): yield b'\0\0'
        class Wrong(Store):
            def tensor(self,*args): return DerivedTensor('F32',(4,),16)
        for store in (Short(),Wrong()):
            with self.assertRaises(ValidationError):
                list(cast_chunks(store,s.values[source],s.values[target],8,chunk_size=5))


if __name__ == '__main__': unittest.main()
