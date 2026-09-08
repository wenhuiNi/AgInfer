from dataclasses import replace
import unittest

from aginfer.compiler.linear_precision import bf16_large_linears
from aginfer.errors import ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value, attributes, dump_program


def fixture(name='encoder', rows=256, source_dtype=DType.BF16):
    def t(dtype,shape): return TensorType(dtype,shape,device=Device.CUDA)
    ops=[]
    for key,shape in (('w',(256,256)),('b',(256,))):
        if source_dtype == DType.F32:
            ops.append(Op('constant_ref',(),(Value(key+'32',t(source_dtype,shape)),),attributes(namespace='model',name=key)))
            continue
        ops += [Op('constant_ref',(),(Value(key,t(source_dtype,shape)),),attributes(namespace='model',name=key)),
                Op('cast',(key,),(Value(key+'32',t(DType.F32,shape)),),attributes(dtype='f32'))]
    ops += [Op('linear',('x','w32','b32'),(Value('a',t(DType.F32,(1,rows,256))),)),
            Op('linear',('x','w32','b32'),(Value('b_out',t(DType.F32,(1,rows,256))),))]
    return Program((Function(name,(Value('x',t(DType.F32,(1,rows,256))),),('a','b_out'),Region(tuple(ops))),),name)


class LinearPrecisionTests(unittest.TestCase):
    def test_unrelated_families_share_input_cast_and_preserve_boundary(self):
        for name in ('vision_family','unrelated_prefill_family'):
            p=fixture(name); result,report=bf16_large_linears(p)
            self.assertEqual(result.functions[0].inputs,p.functions[0].inputs)
            self.assertEqual(result.functions[0].outputs,p.functions[0].outputs)
            linears=[o for o in result.functions[0].body.ops if o.opcode=='linear']
            self.assertEqual(len(report['static_sites']),2)
            self.assertEqual(linears[0].inputs,linears[1].inputs)
            self.assertEqual(linears[0].inputs[1:],('w','b'))
            self.assertTrue(all(o.outputs[0].type.dtype==DType.BF16 for o in linears))
            self.assertEqual(dump_program(result),dump_program(bf16_large_linears(p)[0]))

    def test_small_rows_or_true_f32_parameters_refused(self):
        for p in (fixture(rows=50),fixture(source_dtype=DType.F32)):
            with self.assertRaisesRegex(ValidationError,'no eligible'): bf16_large_linears(p)

    def test_generated_names_do_not_collide(self):
        p=fixture(); f=p.functions[0]; t=f.inputs[0].type
        f=replace(f,inputs=(*f.inputs,Value('x.bf16',t)))
        result,_=bf16_large_linears(replace(p,functions=(f,)))
        self.assertTrue(any(o.inputs[0]=='x.bf16.1' for o in result.functions[0].body.ops if o.opcode=='linear'))


if __name__=='__main__': unittest.main()
