"""Explicit candidate precision form for large F32 linears with BF16 weights."""
from dataclasses import replace
import math

from ..ir import DType, Device, Op, Region, Value, attributes, verify_program
from ..errors import ValidationError

POLICY = 'bf16-large-linear-islands.v1'


def bf16_large_linears(program):
    """Keep F32 region boundaries/reductions, use BF16 GEMM islands inside.

    This is opt-in and changes numerical results. Only immutable BF16 weights
    and biases widened by the source IR qualify; never narrow F32 parameters
    silently. Dynamic F32 inputs round once to BF16, shared by sibling linears.
    Shapes and masks stay unchanged. No model IDs or parameter names are used.
    """
    verify_program(program)
    functions, sites = [], []
    for function in program.functions:
        producers = {v.value_id:op for op in function.body.ops for v in op.outputs}
        types = {v.value_id:v.type for v in function.inputs}
        types.update({v.value_id:v.type for op in function.body.ops for v in op.outputs})
        names = set(types); narrowed = {}; ops = []

        def fresh(name, tensor):
            suffix = 0; candidate = name + '.bf16'
            while candidate in names:
                suffix += 1; candidate = name + f'.bf16.{suffix}'
            names.add(candidate)
            return Value(candidate, replace(tensor, dtype=DType.BF16))

        def bf16_constant(name):
            cast = producers.get(name)
            if cast is None or cast.opcode != 'cast' or len(cast.inputs) != 1:
                return None
            source = producers.get(cast.inputs[0])
            if (source is None or source.opcode != 'constant_ref'
                    or types[cast.inputs[0]].dtype != DType.BF16
                    or types[name].dtype != DType.F32
                    or types[cast.inputs[0]].shape != types[name].shape):
                return None
            return cast.inputs[0]

        for op in function.body.ops:
            if op.opcode != 'linear' or len(op.inputs) != 3 or len(op.outputs) != 1:
                ops.append(op); continue
            x,w,b = (types[i] for i in op.inputs); y = op.outputs[0]
            if (any(t.dtype != DType.F32 or t.device != Device.CUDA or t.layout != 'row_major'
                    for t in (x,w,b,y.type))
                    or any(type(d) is not int for t in (x,w,b,y.type) for d in t.shape)
                    or math.prod(x.shape[:-1]) < 128 or min(w.shape) < 256):
                ops.append(op); continue
            weight, bias = (bf16_constant(i) for i in op.inputs[1:])
            if weight is None or bias is None:
                ops.append(op); continue
            if op.inputs[0] not in narrowed:
                value = fresh(op.inputs[0],x)
                ops.append(Op('cast',(op.inputs[0],),(value,),attributes(dtype='bf16')))
                narrowed[op.inputs[0]] = value.value_id
            out = fresh(y.value_id,y.type)
            ops.append(replace(op,inputs=(narrowed[op.inputs[0]],weight,bias),outputs=(out,)))
            ops.append(Op('cast',(out.value_id,),(y,),attributes(dtype='f32')))
            sites.append({'function':function.name,'output':y.value_id,
                          'm':math.prod(x.shape[:-1]),'n':w.shape[0],'k':w.shape[1]})
        functions.append(replace(function,body=Region(tuple(ops))))
    result = replace(program,functions=tuple(functions))
    verify_program(result)
    if not sites:
        raise ValidationError('no eligible BF16-weight large F32 linear islands')
    return result, {'policy':POLICY,'static_sites':sites,'numerical_validation':'not_run'}
