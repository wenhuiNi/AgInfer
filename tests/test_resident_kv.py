from dataclasses import replace
import unittest
from aginfer.compiler.resident_kv import make_kv_resident
from aginfer.compiler.verify import decode_command_payload
from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, State, StateAccess, TensorType, Value, attributes, verify_program
from aginfer.ir.executor import Tensor, execute
from aginfer.lowering import build_execution_schedule, build_memory_plan, placements_from_partial_lowering
from aginfer.providers.state_update import StateUpdatePayload, StateUpdateProblem, lower_state_updates
from aginfer.schema import CudaArch


def fixture(prefix=3, suffix=2, width=8):
    def t(n): return TensorType(DType.BF16,(1,1,n,width),device=Device.CUDA)
    states=tuple(State(n,t(prefix),StateAccess.READ_WRITE) for n in ('a','b'))
    fill=Function('fill',(Value('k',t(prefix)),Value('v',t(prefix))),('k',),Region(tuple(
        Op('state_write',(i,),(),attributes(state=n)) for i,n in (('k','a'),('v','b')))))
    ops=[]
    for n,i in (('a','k'),('b','v')):
        ops.extend((Op('state_read',(),(Value(n,t(prefix)),),attributes(state=n)),
                    Op('concat',(n,i),(Value(n+'full',t(prefix+suffix)),),attributes(axis=2))))
    ops.append(Op('add',('afull','bfull'),(Value('out',t(prefix+suffix)),)))
    step=Function('step',(Value('k',t(suffix)),Value('v',t(suffix))),('out',),Region(tuple(ops)))
    inputs=(Value('pk',t(prefix)),Value('pv',t(prefix)),Value('sk',t(suffix)),Value('sv',t(suffix)))
    entry=Function('main',inputs,('result',),Region((
        Op('call',('pk','pv'),(Value('unused',t(prefix)),),attributes(callee='fill',repeat=1)),
        Op('call',('sk','sv'),(Value('result',t(prefix+suffix)),),attributes(callee='step',repeat=1)))))
    return Program((entry,fill,step),'main',states)


class ResidentKvTests(unittest.TestCase):
    def test_reference_update_handles_noncontiguous_outer_rows(self):
        source=TensorType(DType.F32,(2,2))
        target=TensorType(DType.F32,(2,3))
        p=Program((Function('main',(Value('x',source),),('y',),Region((
            Op('state_update',('x',),(),attributes(state='cache',axis=1,start=1)),
            Op('state_read',(),(Value('y',target),),attributes(state='cache'))))),),
            'main',(State('cache',target,StateAccess.READ_WRITE),))
        result=execute(p,{'x':Tensor(DType.F32,(2,2),(9.,8.,7.,6.))},
                       states={'cache':Tensor(DType.F32,(2,3),(1.,2.,3.,4.,5.,6.))})
        self.assertEqual(result.outputs[0].data,(1.,9.,8.,4.,7.,6.))

    def test_transform_parity_refresh_and_suffix_overwrite(self):
        for prefix,suffix,width in ((3,2,8),(5,1,16)):
            p=fixture(prefix,suffix,width); r=make_kv_resident(p).program
            def cpu(program):
                def value(v): return replace(v,type=replace(v.type,device=Device.CPU))
                return replace(program,states=tuple(replace(s,type=replace(s.type,device=Device.CPU)) for s in program.states),
                    functions=tuple(replace(f,inputs=tuple(value(v) for v in f.inputs),
                        body=Region(tuple(replace(op,outputs=tuple(value(v) for v in op.outputs)) for op in f.body.ops)))
                        for f in program.functions))
            p,r=cpu(p),cpu(r)
            def tensor(n,v): return Tensor(DType.BF16,(1,1,n,width),(v,)*(n*width))
            old={n:tensor(prefix,99) for n in ('a','b')}
            new={n:tensor(prefix+suffix,99) for n in ('a','b')}
            for observation in (1.,2.,1.):
                inputs={'pk':tensor(prefix,observation),'pv':tensor(prefix,3.),'sk':tensor(suffix,4.),'sv':tensor(suffix,5.)}
                a,b=execute(p,inputs,states=old),execute(r,inputs,states=new)
                self.assertEqual(a.outputs,b.outputs)
                old,new=dict(a.states),dict(b.states)
                frozen=new['a'].data[:prefix*width]
                # Same prefix, multiple different suffixes. A skipped suffix
                # refresh or wrong offset cannot match the source oracle.
                for current in (6.,7.):
                    inputs={'k':tensor(suffix,current),'v':tensor(suffix,8.)}
                    a,b=execute(p,inputs,states=old,function='step'),execute(r,inputs,states=new,function='step')
                    self.assertEqual(a.outputs,b.outputs)
                    self.assertEqual(b.state('a').data[:prefix*width],frozen)
                    self.assertEqual(b.state('a').data[prefix*width:],(current,)*(suffix*width))
                    new=dict(b.states)

    def test_bounded_update_verifier_rejects_wrong_axis_offset_and_readonly(self):
        p=make_kv_resident(fixture()).program
        f=p.functions[1]
        for kwargs in ({'start':-1},{'start':5},{'axis':9},{'axis':True}):
            op=f.body.ops[0]; attrs=dict(op.attributes);attrs.update(kwargs)
            bad=replace(f,body=Region((replace(op,attributes=attributes(**attrs)),)+f.body.ops[1:]))
            with self.assertRaises(ValidationError): verify_program(replace(p,functions=(p.functions[0],bad,p.functions[2])))
        with self.assertRaises(ValidationError):
            verify_program(replace(p,states=(replace(p.states[0],access=StateAccess.READ_ONLY),p.states[1])))
        with self.assertRaises(ValidationError): make_kv_resident(p)

    def test_cache_without_initial_refresh_is_refused(self):
        p=fixture()
        with self.assertRaisesRegex(ValidationError,'refresh'):
            make_kv_resident(replace(p,entry='step'))

    def test_mutable_cache_view_cannot_escape_through_reshape(self):
        p=fixture()
        step=p.functions[2]
        shape=(1,1,5,8)
        op=Op('reshape',('afull',),(Value('out',TensorType(DType.BF16,shape,device=Device.CUDA)),),attributes(shape=shape))
        step=replace(step,body=Region(step.body.ops[:-1]+(op,)))
        with self.assertRaisesRegex(ValidationError,'no complete'):
            make_kv_resident(replace(p,functions=p.functions[:2]+(step,)))

    def test_pair_lowering_payload_and_full_state_lifetime(self):
        p=make_kv_resident(fixture()).program
        s=build_execution_schedule(p); memory=build_memory_plan(s)
        lowered=lower_state_updates(s,64000,'6'*64)
        self.assertEqual(len(lowered.commands),2)
        self.assertEqual(len(placements_from_partial_lowering(s,lowered)),2)
        payloads=[decode_command_payload(item.command) for item in lowered.commands]
        self.assertEqual([(p.problem.count,p.problem.offset) for p in payloads],[(24,0),(16,24)])
        self.assertEqual(memory.state_bytes,512) # two aligned full state allocations
        for payload in payloads:
            self.assertEqual(StateUpdatePayload.from_bytes(payload.to_bytes()),payload)
            self.assertEqual(len(payload.to_bytes()),128)

    def test_payload_refuses_corruption_overflow_and_nonvector_sizes(self):
        p=StateUpdatePayload(StateUpdateProblem(CudaArch.SM120,40,16,24),64000,'6'*64)
        for offset in (0,8,10,12,16,24,32,80,127):
            bad=bytearray(p.to_bytes());bad[offset]^=1
            with self.subTest(offset=offset),self.assertRaises(FormatError):StateUpdatePayload.from_bytes(bad)
        for total,count,offset in ((40,16,32),(2**31,8,0),(40,7,0),(40,8,-1)):
            with self.assertRaises(ValidationError):StateUpdateProblem(CudaArch.SM120,total,count,offset)
