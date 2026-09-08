from dataclasses import replace
import unittest

from aginfer.ir import DType, Device, Op, Region, TensorType, Value, attributes
from aginfer.lowering import build_execution_schedule, build_lowering_inventory, build_memory_plan, placements_from_partial_lowering
from aginfer.lowering.command import CommandOperand, CommandTag, OperandAccess, ProviderCommand
from aginfer.lowering.memory import AllocationRegion, replan_memory_for_commands
from aginfer.providers import RopePayload, RopeProblem, RopeVariant, lower_rope_commands, KvPackPayload, KvPackProblem, lower_kv_pack_commands
from aginfer.providers.build_binding import BuildBinding
from aginfer.providers.projection_split import lower_projection_splits, SplitCommand, SplitLowering
from aginfer.providers.qkv_rope_pack import QkvRopePackPayload, fuse_qkv_rope_pack
from aginfer.compiler.verify import decode_command_payload
from aginfer.errors import FormatError
from aginfer.schema import CudaArch
from test_aot_kv_pack_provider import _program


def fixture():
    p = _program(); f = p.functions[0]
    def t(shape, dtype=DType.BF16):
        return TensorType(dtype, shape, device=Device.CUDA)
    ops = []
    start = 0
    for name, heads in (('q', 8), ('k', 1), ('v', 1)):
        width = heads * 256
        ops.extend((Op('slice', ('qkv',), (Value(name+'slice', t((1,50,width))),),
                       attributes(axis=2,start=start,stop=start+width)),
                    Op('reshape', (name+'slice',), (Value(name, t((1,50,heads,256))),),
                       attributes(shape=(1,50,heads,256)))))
        if name != 'v':
            ops.extend((Op('transpose', (name,), (Value(name+'trans',t((1,heads,50,256))),),
                           attributes(permutation=(0,2,1,3))),
                        Op('rope_default', (name+'trans','positions'),
                           (Value('query' if name=='q' else 'current_k',t((1,heads,50,256))),),
                           attributes(frequency_dtype='bf16',pairing='split_half',theta=10000.0))))
        start += width
    for op in f.body.ops:
        if op.opcode == 'rope_default':
            continue
        ops.append(replace(op, inputs=tuple('v' if x=='current_v' else x for x in op.inputs)))
    inputs = (Value('qkv',t((1,50,2560))),) + tuple(v for v in f.inputs if v.value_id in ('positions','mask'))
    p = replace(p, functions=(replace(f,inputs=inputs,body=Region(tuple(ops))),))
    s = build_execution_schedule(p); inv = build_lowering_inventory(p); mem = build_memory_plan(s)
    variants = (RopeVariant.BF16_SEQUENCE50_HEADS8, RopeVariant.BF16_SEQUENCE50_HEADS1)
    bindings = tuple(BuildBinding((RopeProblem(CudaArch.SM120,v),),
                                 (RopePayload(RopeProblem(CudaArch.SM120,v),64000,'6'*64),)) for v in variants)
    kp = KvPackProblem(CudaArch.SM120)
    lowered = {'projection_split':lower_projection_splits(s,64000,'6'*64),
               'rope':lower_rope_commands(s,inv,mem,bindings,target_arch=CudaArch.SM120),
               'kv_pack':lower_kv_pack_commands(s,inv,mem,BuildBinding((kp,),(KvPackPayload(kp,64000,'6'*64),)),target_arch=CudaArch.SM120)}
    op = s.ops[-1]
    command = ProviderCommand(CommandTag.ATTENTION,5,1,0,'5'*64,
        tuple(CommandOperand(v,OperandAccess.READ) for v in op.inputs)+
        (CommandOperand(op.outputs[0],OperandAccess.WRITE),),b'fixture',0,0,False)
    lowered['attention'] = SplitLowering((SplitCommand(op.execution_index,(op.execution_index,),command),),())
    return s, mem, lowered


class QkvRopePackTests(unittest.TestCase):
    def test_join_aliases_preserve_coverage_and_remove_storage(self):
        s,m,l = fixture()
        before = {i for group in l.values() for p in placements_from_partial_lowering(s,group) for i in p.covered_execution_indices}
        old = l['projection_split'].commands[0].command.operands[1:]
        fuse_qkv_rope_pack(s,m,l,64000,'6'*64)
        self.assertEqual(set(l), {'attention','qkv_rope_pack'})
        c = l['qkv_rope_pack'].commands[0].command
        self.assertEqual(decode_command_payload(c),QkvRopePackPayload(64000,'6'*64))
        self.assertEqual([o.access.name for o in c.operands], ['READ']*4+['WRITE']*3)
        placements = tuple(p for g in l.values() for p in placements_from_partial_lowering(s,g))
        self.assertEqual(before, {i for p in placements for i in p.covered_execution_indices})
        memory = replan_memory_for_commands(s,m,placements)
        for o in old:
            self.assertEqual(memory.allocations[o.value_id].region,AllocationRegion.UNUSED)
        # Running the pass twice is a no-op.
        previous = dict(l); fuse_qkv_rope_pack(s,m,l,64000,'6'*64); self.assertEqual(previous,l)

    def test_exported_shared_partial_positions_and_module_refuse(self):
        for variant in ('export','shared','offset','positions','module','resident'):
            s,m,l = fixture()
            split = l['projection_split'].commands[0]
            if variant == 'export':
                # Export a reshape alias, not only the original split value.
                v = l['rope'].commands[0].command.operands[0].value_id
                s = replace(s,entry_outputs=(*s.entry_outputs,('extra',v)))
            elif variant == 'shared':
                group = l['attention']; c = group.commands[0]
                c = replace(c,command=replace(c.command,operands=(*c.command.operands,replace(split.command.operands[1],access=OperandAccess.READ))))
                l['attention'] = replace(group,commands=(c,))
            elif variant in ('offset','positions'):
                g = l['rope']; records = list(g.commands); c = records[0]; ops = list(c.command.operands)
                ops[1] = replace(ops[1], **({'byte_offset':16} if variant=='offset' else {'value_id':split.command.operands[0].value_id}))
                records[0] = replace(c,command=replace(c.command,operands=tuple(ops)))
                l['rope'] = replace(g,commands=tuple(records))
            elif variant == 'module':
                split = replace(split,command=replace(split.command,payload=split.command.payload[:40]+bytes.fromhex('7'*64)+split.command.payload[72:]))
                l['projection_split'] = replace(l['projection_split'],commands=(split,))
            else:
                del l['kv_pack']
            old = dict(l); fuse_qkv_rope_pack(s,m,l,64000,'6'*64)
            self.assertEqual(l,old,variant)

    def test_payload_canonical_fixed_envelope(self):
        p = QkvRopePackPayload(64000,'6'*64); data = p.to_bytes()
        self.assertEqual(QkvRopePackPayload.from_bytes(data),p)
        for offset in (0,8,10,12,16,20,24,28,72,127):
            bad = bytearray(data); bad[offset] ^= 8
            with self.assertRaises(FormatError): QkvRopePackPayload.from_bytes(bad)
        with self.assertRaises(FormatError): QkvRopePackPayload.from_bytes(data[:-1])
