"""Physical cross-command form: omit gate broadcast, round modulation at its consumer."""
from dataclasses import dataclass, replace
import hashlib

from .rounded_mul_add import RoundedMulAddPayload, RoundedMulAddProblem, MAGIC
from ..errors import FormatError, ValidationError
from ..lowering.capability import ProviderCapability
from ..lowering.command import OperandAccess
from ..lowering.inventory import LoweringKind
from ..lowering.schedule import ValueStorage
from ..schema import CudaArch

MAGICS = {'norm': b'AIANG1\0\0', 'residual': b'AIGRD1\0\0', 'residual_norm': b'AIGRN1\0\0'}


@dataclass(frozen=True)
class CompactGatePayload:
    kind: str
    module_bytes: int
    module_sha256: str

    def __post_init__(self):
        if self.kind not in MAGICS:
            raise ValidationError('unknown compact-gate form')
        RoundedMulAddPayload(RoundedMulAddProblem(CudaArch.SM120,51200),self.module_bytes,self.module_sha256)

    def to_bytes(self):
        base=RoundedMulAddPayload(RoundedMulAddProblem(CudaArch.SM120,51200),self.module_bytes,self.module_sha256)
        return MAGICS[self.kind] + base.to_bytes()[8:]

    @classmethod
    def from_bytes(cls,data):
        kind=next((k for k,v in MAGICS.items() if bytes(data[:8])==v),None)
        if kind is None:
            raise FormatError('unknown compact-gate magic')
        base=RoundedMulAddPayload.from_bytes(MAGIC+bytes(data[8:]))
        if base.problem.numel!=51200:
            raise FormatError('compact-gate form requires rows50/width1024')
        return cls(kind,base.module_bytes,base.module_sha256)


def compact_gate_commands(schedule, lowered, module_bytes, module_sha256):
    """Negotiate both boundaries together; shared/exported/aliased gates stay unchanged.

    Uses only already-verified exact norm and residual forms, not model names.
    ProgramIR semantics stay full-shaped; the physical stream no longer exposes
    the gate at all. Command liveness therefore eliminates its allocation.
    """
    if 'rounded_mul_add' not in lowered:
        return
    consumers={}
    for op in schedule.ops:
        for v in op.inputs: consumers.setdefault(v,[]).append(op)
    exported={v for _,v in schedule.entry_outputs}
    norms=list(lowered['adaptive'].commands)
    residuals=list(lowered['rounded_mul_add'].commands)
    caps={'adaptive':list(lowered['adaptive'].capabilities),
          'rounded_mul_add':list(lowered['rounded_mul_add'].capabilities)}
    def change(placed,kind,operands,group):
        payload=CompactGatePayload(kind,module_bytes,module_sha256).to_bytes()
        inputs=tuple(schedule.values[o.value_id].type for o in operands if o.access==OperandAccess.READ)
        outputs=tuple(schedule.values[o.value_id].type for o in operands if o.access==OperandAccess.WRITE)
        cap=ProviderCapability(2,1,0,'aginfer-aot-cuda=1','aginfer.compact_gate.'+kind+'.v1',
            hashlib.sha256(payload).hexdigest(),CudaArch.SM120,LoweringKind.AOT_CUDA,
            'compact_gate_'+kind,inputs,outputs,(('gate_round','bf16'),),False,0)
        caps[group].append(cap)
        return replace(placed,command=replace(placed.command,operands=operands,payload=payload,
            capability_digest=cap.digest,capture_safe=False))
    for n,norm in enumerate(norms):
        operands=norm.command.operands
        if (not norm.command.payload.startswith(b'AIARM1') or len(operands)!=4
                or any(o.byte_offset for o in operands)):
            continue
        gate=operands[3].value_id
        modulation=schedule.values[operands[1].value_id]
        if (modulation.alias_of is not None or modulation.storage not in
                (ValueStorage.CONSTANT,ValueStorage.TEMPORARY,ValueStorage.ENTRY_INPUT)):
            continue
        users=consumers.get(gate,())
        if gate in exported or len(users)!=1 or users[0].opcode!='mul':
            continue
        for i,residual in enumerate(residuals):
            ro=residual.command.operands
            if (not residual.command.payload.startswith(MAGIC) or len(ro)!=4
                    or any(o.byte_offset for o in ro)
                    or users[0].execution_index not in residual.fused_execution_indices
                    or sum(o.value_id==gate for o in ro)!=1):
                continue
            gate_slot=next((j for j in (0,1) if ro[j].value_id==gate),None)
            if gate_slot is None:
                continue
            # These exact providers agree on BF16 [1,50,1024] and F32 [1,3072].
            t=schedule.values[gate].type
            if t.shape!=(1,50,1024) or any(schedule.values[o.value_id].type!=t for o in ro):
                continue
            norms[n]=change(norm,'norm',operands[:3],'adaptive')
            residuals[i]=change(residual,'residual',
                (ro[1-gate_slot],operands[1],ro[2],ro[3]),'rounded_mul_add')
            break
    for key,commands in (('adaptive',norms),('rounded_mul_add',residuals)):
        lowered[key]=replace(lowered[key],commands=tuple(commands),capabilities=tuple(caps[key]))
    fuse_residual_norm_commands(schedule,lowered,module_bytes,module_sha256)


def fuse_residual_norm_commands(schedule,lowered,module_bytes,module_sha256):
    """Keep both residual and normalized outputs; only remove their reread/launch."""
    norms=lowered['adaptive']; residuals=lowered['rounded_mul_add']
    # Include shared broadcasts/constant producers assigned by the existing
    # coverage mechanism before moving an entire norm into another group.
    from ..lowering.assemble import placements_from_partial_lowering
    coverage={p.execution_index:p.covered_execution_indices
              for p in placements_from_partial_lowering(schedule,norms)}
    norms=replace(norms,commands=tuple(replace(n,fused_execution_indices=coverage[n.execution_index])
                                      for n in norms.commands))
    targets={}
    for n in norms.commands:
        if n.command.payload.startswith(MAGICS['norm']):
            targets.setdefault(n.command.operands[0].value_id,[]).append(n)
    replaced=[]; removed=set(); covered=set(); caps=list(residuals.capabilities)
    for r in residuals.commands:
        ro=r.command.operands
        matches=targets.get(ro[-1].value_id,())
        if not r.command.payload.startswith(MAGICS['residual']) or len(matches)!=1:
            replaced.append(r); continue
        n=matches[0]; no=n.command.operands
        # Moving either modulation read across mutable state requires a separate
        # state-aware executable form; do not infer immutability from an alias.
        if any(schedule.values[o.value_id].alias_of is not None or
               schedule.values[o.value_id].storage not in (ValueStorage.CONSTANT,ValueStorage.TEMPORARY,ValueStorage.ENTRY_INPUT)
               for o in (ro[1],no[1])):
            replaced.append(r); continue
        operands=(*ro[:3],no[1],ro[3],no[2])
        payload=CompactGatePayload('residual_norm',module_bytes,module_sha256).to_bytes()
        cap=ProviderCapability(2,1,0,'aginfer-aot-cuda=1','aginfer.compact_gate.residual_norm.v1',
            hashlib.sha256(payload).hexdigest(),CudaArch.SM120,LoweringKind.AOT_CUDA,'compact_gate_residual_norm',
            tuple(schedule.values[o.value_id].type for o in operands[:4]),
            tuple(schedule.values[o.value_id].type for o in operands[4:]),(('gate_round','bf16'),),False,0)
        indices=tuple(sorted(set(r.fused_execution_indices)|set(n.fused_execution_indices)))
        replaced.append(replace(r,fused_execution_indices=indices,command=replace(r.command,
            operands=operands,payload=payload,capability_digest=cap.digest,capture_safe=False)))
        caps.append(cap); removed.add(n.execution_index); covered.update(n.fused_execution_indices)
    if not removed:
        return
    lowered['rounded_mul_add']=replace(residuals,commands=tuple(replaced),capabilities=tuple(caps))
    lowered['adaptive']=replace(norms,commands=tuple(n for n in norms.commands if n.execution_index not in removed),
        fused_execution_indices=tuple(i for i in norms.fused_execution_indices if i not in covered))
