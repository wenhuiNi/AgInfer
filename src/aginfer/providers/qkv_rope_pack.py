"""Packed projection -> rounded RoPE and packed attention KV, in one launch.

This physical form preserves the original prefix copy and all rounding points.
It does not compact tokens, change cache cadence, or select an attention tactic.
"""
from dataclasses import dataclass, replace
import hashlib

from .projection_split import ProjectionSplitPayload, ProjectionSplitProblem, SplitCommand, SplitLowering, MAGIC as SPLIT_MAGIC
from .aot_rope import RopePayload, RopeVariant
from .aot_kv_pack import KvPackPayload
from ..errors import FormatError, ValidationError
from ..lowering.assemble import placements_from_partial_lowering
from ..lowering.capability import ProviderCapability
from ..lowering.command import OperandAccess
from ..lowering.inventory import LoweringKind
from ..lowering.memory import AllocationRegion
from ..schema import CudaArch

MAGIC = b'AIQRP1\0\0'
PROBLEM = ProjectionSplitProblem(CudaArch.SM120, 50, (2048, 256, 256))


@dataclass(frozen=True)
class QkvRopePackPayload:
    module_bytes: int
    module_sha256: str

    def __post_init__(self):
        ProjectionSplitPayload(PROBLEM, self.module_bytes, self.module_sha256)

    @property
    def problem(self):
        return PROBLEM

    def to_bytes(self):
        return MAGIC + ProjectionSplitPayload(PROBLEM, self.module_bytes, self.module_sha256).to_bytes()[8:]

    @classmethod
    def from_bytes(cls, data):
        if bytes(data[:8]) != MAGIC:
            raise FormatError('invalid QKV RoPE pack magic')
        base = ProjectionSplitPayload.from_bytes(SPLIT_MAGIC + bytes(data[8:]))
        if base.problem != PROBLEM:
            raise FormatError('QKV RoPE pack requires s50/h8+1/d256, prefix968')
        return cls(base.module_bytes, base.module_sha256)


def fuse_qkv_rope_pack(schedule, memory, lowered, module_bytes, module_sha256):
    """Join verified forms by storage identity, preserving public/shared values.

    Reshape aliases are resolved through the existing memory plan. Every removed
    intermediate must be private arena storage, with exactly its expected reader.
    State-read aliases are retained as operands, so normal state ordering applies.
    Unsupported or resident-cache forms retain their original commands.
    """
    groups = ('projection_split', 'rope', 'kv_pack')
    if any(g not in lowered or not lowered[g].commands for g in groups):
        return
    allocations = {a.value_id: a for a in memory.allocations}
    def root(v):
        seen = set()
        while allocations[v].region == AllocationRegion.ALIAS:
            if v in seen:
                raise ValidationError('QKV fusion encountered cyclic storage alias')
            seen.add(v)
            v = allocations[v].alias_of
        return v
    exported = {root(v) for _, v in schedule.entry_outputs}
    readers = {}
    for group in lowered.values():
        for record in group.commands:
            for o in record.command.operands:
                if o.access in (OperandAccess.READ, OperandAccess.READ_WRITE):
                    readers.setdefault(root(o.value_id), []).append(record.execution_index)
    coverage = {g: {p.execution_index: p.covered_execution_indices
                   for p in placements_from_partial_lowering(schedule, lowered[g])} for g in groups}
    ropes = {}
    for r in lowered['rope'].commands:
        ropes.setdefault(root(r.command.operands[0].value_id), []).append(r)
    packs = {}
    for p in lowered['kv_pack'].commands:
        packs.setdefault(root(p.command.operands[2].value_id), []).append(p)
    def private(value, reader):
        v = root(value)
        return (v not in exported and allocations[v].region == AllocationRegion.ARENA
                and readers.get(v) == [reader.execution_index])
    def valid(record, accesses):
        return (len(record.command.operands) == len(accesses)
                and all(o.byte_offset == 0 and o.access == a
                        for o, a in zip(record.command.operands, accesses)))
    R, W = OperandAccess.READ, OperandAccess.WRITE
    removed = {g: set() for g in groups}
    commands, caps = [], []
    for split in lowered['projection_split'].commands:
        so = split.command.operands
        if not valid(split, (R, W, W, W)):
            continue
        base = ProjectionSplitPayload.from_bytes(split.command.payload)
        if base.problem != PROBLEM:
            continue
        qs, ks = (ropes.get(root(o.value_id), ()) for o in so[1:3])
        if len(qs) != 1 or len(ks) != 1:
            continue
        q, k = qs[0], ks[0]
        if q.execution_index == k.execution_index or not all(valid(r, (R, R, W)) for r in (q, k)):
            continue
        qp, kp = (RopePayload.from_bytes(r.command.payload) for r in (q, k))
        if (qp.problem.variant != RopeVariant.BF16_SEQUENCE50_HEADS8
                or kp.problem.variant != RopeVariant.BF16_SEQUENCE50_HEADS1):
            continue
        qo, ko = q.command.operands, k.command.operands
        ps = packs.get(root(ko[2].value_id), ())
        if len(ps) != 1 or root(qo[1].value_id) != root(ko[1].value_id):
            continue
        pack = ps[0]; po = pack.command.operands
        if not valid(pack, (R, R, R, R, W, W)) or root(po[3].value_id) != root(so[3].value_id):
            continue
        pp = KvPackPayload.from_bytes(pack.command.payload)
        if any((p.module_bytes, p.module_sha256) != (module_bytes, module_sha256) for p in (base, qp, kp, pp)):
            continue
        if not all(private(v, r) for v, r in ((so[1].value_id, q), (so[2].value_id, k),
                                              (so[3].value_id, pack), (ko[2].value_id, pack))):
            continue
        if allocations[root(qo[1].value_id)].region == AllocationRegion.STATE:
            continue
        operands = (so[0], qo[1], po[0], po[1], qo[2], po[4], po[5])
        payload = QkvRopePackPayload(module_bytes, module_sha256).to_bytes()
        cap = ProviderCapability(2, 1, 0, 'aginfer-aot-cuda=1', 'aginfer.qkv_rope_pack.bf16.v1',
            hashlib.sha256(payload).hexdigest(), CudaArch.SM120, LoweringKind.AOT_CUDA,
            'qkv_rope_pack', tuple(schedule.values[o.value_id].type for o in operands[:4]),
            tuple(schedule.values[o.value_id].type for o in operands[4:]),
            (('prefix_sequence', 968), ('rounding', 'bf16_products_and_sum')), False, 0)
        selected = (('projection_split', split), ('rope', q), ('rope', k), ('kv_pack', pack))
        covered = tuple(sorted({i for g, r in selected for i in coverage[g][r.execution_index]}))
        commands.append(SplitCommand(split.execution_index, covered, replace(split.command,
            operands=operands, payload=payload, capability_digest=cap.digest, capture_safe=False)))
        caps.append(cap)
        for g, r in selected:
            removed[g].add(r.execution_index)
    if not commands:
        return
    for g in groups:
        old = lowered[g]
        remaining = tuple(replace(r, fused_execution_indices=coverage[g][r.execution_index])
                          for r in old.commands if r.execution_index not in removed[g])
        if not remaining:
            del lowered[g]
            continue
        changes = {'commands': remaining}
        if hasattr(old, 'fused_execution_indices'):
            changes['fused_execution_indices'] = tuple(sorted({i for r in remaining for i in r.fused_execution_indices}))
        lowered[g] = replace(old, **changes)
    lowered['qkv_rope_pack'] = SplitLowering(tuple(commands), tuple({c.digest: c for c in caps}.values()))
