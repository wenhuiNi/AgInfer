"""Paired bounded BF16 state writes; untouched prefix/suffix bytes are preserved."""
from dataclasses import dataclass
import hashlib
import math
import struct
from ..errors import FormatError, ValidationError
from ..schema import CudaArch
from ..lowering.capability import ProviderCapability
from ..lowering.inventory import LoweringKind
from ..lowering.command import CommandOperand, CommandTag, OperandAccess, ProviderCommand
from .projection_split import SplitCommand, SplitLowering

PAYLOAD = struct.Struct('<8sHHIQQQQ32s48s')


@dataclass(frozen=True)
class StateUpdateProblem:
    target_arch: CudaArch
    total: int
    count: int
    offset: int

    def __post_init__(self):
        if (self.target_arch != CudaArch.SM120 or any(type(x) is not int for x in (self.total, self.count, self.offset))
                or not 0 < self.count <= self.total <= 2**30 or self.offset < 0
                or self.offset + self.count > self.total or any(x % 8 for x in (self.total, self.count, self.offset))):
            raise ValidationError('state update outside vectorized BF16 bounds')


@dataclass(frozen=True)
class StateUpdatePayload:
    problem: StateUpdateProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self):
        if (not isinstance(self.problem, StateUpdateProblem) or type(self.module_bytes) is not int
                or not 0 < self.module_bytes < 2**64 or not isinstance(self.module_sha256, str)
                or len(self.module_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.module_sha256)
                or set(self.module_sha256) == {'0'}):
            raise ValidationError('invalid state update module identity')

    def to_bytes(self):
        p = self.problem
        return PAYLOAD.pack(b'AISTU1\0\0', 1, 0, int(p.target_arch), p.total, p.count, p.offset,
                            self.module_bytes, bytes.fromhex(self.module_sha256), bytes(48))

    @classmethod
    def from_bytes(cls, data):
        try:
            f = PAYLOAD.unpack(data)
            result = cls(StateUpdateProblem(CudaArch(f[3]), *f[4:7]), f[7], f[8].hex())
            if result.to_bytes() != bytes(data):
                raise ValueError('noncanonical')
            return result
        except (ValueError, TypeError, struct.error, ValidationError) as e:
            raise FormatError('invalid state update payload') from e


def lower_state_updates(schedule, module_bytes, module_sha256):
    states = dict(schedule.states)
    updates = [op for op in schedule.ops if op.opcode == 'state_update']
    if len(updates) % 2:
        raise ValidationError('paired state update requires complete pairs')
    commands, caps = [], {}
    for left, right in zip(updates[::2], updates[1::2]):
        if (left.invocation_id != right.invocation_id or any(op.opcode != 'state_read'
                for op in schedule.ops[left.execution_index + 1:right.execution_index])):
            raise ValidationError('state update fusion would cross another computation')
        targets, sources, problems = [], [], []
        for op in (left, right):
            attrs = dict(op.attributes)
            target = states[attrs['state']]
            source = op.inputs[0]
            t, s = schedule.values[target].type, schedule.values[source].type
            axis, start = attrs['axis'], attrs['start']
            if (t.dtype != 'bf16' or t.device != 'cuda' or t.layout != 'row_major'
                    or s.dtype != t.dtype or s.device != t.device or s.layout != t.layout
                    or math.prod(t.shape[:axis]) != 1):
                raise ValidationError('state update is not one contiguous BF16 segment')
            problems.append(StateUpdateProblem(CudaArch.SM120, math.prod(t.shape), math.prod(s.shape),
                                               start * math.prod(t.shape[axis + 1:])))
            targets.append(target)
            sources.append(source)
        if targets[0] == targets[1] or problems[0] != problems[1]:
            raise ValidationError('paired state update shape/target mismatch')
        payload = StateUpdatePayload(problems[0], module_bytes, module_sha256).to_bytes()
        cap = ProviderCapability(2, 1, 0, 'aginfer-aot-cuda=1', 'aginfer.state_update.bf16.pair.v1',
            hashlib.sha256(payload).hexdigest(), CudaArch.SM120, LoweringKind.MEMORY,
            'state_update', tuple(schedule.values[i].type for i in sources),
            tuple(schedule.values[i].type for i in targets),
            (('count', problems[0].count), ('offset', problems[0].offset)), False, 0)
        cmd = ProviderCommand(CommandTag.CUDA_KERNEL, 2, 1, 0, cap.digest,
            tuple(CommandOperand(i, OperandAccess.READ) for i in sources)
            + tuple(CommandOperand(i, OperandAccess.WRITE) for i in targets), payload)
        commands.append(SplitCommand(left.execution_index, (left.execution_index, right.execution_index), cmd))
        caps[cap.digest] = cap
    return SplitLowering(tuple(commands), tuple(caps[k] for k in sorted(caps)))
