"""A single launch for BF16(round(BF16 product) + residual), not FMA."""
from dataclasses import dataclass
import hashlib
import math
import struct

from ..errors import FormatError, ValidationError
from ..lowering.capability import ProviderCapability
from ..lowering.command import CommandOperand, CommandTag, OperandAccess, ProviderCommand
from ..lowering.inventory import LoweringKind
from ..schema import CudaArch

PAYLOAD = struct.Struct('<8sHHIQQ32s64s')
MAGIC = b'AIMAD1\0\0'


@dataclass(frozen=True)
class RoundedMulAddProblem:
    target_arch: CudaArch
    numel: int

    def __post_init__(self):
        if self.target_arch != CudaArch.SM120 or type(self.numel) is not int or not 1 <= self.numel <= 1 << 26:
            raise ValidationError('rounded multiply-add requires bounded contiguous BF16 on SM120')


@dataclass(frozen=True)
class RoundedMulAddPayload:
    problem: RoundedMulAddProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self):
        if (not isinstance(self.problem, RoundedMulAddProblem) or type(self.module_bytes) is not int
                or not 0 < self.module_bytes < 2**64 or not isinstance(self.module_sha256, str)
                or len(self.module_sha256) != 64 or any(x not in '0123456789abcdef' for x in self.module_sha256)
                or set(self.module_sha256) == {'0'}):
            raise ValidationError('invalid rounded multiply-add module identity')

    def to_bytes(self):
        return PAYLOAD.pack(MAGIC, 1, 0, int(self.problem.target_arch), self.problem.numel,
                            self.module_bytes, bytes.fromhex(self.module_sha256), bytes(64))

    @classmethod
    def from_bytes(cls, data):
        try:
            if len(data) != PAYLOAD.size:
                raise ValueError('size')
            f = PAYLOAD.unpack(data)
            result = cls(RoundedMulAddProblem(CudaArch(f[3]), f[4]), f[5], f[6].hex())
            if result.to_bytes() != bytes(data):
                raise ValueError('noncanonical')
            return result
        except (ValueError, TypeError, ValidationError) as e:
            raise FormatError('invalid rounded multiply-add payload') from e


@dataclass(frozen=True)
class RoundedMulAddCommand:
    execution_index: int
    fused_execution_indices: tuple
    command: ProviderCommand


@dataclass(frozen=True)
class RoundedMulAddLowering:
    commands: tuple
    capabilities: tuple


def lower_rounded_mul_add(schedule, module_bytes, module_sha256, *, excluded=()):
    consumers = {}
    for op in schedule.ops:
        for v in op.inputs:
            consumers.setdefault(v, []).append(op)
    exported = {v for _, v in schedule.entry_outputs}
    claimed = set(excluded)
    commands, capabilities = [], {}
    for mul in schedule.ops:
        if mul.execution_index in claimed or mul.opcode != 'mul' or mul.attributes or len(mul.outputs) != 1:
            continue
        mid = mul.outputs[0]
        users = consumers.get(mid, ())
        if mid in exported or len(users) != 1:
            continue
        add = users[0]
        if (add.execution_index in claimed or add.opcode != 'add' or add.attributes
                or len(add.inputs) != 2 or len(add.outputs) != 1):
            continue
        residual = add.inputs[1] if add.inputs[0] == mid else add.inputs[0]
        t = schedule.values[mid].type
        if (len(mul.inputs) != 2 or t.dtype != 'bf16' or t.device != 'cuda' or t.layout != 'row_major'
                or not 1 <= len(t.shape) <= 4 or any(type(d) is not int or d <= 0 for d in t.shape)
                or any(schedule.values[v].type != t for v in (*mul.inputs, residual, *add.outputs))):
            continue
        try:
            problem = RoundedMulAddProblem(CudaArch.SM120, math.prod(t.shape))
        except ValidationError:
            continue
        payload = RoundedMulAddPayload(problem, module_bytes, module_sha256).to_bytes()
        cap = ProviderCapability(2, 1, 0, 'aginfer-aot-cuda=1', 'aginfer.mul_add.bf16.rounded.v1',
            hashlib.sha256(payload).hexdigest(), CudaArch.SM120, LoweringKind.AOT_CUDA,
            'mul_add', (t, t, t), (t,), (('intermediate_round', 'bf16'),), False, 0)
        command = ProviderCommand(CommandTag.CUDA_KERNEL, 2, 1, 0, cap.digest,
            tuple(CommandOperand(v, OperandAccess.READ) for v in (*mul.inputs, residual)) +
            (CommandOperand(add.outputs[0], OperandAccess.WRITE),), payload, 0, 0, False)
        covered = tuple(sorted((mul.execution_index, add.execution_index)))
        commands.append(RoundedMulAddCommand(covered[0], covered, command))
        claimed.update(covered); capabilities[cap.digest] = cap
    return RoundedMulAddLowering(tuple(commands), tuple(capabilities[k] for k in sorted(capabilities)))
