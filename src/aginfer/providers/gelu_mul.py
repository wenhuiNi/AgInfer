"""BF16 GELU(tanh) -> BF16 round -> multiply, without intermediate storage."""
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
MAGIC = b'AIGMU1\0\0'
MAX_NUMEL = 1 << 26


@dataclass(frozen=True)
class GeluMulProblem:
    target_arch: CudaArch
    numel: int

    def __post_init__(self):
        if self.target_arch != CudaArch.SM120 or type(self.numel) is not int or not 1 <= self.numel <= MAX_NUMEL:
            raise ValidationError('GELU-multiply requires bounded contiguous BF16 on SM120')


@dataclass(frozen=True)
class GeluMulPayload:
    problem: GeluMulProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self):
        if (not isinstance(self.problem, GeluMulProblem) or type(self.module_bytes) is not int
                or not 0 < self.module_bytes < 2**64 or not isinstance(self.module_sha256, str)
                or len(self.module_sha256) != 64 or any(x not in '0123456789abcdef' for x in self.module_sha256)
                or set(self.module_sha256) == {'0'}):
            raise ValidationError('invalid GELU-multiply module identity')

    def to_bytes(self):
        return PAYLOAD.pack(MAGIC, 1, 0, int(self.problem.target_arch), self.problem.numel,
                            self.module_bytes, bytes.fromhex(self.module_sha256), bytes(64))

    @classmethod
    def from_bytes(cls, data):
        try:
            if len(data) != PAYLOAD.size:
                raise ValueError('size')
            f = PAYLOAD.unpack(data)
            result = cls(GeluMulProblem(CudaArch(f[3]), f[4]), f[5], f[6].hex())
            if result.to_bytes() != bytes(data):
                raise ValueError('noncanonical')
            return result
        except (ValueError, TypeError, ValidationError) as e:
            raise FormatError('invalid GELU-multiply payload') from e


@dataclass(frozen=True)
class FusedCommand:
    execution_index: int
    fused_execution_indices: tuple
    command: ProviderCommand


@dataclass(frozen=True)
class GeluMulLowering:
    commands: tuple
    capabilities: tuple


def lower_gelu_mul(schedule, module_bytes, module_sha256):
    consumers = {}
    for op in schedule.ops:
        for value in op.inputs:
            consumers.setdefault(value, []).append(op)
    exported = {v for _, v in schedule.entry_outputs}
    commands, capabilities, claimed = [], {}, set()
    for gelu in schedule.ops:
        if (gelu.opcode != 'gelu' or dict(gelu.attributes) != {'approximation': 'tanh'}
                or len(gelu.inputs) != 1 or len(gelu.outputs) != 1):
            continue
        intermediate = gelu.outputs[0]
        users = consumers.get(intermediate, ())
        if intermediate in exported or len(users) != 1:
            continue
        mul = users[0]
        if mul.execution_index in claimed:
            continue
        if mul.opcode != 'mul' or mul.attributes or len(mul.inputs) != 2 or len(mul.outputs) != 1:
            continue
        other = mul.inputs[1] if mul.inputs[0] == intermediate else mul.inputs[0]
        t = schedule.values[gelu.inputs[0]].type
        if (t.dtype != 'bf16' or t.device != 'cuda' or t.layout != 'row_major'
                or not 1 <= len(t.shape) <= 4 or any(type(d) is not int or d <= 0 for d in t.shape)
                or any(schedule.values[v].type != t for v in (intermediate, other, mul.outputs[0]))):
            continue
        try:
            problem = GeluMulProblem(CudaArch.SM120, math.prod(t.shape))
        except ValidationError:
            continue
        payload = GeluMulPayload(problem, module_bytes, module_sha256).to_bytes()
        cap = ProviderCapability(2, 1, 0, 'aginfer-aot-cuda=1', 'aginfer.gelu_tanh_mul.bf16.rounded.v1',
            hashlib.sha256(payload).hexdigest(), CudaArch.SM120, LoweringKind.AOT_CUDA,
            'gelu_mul', (t, t), (t,), (('approximation', 'tanh'), ('intermediate_round', 'bf16')), False, 0)
        command = ProviderCommand(CommandTag.CUDA_KERNEL, 2, 1, 0, cap.digest,
            (CommandOperand(gelu.inputs[0], OperandAccess.READ), CommandOperand(other, OperandAccess.READ),
             CommandOperand(mul.outputs[0], OperandAccess.WRITE)), payload, 0, 0, False)
        covered = tuple(sorted((gelu.execution_index, mul.execution_index)))
        commands.append(FusedCommand(covered[0], covered, command))
        claimed.update(covered)
        capabilities[cap.digest] = cap
    return GeluMulLowering(tuple(commands), tuple(capabilities[k] for k in sorted(capabilities)))
