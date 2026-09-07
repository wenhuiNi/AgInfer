"""One vectorized copy kernel splits a packed BF16 projection into three tensors."""
from dataclasses import dataclass
import hashlib
import struct

from ..errors import FormatError, ValidationError
from ..lowering.capability import ProviderCapability
from ..lowering.command import CommandOperand, CommandTag, OperandAccess, ProviderCommand
from ..lowering.inventory import LoweringKind, TensorSignature
from ..schema import CudaArch

PAYLOAD = struct.Struct('<8sHHII3IQ32s56s')
MAGIC = b'AIPSP1\0\0'


@dataclass(frozen=True)
class ProjectionSplitProblem:
    target_arch: CudaArch
    rows: int
    widths: tuple[int, int, int]

    def __post_init__(self):
        if (self.target_arch != CudaArch.SM120 or type(self.rows) is not int or not 1 <= self.rows <= 128
                or not isinstance(self.widths, tuple) or len(self.widths) != 3
                or any(type(w) is not int or w <= 0 or w % 8 for w in self.widths) or sum(self.widths) > 65536):
            raise ValidationError('projection split is outside its BF16 vectorized envelope')


@dataclass(frozen=True)
class ProjectionSplitPayload:
    problem: ProjectionSplitProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self):
        if (not isinstance(self.problem, ProjectionSplitProblem) or type(self.module_bytes) is not int
                or not 0 < self.module_bytes < 2**64 or not isinstance(self.module_sha256, str)
                or len(self.module_sha256) != 64 or any(x not in '0123456789abcdef' for x in self.module_sha256)
                or set(self.module_sha256) == {'0'}):
            raise ValidationError('invalid projection split module identity')

    def to_bytes(self):
        return PAYLOAD.pack(MAGIC, 1, 0, int(self.problem.target_arch), self.problem.rows,
                            *self.problem.widths, self.module_bytes, bytes.fromhex(self.module_sha256), bytes(56))

    @classmethod
    def from_bytes(cls, data):
        try:
            if len(data) != PAYLOAD.size:
                raise ValueError('size')
            f = PAYLOAD.unpack(data)
            result = cls(ProjectionSplitProblem(CudaArch(f[3]), f[4], tuple(f[5:8])), f[8], f[9].hex())
            if result.to_bytes() != bytes(data):
                raise ValueError('noncanonical')
            return result
        except (ValueError, TypeError, ValidationError) as e:
            raise FormatError('invalid projection split payload') from e


@dataclass(frozen=True)
class SplitCommand:
    execution_index: int
    fused_execution_indices: tuple
    command: ProviderCommand


@dataclass(frozen=True)
class SplitLowering:
    commands: tuple
    capabilities: tuple


def lower_projection_splits(schedule, module_bytes, module_sha256):
    consumers = {}
    for op in schedule.ops:
        for v in op.inputs:
            consumers.setdefault(v, []).append(op)
    exported = {v for _, v in schedule.entry_outputs}
    commands, capabilities = [], {}
    for value_id, group in consumers.items():
        if value_id in exported or len(group) != 3 or any(op.opcode != 'slice' for op in group):
            continue
        source = schedule.values[value_id].type
        if (source.dtype != 'bf16' or source.device != 'cuda' or len(source.shape) != 3
                or source.shape[0] != 1 or source.layout != 'row_major'):
            continue
        if any(dict(op.attributes).get('axis') != 2 for op in group):
            continue
        group = sorted(group, key=lambda op: dict(op.attributes)['start'])
        widths = tuple(dict(op.attributes)['stop'] - dict(op.attributes)['start'] for op in group)
        start = 0
        valid = True
        for op, width in zip(group, widths):
            attrs = dict(op.attributes)
            expected = TensorSignature('bf16', (1, source.shape[1], width), 'row_major', 'cuda')
            valid &= (attrs == {'axis': 2, 'start': start, 'stop': start + width}
                      and len(op.outputs) == 1 and schedule.values[op.outputs[0]].type == expected)
            start += width
        if not valid or start != source.shape[2]:
            continue
        try:
            problem = ProjectionSplitProblem(CudaArch.SM120, source.shape[1], widths)
        except ValidationError:
            continue
        payload = ProjectionSplitPayload(problem, module_bytes, module_sha256).to_bytes()
        output_types = tuple(schedule.values[op.outputs[0]].type for op in group)
        cap = ProviderCapability(2, 1, 0, 'aginfer-aot-cuda=1', 'aginfer.projection_split.bf16.v1',
            hashlib.sha256(payload).hexdigest(), CudaArch.SM120, LoweringKind.MEMORY,
            'split_projection', (source,), output_types, (('axis', 2), ('widths', widths)), False, 0)
        # A candidate capability is not a manufactured capture/numerical receipt.
        command = ProviderCommand(CommandTag.CUDA_KERNEL, 2, 1, 0, cap.digest,
            (CommandOperand(value_id, OperandAccess.READ),) + tuple(CommandOperand(op.outputs[0], OperandAccess.WRITE) for op in group),
            payload, 0, 0, False)
        covered = tuple(sorted(op.execution_index for op in group))
        commands.append(SplitCommand(covered[0], covered, command))
        capabilities[cap.digest] = cap
    return SplitLowering(tuple(commands), tuple(capabilities[k] for k in sorted(capabilities)))
