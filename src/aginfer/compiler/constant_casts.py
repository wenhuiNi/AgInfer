"""Offline BF16 constant widening, sharing one packed F32 copy per source."""
from dataclasses import replace
import hashlib

from .identity import digest
from .. import providers as p
from ..errors import FormatError, ValidationError
from ..lowering.assemble import order_command_placements
from ..lowering.command import OperandAccess as A
from ..lowering.memory import (AllocationRegion as R, ValueAllocation,
    replan_memory_for_commands, dump_memory_plan)
from ..packed_weights import _roots


def fold_constant_casts(schedule, memory, commands, placements):
    # A previous constant evaluator may have removed some original placements.
    active = set(commands.commands)
    records = order_command_placements(schedule, memory,
        tuple(x for x in placements if x.command in active))
    roots = _roots(memory)
    exported = {roots[v] for _, v in schedule.entry_outputs}
    shared, derived, aliases, remaining, removed = {}, {}, {}, [], []
    for record in records:
        c = record.command
        eligible = c.provider_id == 2 and (c.abi_major, c.abi_minor) == (1, 0)
        try:
            payload = p.CudaKernelPayload.from_bytes(c.payload) if eligible else None
        except (FormatError, ValidationError, ValueError):
            payload = None
        if (payload is None or payload.problem.kernel_id != p.CudaKernelId.CAST_BF16_TO_F32
                or len(c.operands) != 2 or tuple(o.access for o in c.operands) != (A.READ, A.WRITE)
                or any(o.byte_offset for o in c.operands)):
            remaining.append(record); continue
        source, output = (roots[o.value_id] for o in c.operands)
        src, dst = schedule.values[source], schedule.values[output]
        if (memory.allocations[source].region != R.CONSTANT
                or memory.allocations[output].region != R.ARENA or output in exported
                or src.constant_identity is None or src.constant_identity.startswith('literal:')
                or src.type.dtype != 'bf16' or dst.type.dtype != 'f32'
                or src.type.shape != dst.type.shape
                or memory.allocations[source].byte_size != 2 * payload.problem.numel
                or memory.allocations[output].byte_size != 4 * payload.problem.numel):
            remaining.append(record); continue
        if source in shared:
            aliases[output] = shared[source]
        else:
            shared[source] = output
            derived[output] = source
        removed.append(record)
    if not removed:
        return memory, commands, {}, None
    allocations = []
    for a in memory.allocations:
        if a.value_id in derived:
            a = ValueAllocation(a.value_id, R.CONSTANT, a.byte_size)
        elif a.value_id in aliases:
            a = ValueAllocation(a.value_id, R.ALIAS, a.byte_size, alias_of=aliases[a.value_id])
        elif a.value_id in exported:
            a = ValueAllocation(a.value_id, R.ARENA, a.byte_size, 0, a.byte_size)
        allocations.append(a)
    elided = set(memory.elided_ops)
    elided.update(i for x in removed for i in x.covered_execution_indices)
    base = replace(memory, allocations=tuple(allocations), elided_ops=tuple(sorted(elided)),
        constant_bytes=memory.constant_bytes + sum(memory.allocations[v].byte_size for v in derived))
    result = replan_memory_for_commands(schedule, base, remaining)
    ordered = order_command_placements(schedule, result, remaining)
    stream = replace(commands, commands=tuple(x.command for x in ordered),
        memory_plan_sha256=hashlib.sha256(dump_memory_plan(result).encode()).hexdigest(),
        arena_bytes=result.arena_bytes, state_bytes=result.state_bytes)
    report = {'policy': 'constant-bf16-widen-share.v1', 'removed_commands': len(removed),
        'shared_aliases': len(aliases), 'derived_bytes': sum(memory.allocations[v].byte_size for v in derived),
        'values': [{'value_id': v, 'source_value_id': source,
                    'source': schedule.values[source].constant_identity} for v, source in sorted(derived.items())]}
    report['sha256'] = digest(report)
    return result, stream, derived, report


def validate_cast_report(report, plan):
    from ..executable import ExecutableValueRegion as V, ExecutableDType as T
    fields = {'policy', 'removed_commands', 'shared_aliases', 'derived_bytes', 'values', 'sha256'}
    if (not isinstance(report, dict) or set(report) != fields
            or report['policy'] != 'constant-bf16-widen-share.v1'
            or report['sha256'] != digest({k:v for k,v in report.items() if k != 'sha256'})
            or any(type(report[k]) is not int or report[k] < 0
                   for k in ('removed_commands','shared_aliases','derived_bytes'))
            or not isinstance(report['values'], list) or not report['values']
            or report['removed_commands'] != len(report['values']) + report['shared_aliases']):
        raise ValidationError('invalid constant cast receipt')
    seen, sources, total = set(), set(), 0
    for entry in report['values']:
        if (not isinstance(entry, dict) or set(entry) != {'value_id','source_value_id','source'}
                or any(type(entry[k]) is not int or not 0 <= entry[k] < len(plan.values)
                       for k in ('value_id','source_value_id'))
                or not isinstance(entry['source'], str) or ':' not in entry['source']):
            raise ValidationError('invalid constant cast value receipt')
        value, source = plan.values[entry['value_id']], plan.values[entry['source_value_id']]
        if (value.value_id in seen or source.value_id in sources or value.region != V.WEIGHTS
                or value.dtype != T.F32 or source.dtype != T.BF16 or value.shape != source.shape
                or value.byte_size != 2 * source.byte_size):
            raise ValidationError('constant cast receipt differs from packed values')
        seen.add(value.value_id); sources.add(source.value_id); total += value.byte_size
    if total != report['derived_bytes']:
        raise ValidationError('constant cast receipt byte count differs')
