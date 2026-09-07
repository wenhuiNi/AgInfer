"""Make prefix-state + current-suffix concatenations resident and explicit.

The only changed arithmetic is none: all values are copied bit-for-bit.
Recognition is by exclusive state-read/concat dataflow, not checkpoint names.
"""
from dataclasses import dataclass, replace
from ..errors import ValidationError
from ..ir import Op, Region, attributes, verify_program, DType, Device


@dataclass(frozen=True)
class ResidentResult:
    program: object
    states: tuple

    def report(self):
        return {"policy": "resident-prefix-suffix.v1", "states": list(self.states)}


def make_kv_resident(program):
    verify_program(program)
    if any(op.opcode == 'state_update' for f in program.functions for op in f.body.ops):
        raise ValidationError('resident cache conversion must run once on source states')
    matches = {}
    for state in program.states:
        t = state.type
        if (t.dtype != DType.BF16 or t.device != Device.CUDA or t.rank != 4 or t.shape[:2] != (1, 1)
                or any(type(n) is not int for n in t.shape)):
            continue
        reads, writes, target = [], [], None
        valid = True
        for f in program.functions:
            consumers = {}
            for op in f.body.ops:
                for value in op.inputs:
                    consumers.setdefault(value, []).append(op)
            for index, op in enumerate(f.body.ops):
                if op.attribute('state') != state.name:
                    continue
                if op.opcode == 'state_write':
                    writes.append((f.name, index))
                    continue
                if op.opcode != 'state_read':
                    valid = False
                    continue
                value = op.outputs[0].value_id
                uses = consumers.get(value, [])
                if value in f.outputs or len(uses) != 1:
                    valid = False
                    continue
                concat = uses[0]
                if (concat.opcode != 'concat' or len(concat.inputs) != 2 or concat.inputs[0] != value
                        or concat.attribute('axis') != 2 or concat.outputs[0].value_id in f.outputs):
                    valid = False
                    continue
                uses = consumers.get(concat.outputs[0].value_id, [])
                # Do not let an old SSA snapshot escape as a mutable cache view.
                # A nested call may update the same cache before the consumer.
                if (len(uses) != 1 or uses[0].opcode not in {'scaled_dot_product_attention', 'add', 'mul', 'linear'}
                        or any(item.opcode == 'call' for item in f.body.ops)):
                    valid = False
                    continue
                output = concat.outputs[0].type
                if (target is not None and target != output) or output.shape[2] <= t.shape[2]:
                    valid = False
                target = output
                # Updating this state must not invalidate another read of it
                # within the same function invocation.
                reads.append((f.name, index, concat))
        if (valid and target is not None and reads and len(writes) == 1
                and len({name for name, _, _ in reads}) == len(reads)
                and writes[0][0] not in {name for name, _, _ in reads}):
            matches[state.name] = (target, reads, writes[0], t.shape[2])
    if not matches:
        raise ValidationError('no complete resident prefix/suffix state pattern')
    from ..lowering import build_execution_schedule
    source_schedule = build_execution_schedule(program)
    for name in matches:
        events = [op.opcode for op in source_schedule.ops if dict(op.attributes).get('state') == name]
        if not events or events[0] != 'state_write' or events.count('state_write') != 1:
            raise ValidationError('resident state requires one prefix refresh before all suffix uses per entry')
    replacements, removed, receipts = {}, set(), []
    for name, (target, reads, write, prefix) in matches.items():
        f = next(f for f in program.functions if f.name == write[0])
        op = f.body.ops[write[1]]
        replacements[write] = (Op('state_update', op.inputs, (), attributes(state=name, axis=2, start=0)),)
        for fname, index, concat in reads:
            f = next(f for f in program.functions if f.name == fname)
            concat_index = next(i for i, op in enumerate(f.body.ops) if op is concat)
            removed.add((fname, index))
            replacements[(fname, concat_index)] = (
                Op('state_update', (concat.inputs[1],), (), attributes(state=name, axis=2, start=prefix)),
                Op('state_read', (), concat.outputs, attributes(state=name)),
            )
        receipts.append({'state': name, 'prefix_length': prefix, 'total_length': target.shape[2],
                         'head_dim': target.shape[3]})
    functions = []
    for f in program.functions:
        ops = []
        for index, op in enumerate(f.body.ops):
            key = (f.name, index)
            if key not in removed:
                ops.extend(replacements.get(key, (op,)))
        functions.append(replace(f, body=Region(tuple(ops))))
    result = replace(program, functions=tuple(functions), states=tuple(
        replace(s, type=matches[s.name][0]) if s.name in matches else s for s in program.states))
    verify_program(result)
    return ResidentResult(result, tuple(receipts))
