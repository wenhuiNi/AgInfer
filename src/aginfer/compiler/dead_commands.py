"""Conservative output/state-rooted elimination after native fusion lowering."""
from collections import Counter
from dataclasses import replace

from ..lowering.assemble import order_command_placements
from ..lowering.command import OperandAccess as A
from ..lowering.memory import AllocationRegion as R
from ..packed_weights import _roots


# Only delivered pure tensor operations qualify. Unknown operations/providers,
# mutable writes and partial/in-place writes remain observable conservatively.
PURE_OPS = frozenset({"add", "mul", "cast", "linear", "gelu", "silu",
    "layer_norm", "rms_norm", "rope_default", "scaled_dot_product_attention",
    "reshape", "transpose", "slice", "broadcast_in_dim"})


def eliminate_dead_commands(schedule, memory, placements):
    """Return a coverage-preserving base plan and live command placements.

    Entry outputs and every state write are roots. Alias dependencies refer to
    storage roots, not debug names or physical arena offsets. No write is killed
    by a later write: mutable/partial operations are retained in source order.
    This also preserves standalone functions that export intermediate features.
    """
    records = order_command_placements(schedule, memory, placements)
    roots = _roots(memory)
    writes = [[roots[o.value_id] for o in p.command.operands if o.access != A.READ]
              for p in records]
    writers = Counter(v for row in writes for v in row)
    needed = {roots[v] for _, v in schedule.entry_outputs}
    kept, removed = [], []
    for p, outputs in reversed(tuple(zip(records, writes))):
        c = p.command
        pure = (c.provider_id in (1, 2, 4, 5) and (c.abi_major, c.abi_minor) == (1, 0)
                and all(schedule.ops[i].opcode in PURE_OPS for i in p.covered_execution_indices)
                and all(o.access != A.READ_WRITE and o.byte_offset == 0 for o in c.operands)
                and all(memory.allocations[v].region == R.ARENA and writers[v] == 1 for v in outputs))
        if pure and not needed.intersection(outputs):
            removed.append(p)
        else:
            kept.append(p)
            needed.update(roots[o.value_id] for o in c.operands if o.access != A.WRITE)
    elided = set(memory.elided_ops)
    elided.update(i for p in removed for i in p.covered_execution_indices)
    return replace(memory, elided_ops=tuple(sorted(elided))), tuple(reversed(kept))
