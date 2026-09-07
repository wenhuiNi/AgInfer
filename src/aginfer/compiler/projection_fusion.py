"""Opt-in horizontal fusion of three small-row BF16 projections.

Weights are concatenated offline along output channels, as in merged/QKV
linear layers. Matching is by SSA/shape/constant semantics, never model names.
The initial envelope deliberately leaves large-row and nonzero-bias forms alone.
"""
from dataclasses import dataclass, replace
import hashlib
import math

from .identity import digest
from ..errors import ValidationError
from ..ir import DType, Device, Op, Region, TensorType, Value, attributes, verify_program

NAMESPACE = "aginfer_fused_projection"
POLICY = "three-shared-input-bf16-small-rows.v1"


@dataclass(frozen=True)
class FusionResult:
    program: object
    constants: dict
    groups: tuple

    def report(self):
        return {"policy": POLICY, "static_groups": list(self.groups), "constants": self.constants}


def fuse_projections(program):
    verify_program(program)
    constants, receipts, functions = {}, [], []
    for function in program.functions:
        ops = function.body.ops
        producers = {v.value_id: op for op in ops for v in op.outputs}
        types = {v.value_id: v.type for op in ops for v in op.outputs}
        types.update({v.value_id: v.type for v in function.inputs})
        if any(op.opcode == "constant_ref" and op.attribute("namespace") == NAMESPACE for op in ops):
            raise ValidationError("projection fusion must run once on the source IR")

        def zero(value_id):
            op = producers.get(value_id)
            if op is None:
                return False
            if op.opcode == "constant":
                return all(v == 0 and math.copysign(1, v) > 0 for v in op.attribute("value"))
            return op.opcode in ("broadcast_in_dim", "reshape") and zero(op.inputs[0])

        candidates = {}
        for index, op in enumerate(ops):
            if op.opcode == "linear":
                candidates.setdefault(op.inputs[0], []).append((index, op))
        insertions, removed = {}, set()
        for input_id, group in candidates.items():
            t = types[input_id]
            if (len(group) != 3 or t.dtype != DType.BF16 or t.device != Device.CUDA
                    or t.rank != 3 or t.shape[0] != 1 or type(t.shape[1]) is not int
                    or not 1 <= t.shape[1] <= 128 or type(t.shape[2]) is not int):
                continue
            projections = [op for _, op in group]
            widths = tuple(op.outputs[0].type.shape[-1] for op in projections)
            if any(type(w) is not int or w <= 0 or w % 8 for w in widths) or sum(widths) > 65536:
                continue
            weights = [producers.get(op.inputs[1]) for op in projections]
            if any(w is None or w.opcode != "constant_ref" for w in weights):
                continue
            if not all(zero(op.inputs[2]) for op in projections):
                continue
            parts = [{"namespace": w.attribute("namespace"), "name": w.attribute("name"),
                      "shape": list(w.outputs[0].type.shape)} for w in weights]
            weight_spec = {"dtype": "BF16", "shape": [sum(widths), t.shape[2]], "parts": parts}
            bias_spec = {"dtype": "BF16", "shape": [sum(widths)], "zero": True}
            names = [digest(weight_spec), digest(bias_spec)]
            for name, spec in zip(names, (weight_spec, bias_spec)):
                constants[name] = spec
            prefix = "__fused_projection_" + hashlib.sha256((function.name + input_id).encode()).hexdigest()[:24]
            ids = [prefix + suffix for suffix in (".weight", ".bias", ".output")]
            if any(v in types for v in ids):
                raise ValidationError("projection fusion generated SSA ID collision")
            merged_type = replace(t, shape=(1, t.shape[1], sum(widths)))
            merged = [Op("constant_ref", (), (Value(ids[0], replace(t, shape=tuple(weight_spec['shape']))),),
                         attributes(namespace=NAMESPACE, name=names[0])),
                      Op("constant_ref", (), (Value(ids[1], replace(t, shape=tuple(bias_spec['shape']))),),
                         attributes(namespace=NAMESPACE, name=names[1])),
                      Op("linear", (input_id, ids[0], ids[1]), (Value(ids[2], merged_type),))]
            start = 0
            for op, width in zip(projections, widths):
                merged.append(Op("slice", (ids[2],), op.outputs, attributes(axis=2, start=start, stop=start + width)))
                start += width
            insertions[group[0][0]] = merged
            removed.update(index for index, _ in group)
            receipts.append({"function": function.name, "input": input_id, "rows": t.shape[1],
                             "input_width": t.shape[2], "widths": list(widths), "weight": names[0]})
        result = []
        for i, op in enumerate(ops):
            result.extend(insertions.get(i, ()))
            if i not in removed:
                result.append(op)
        functions.append(replace(function, body=Region(tuple(result))))
    optimized = replace(program, functions=tuple(functions))
    verify_program(optimized)
    return FusionResult(optimized, constants, tuple(receipts))


@dataclass(frozen=True)
class DerivedTensor:
    dtype: str
    shape: tuple
    byte_length: int


class FusionConstantView:
    """Read-only packing view; never repacks at runtime or changes the checkpoint."""
    def __init__(self, source, constants):
        if NAMESPACE in source.namespaces:
            raise ValidationError("source reserves the projection fusion namespace")
        self.source, self.constants = source, constants

    def tensor(self, namespace, name):
        if namespace != NAMESPACE:
            return self.source.tensor(namespace, name)
        if name not in self.constants or digest(self.constants[name]) != name:
            raise ValidationError("derived projection constant identity mismatch")
        spec = self.constants[name]
        return DerivedTensor(spec['dtype'], tuple(spec['shape']), math.prod(spec['shape']) * 2)

    def iter_chunks(self, namespace, name, *, chunk_size):
        if namespace != NAMESPACE:
            yield from self.source.iter_chunks(namespace, name, chunk_size=chunk_size)
            return
        target = self.tensor(namespace, name)
        spec, written = self.constants[name], 0
        if spec.get('zero') is True:
            for offset in range(0, target.byte_length, chunk_size):
                yield bytes(min(chunk_size, target.byte_length - offset))
            return
        for part in spec['parts']:
            tensor = self.source.tensor(part['namespace'], part['name'])
            expected = math.prod(part['shape']) * 2
            if tensor.dtype != target.dtype or tuple(tensor.shape) != tuple(part['shape']) or tensor.byte_length != expected:
                raise ValidationError("projection source constant contract changed")
            part_bytes = 0
            for data in self.source.iter_chunks(part['namespace'], part['name'], chunk_size=chunk_size):
                part_bytes += len(data)
                yield data
            if part_bytes != expected:
                raise ValidationError("projection source constant is short")
            written += part_bytes
        if written != target.byte_length:
            raise ValidationError("projection concatenation byte count mismatch")
