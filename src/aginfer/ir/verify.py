from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import TypeVar

from ..errors import ValidationError
from .model import (
    PROGRAM_IR_SCHEMA_MAJOR,
    PROGRAM_IR_SCHEMA_MINOR,
    DType,
    Function,
    Op,
    Program,
    Region,
    ShapeDomain,
    State,
    StateAccess,
    TensorType,
)

_SUPPORTED_OPS = frozenset(
    {
        "constant",
        "constant_ref",
        "broadcast_in_dim",
        "cast",
        "add",
        "mul",
        "logical_and",
        "cumulative_sum",
        "concat",
        "slice",
        "reduce_sum",
        "reduce_max",
        "gather",
        "rms_norm",
        "layer_norm",
        "softmax",
        "call",
        "rope",
        "rope_default",
        "scaled_dot_product_attention",
        "conv2d",
        "linear",
        "gelu",
        "matmul",
        "reshape",
        "transpose",
        "relu",
        "silu",
        "sinusoidal_embedding",
        "state_read",
        "state_write",
    }
)

_Named = TypeVar("_Named")


def verify_program(program: Program) -> None:
    """Fail closed unless ``program`` satisfies the ProgramIR v1 contract."""

    if not isinstance(program, Program):
        raise ValidationError("ProgramIR verifier requires a Program")
    if program.schema_major != PROGRAM_IR_SCHEMA_MAJOR or program.schema_minor > PROGRAM_IR_SCHEMA_MINOR:
        raise ValidationError(
            f"unsupported ProgramIR schema {program.schema_major}.{program.schema_minor}"
        )
    if not isinstance(program.shape_domain, ShapeDomain):
        raise ValidationError("ProgramIR shape_domain must be a ShapeDomain")
    if not all(isinstance(function, Function) for function in program.functions):
        raise ValidationError("ProgramIR functions must contain only Function records")
    if not all(isinstance(state, State) for state in program.states):
        raise ValidationError("ProgramIR states must contain only State records")
    functions = _unique_by_name(program.functions, "function")
    states = _unique_by_name(program.states, "state")
    if not functions:
        raise ValidationError("ProgramIR must contain at least one function")
    if program.entry not in functions:
        raise ValidationError(f"entry function is not defined: {program.entry}")

    declared_symbols = {dimension.symbol for dimension in program.shape_domain.dimensions}
    for state in program.states:
        _verify_tensor_type(state.type, declared_symbols, f"state {state.name}")
    signatures: dict[str, tuple[TensorType, ...]] = {}
    visiting: set[str] = set()

    def verify_function(name: str) -> tuple[TensorType, ...]:
        if name in signatures:
            return signatures[name]
        if name in visiting:
            raise ValidationError(f"ProgramIR call graph is recursive at function {name}")
        visiting.add(name)
        function = functions[name]
        assert isinstance(function, Function)
        signature = _verify_function(
            function,
            states,
            declared_symbols,
            functions,
            verify_function,
        )
        visiting.remove(name)
        signatures[name] = signature
        return signature

    for function in program.functions:
        verify_function(function.name)


def _verify_function(
    function: Function,
    states: dict[str, State],
    declared_symbols: set[str],
    functions: dict[str, Function],
    verify_function: Callable[[str], tuple[TensorType, ...]],
) -> tuple[TensorType, ...]:
    if not function.inputs:
        raise ValidationError(f"function {function.name} must have at least one input")
    if not function.outputs:
        raise ValidationError(f"function {function.name} must have at least one output")
    if not isinstance(function.body, Region) or not all(isinstance(op, Op) for op in function.body.ops):
        raise ValidationError(f"function {function.name} body must contain only Op records")
    values: dict[str, TensorType] = {}
    for value in function.inputs:
        _verify_tensor_type(value.type, declared_symbols, f"function {function.name} input {value.value_id}")
        if value.value_id in values:
            raise ValidationError(f"function {function.name} has duplicate input value {value.value_id}")
        values[value.value_id] = value.type

    for op_index, op in enumerate(function.body.ops):
        label = f"function {function.name} op {op_index} ({op.opcode})"
        if op.opcode not in _SUPPORTED_OPS:
            raise ValidationError(f"{label}: unsupported opcode")
        input_types: list[TensorType] = []
        for value_id in op.inputs:
            if value_id not in values:
                raise ValidationError(f"{label}: input {value_id} is not defined before use")
            input_types.append(values[value_id])
        output_ids: set[str] = set()
        for output in op.outputs:
            if output.value_id in values or output.value_id in output_ids:
                raise ValidationError(f"{label}: duplicate SSA definition {output.value_id}")
            _verify_tensor_type(output.type, declared_symbols, f"{label} output {output.value_id}")
            output_ids.add(output.value_id)

        if op.opcode == "call":
            _verify_call(
                op,
                tuple(input_types),
                functions,
                verify_function,
                label,
            )
        else:
            _verify_op(op, tuple(input_types), states, label)
        for output in op.outputs:
            values[output.value_id] = output.type

    seen_outputs: set[str] = set()
    output_types: list[TensorType] = []
    for value_id in function.outputs:
        if value_id not in values:
            raise ValidationError(f"function {function.name} output is undefined: {value_id}")
        if value_id in seen_outputs:
            raise ValidationError(f"function {function.name} repeats output {value_id}")
        seen_outputs.add(value_id)
        output_types.append(values[value_id])
    return tuple(output_types)


def _verify_op(
    op: Op,
    inputs: tuple[TensorType, ...],
    states: dict[str, State],
    label: str,
) -> None:
    if op.opcode == "constant":
        _signature(op, inputs=0, outputs=1, attributes={"value"}, label=label)
        output = op.outputs[0].type
        if output.static_numel is None:
            raise ValidationError(f"{label}: constant output must have a static shape")
        value = op.attribute("value")
        if not isinstance(value, tuple) or len(value) != output.static_numel:
            raise ValidationError(f"{label}: constant value must be a flat tuple matching output size")
        _verify_literal_values(value, output.dtype, label)
        return
    if op.opcode == "constant_ref":
        _signature(op, inputs=0, outputs=1, attributes={"namespace", "name"}, label=label)
        namespace = op.attribute("namespace")
        name = op.attribute("name")
        if not isinstance(namespace, str) or not namespace or not isinstance(name, str) or not name:
            raise ValidationError(f"{label}: constant_ref namespace and name must be non-empty strings")
        return
    if op.opcode == "cast":
        _signature(op, inputs=1, outputs=1, attributes={"dtype"}, label=label)
        raw_dtype = op.attribute("dtype")
        try:
            target_dtype = DType(raw_dtype)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label}: cast dtype is unknown") from exc
        if target_dtype == inputs[0].dtype:
            raise ValidationError(f"{label}: no-op casts are not canonical ProgramIR")
        expected = TensorType(target_dtype, inputs[0].shape, inputs[0].layout, inputs[0].device)
        if op.outputs[0].type != expected:
            raise ValidationError(f"{label}: cast output type does not match target dtype")
        return
    if op.opcode in {"add", "mul"}:
        _signature(op, inputs=2, outputs=1, attributes=set(), label=label)
        if inputs[0] != inputs[1] or op.outputs[0].type != inputs[0]:
            raise ValidationError(f"{label}: elementwise inputs and output must have identical types")
        return
    if op.opcode == "logical_and":
        _signature(op, inputs=2, outputs=1, attributes=set(), label=label)
        if (
            inputs[0].dtype != DType.BOOL
            or inputs[0] != inputs[1]
            or op.outputs[0].type != inputs[0]
        ):
            raise ValidationError(f"{label}: logical_and requires identical BOOL inputs and output")
        return
    if op.opcode == "cumulative_sum":
        _signature(op, inputs=1, outputs=1, attributes={"axis"}, label=label)
        _axis(op.attribute("axis"), inputs[0].rank, label)
        if inputs[0].dtype == DType.BOOL or op.outputs[0].type != inputs[0]:
            raise ValidationError(
                f"{label}: cumulative_sum requires an unchanged numeric tensor type"
            )
        return
    if op.opcode == "matmul":
        _signature(op, inputs=2, outputs=1, attributes=set(), label=label)
        left, right = inputs
        if left.rank != 2 or right.rank != 2 or not left.dtype.is_float or left.dtype != right.dtype:
            raise ValidationError(f"{label}: matmul requires rank-2 floating inputs with one dtype")
        if left.shape[1] != right.shape[0]:
            raise ValidationError(f"{label}: matmul contracting dimensions do not match")
        expected = TensorType(left.dtype, (left.shape[0], right.shape[1]), left.layout, left.device)
        if right.layout != left.layout or right.device != left.device or op.outputs[0].type != expected:
            raise ValidationError(f"{label}: matmul output/layout/device contract does not match")
        return
    if op.opcode == "concat":
        _signature(op, inputs=len(inputs), outputs=1, attributes={"axis"}, label=label)
        if not inputs:
            raise ValidationError(f"{label}: concat requires at least one input")
        axis = _axis(op.attribute("axis"), inputs[0].rank, label)
        first = inputs[0]
        axis_size = 0
        for tensor in inputs:
            if (
                tensor.dtype != first.dtype
                or tensor.rank != first.rank
                or tensor.layout != first.layout
                or tensor.device != first.device
            ):
                raise ValidationError(f"{label}: concat inputs must share dtype/rank/layout/device")
            for dimension, (left, right) in enumerate(zip(first.shape, tensor.shape)):
                if dimension != axis and left != right:
                    raise ValidationError(f"{label}: concat non-axis dimensions do not match")
            if not isinstance(tensor.shape[axis], int):
                raise ValidationError(f"{label}: v1 concat requires static concatenated dimensions")
            axis_size += tensor.shape[axis]
        expected_shape = list(first.shape)
        expected_shape[axis] = axis_size
        expected = TensorType(first.dtype, tuple(expected_shape), first.layout, first.device)
        if op.outputs[0].type != expected:
            raise ValidationError(f"{label}: concat output type is incorrect")
        return
    if op.opcode == "slice":
        _signature(
            op,
            inputs=1,
            outputs=1,
            attributes={"axis", "start", "stop"},
            label=label,
        )
        axis = _axis(op.attribute("axis"), inputs[0].rank, label)
        start = op.attribute("start")
        stop = op.attribute("stop")
        size = inputs[0].shape[axis]
        if not isinstance(size, int):
            raise ValidationError(f"{label}: v1 slice requires a static sliced dimension")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (start, stop)):
            raise ValidationError(f"{label}: slice start/stop must be integers")
        if not 0 <= start < stop <= size:
            raise ValidationError(f"{label}: slice range is empty or out of bounds")
        expected_shape = list(inputs[0].shape)
        expected_shape[axis] = stop - start
        expected = TensorType(inputs[0].dtype, tuple(expected_shape), inputs[0].layout, inputs[0].device)
        if op.outputs[0].type != expected:
            raise ValidationError(f"{label}: slice output type is incorrect")
        return
    if op.opcode in {"reduce_sum", "reduce_max"}:
        _signature(op, inputs=1, outputs=1, attributes={"axes", "keepdims"}, label=label)
        axes_value = op.attribute("axes")
        keepdims = op.attribute("keepdims")
        if not isinstance(axes_value, tuple) or not axes_value:
            raise ValidationError(f"{label}: reduction axes must be a non-empty tuple")
        if not isinstance(keepdims, bool):
            raise ValidationError(f"{label}: reduction keepdims must be boolean")
        axes = tuple(_axis(axis, inputs[0].rank, label) for axis in axes_value)
        if len(axes) != len(set(axes)):
            raise ValidationError(f"{label}: reduction axes contain duplicates")
        if keepdims:
            output_shape = tuple(1 if index in axes else size for index, size in enumerate(inputs[0].shape))
        else:
            output_shape = tuple(size for index, size in enumerate(inputs[0].shape) if index not in axes)
        expected = TensorType(inputs[0].dtype, output_shape, inputs[0].layout, inputs[0].device)
        if op.outputs[0].type != expected:
            raise ValidationError(f"{label}: reduction output type is incorrect")
        return
    if op.opcode == "gather":
        _signature(op, inputs=2, outputs=1, attributes=set(), label=label)
        table, indices = inputs
        if table.rank != 2 or not table.dtype.is_float or indices.dtype != DType.I32:
            raise ValidationError(f"{label}: gather requires a rank-2 floating table and I32 indices")
        if table.layout != indices.layout or table.device != indices.device:
            raise ValidationError(f"{label}: gather table and indices must share layout/device")
        expected = TensorType(
            table.dtype,
            indices.shape + (table.shape[1],),
            table.layout,
            table.device,
        )
        if op.outputs[0].type != expected:
            raise ValidationError(f"{label}: gather output type is incorrect")
        return
    if op.opcode == "rms_norm":
        _signature(op, inputs=2, outputs=1, attributes={"epsilon"}, label=label)
        _verify_norm_inputs(
            inputs[0],
            inputs[1],
            None,
            op.outputs[0].type,
            label,
            allow_f32_parameters=True,
        )
        _epsilon(op.attribute("epsilon"), label)
        return
    if op.opcode == "layer_norm":
        _signature(op, inputs=3, outputs=1, attributes={"epsilon"}, label=label)
        _verify_norm_inputs(inputs[0], inputs[1], inputs[2], op.outputs[0].type, label)
        _epsilon(op.attribute("epsilon"), label)
        return
    if op.opcode == "softmax":
        _signature(op, inputs=1, outputs=1, attributes={"axis"}, label=label)
        _axis(op.attribute("axis"), inputs[0].rank, label)
        if not inputs[0].dtype.is_float or op.outputs[0].type != inputs[0]:
            raise ValidationError(f"{label}: softmax requires an unchanged floating tensor type")
        return
    if op.opcode == "rope":
        _signature(op, inputs=4, outputs=1, attributes={"pairing"}, label=label)
        source, cosine, sine, positions = inputs
        pairing = op.attribute("pairing")
        if pairing not in {"interleaved", "split_half"}:
            raise ValidationError(f"{label}: RoPE pairing must be interleaved or split_half")
        if source.rank != 4 or not source.dtype.is_float:
            raise ValidationError(f"{label}: RoPE source must be floating [B,H,S,D]")
        head_dim = source.shape[3]
        if not isinstance(head_dim, int) or head_dim % 2 != 0:
            raise ValidationError(f"{label}: RoPE head dimension must be static and even")
        table_type = TensorType(
            source.dtype,
            (cosine.shape[0], head_dim // 2) if cosine.rank == 2 else (),
            source.layout,
            source.device,
        )
        if cosine.rank != 2 or cosine != table_type or sine != table_type:
            raise ValidationError(f"{label}: RoPE cos/sin tables must be [P,D/2] and match source")
        expected_positions = TensorType(
            DType.I32,
            (source.shape[0], source.shape[2]),
            source.layout,
            source.device,
        )
        if positions != expected_positions or op.outputs[0].type != source:
            raise ValidationError(f"{label}: RoPE positions or output type is incorrect")
        return
    if op.opcode == "rope_default":
        _signature(
            op,
            inputs=2,
            outputs=1,
            attributes={"pairing", "theta", "frequency_dtype"},
            label=label,
        )
        source, positions = inputs
        pairing = op.attribute("pairing")
        theta = op.attribute("theta")
        try:
            frequency_dtype = DType(op.attribute("frequency_dtype"))
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label}: RoPE frequency_dtype is unknown") from exc
        if not frequency_dtype.is_float:
            raise ValidationError(f"{label}: RoPE frequency_dtype must be floating")
        if pairing not in {"interleaved", "split_half"}:
            raise ValidationError(f"{label}: RoPE pairing must be interleaved or split_half")
        if (
            isinstance(theta, bool)
            or not isinstance(theta, (int, float))
            or not math.isfinite(float(theta))
            or float(theta) <= 0.0
        ):
            raise ValidationError(f"{label}: RoPE theta must be a positive finite number")
        if source.rank != 4 or not source.dtype.is_float:
            raise ValidationError(f"{label}: RoPE source must be floating [B,H,S,D]")
        head_dim = source.shape[3]
        if not isinstance(head_dim, int) or head_dim % 2 != 0:
            raise ValidationError(f"{label}: RoPE head dimension must be static and even")
        expected_positions = TensorType(
            DType.I32,
            (source.shape[0], source.shape[2]),
            source.layout,
            source.device,
        )
        if positions != expected_positions or op.outputs[0].type != source:
            raise ValidationError(f"{label}: RoPE positions or output type is incorrect")
        return
    if op.opcode == "scaled_dot_product_attention":
        _signature(
            op,
            inputs=4,
            outputs=1,
            attributes={"scale", "kv_group_size", "mask_fill"},
            label=label,
        )
        query, key, value, mask = inputs
        scale = op.attribute("scale")
        group_size = op.attribute("kv_group_size")
        if op.attribute("mask_fill") != "dtype_min":
            raise ValidationError(f"{label}: attention mask_fill must be dtype_min")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or scale <= 0
        ):
            raise ValidationError(f"{label}: attention scale must be a positive finite number")
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
            raise ValidationError(f"{label}: attention kv_group_size must be a positive integer")
        if any(tensor.rank != 4 for tensor in (query, key, value)):
            raise ValidationError(f"{label}: attention q/k/v must be rank-4")
        if not query.dtype.is_float or key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValidationError(f"{label}: attention q/k/v must share one floating dtype")
        if any(
            tensor.layout != query.layout or tensor.device != query.device
            for tensor in (key, value, mask)
        ):
            raise ValidationError(f"{label}: attention tensors must share layout/device")
        if (
            query.shape[0] != key.shape[0]
            or query.shape[0] != value.shape[0]
            or key.shape[1] != value.shape[1]
            or query.shape[3] != key.shape[3]
            or key.shape[2] != value.shape[2]
        ):
            raise ValidationError(f"{label}: attention q/k/v batch/head/sequence/dim contract differs")
        query_heads = query.shape[1]
        kv_heads = key.shape[1]
        if (
            not isinstance(query_heads, int)
            or not isinstance(kv_heads, int)
            or query_heads != kv_heads * group_size
        ):
            raise ValidationError(f"{label}: attention query heads must equal KV heads times kv_group_size")
        expected_mask = TensorType(
            DType.BOOL,
            (query.shape[0], query.shape[1], query.shape[2], key.shape[2]),
            query.layout,
            query.device,
        )
        expected_output = TensorType(
            query.dtype,
            (query.shape[0], query.shape[1], query.shape[2], value.shape[3]),
            query.layout,
            query.device,
        )
        if mask != expected_mask or op.outputs[0].type != expected_output:
            raise ValidationError(f"{label}: attention mask or output type is incorrect")
        return
    if op.opcode == "conv2d":
        _signature(op, inputs=3, outputs=1, attributes={"pads", "strides"}, label=label)
        source, weight, bias = inputs
        if source.rank != 4 or weight.rank != 4 or bias.rank != 1:
            raise ValidationError(f"{label}: conv2d requires NCHW source, OIHW weight, and O bias")
        if (
            not source.dtype.is_float
            or weight.dtype != source.dtype
            or bias.dtype != source.dtype
            or weight.layout != source.layout
            or bias.layout != source.layout
            or weight.device != source.device
            or bias.device != source.device
        ):
            raise ValidationError(f"{label}: conv2d tensors must share floating dtype/layout/device")
        if source.shape[1] != weight.shape[1] or bias.shape[0] != weight.shape[0]:
            raise ValidationError(f"{label}: conv2d channel or bias contract does not match")
        strides = _integer_tuple(op.attribute("strides"), 2, minimum=1, label=f"{label} strides")
        pads = _integer_tuple(op.attribute("pads"), 4, minimum=0, label=f"{label} pads")
        static_values = source.shape[1:] + weight.shape
        if not all(isinstance(value, int) for value in static_values):
            raise ValidationError(f"{label}: v1 conv2d requires static channel/spatial/kernel dimensions")
        _, height, width = source.shape[1:]
        output_channels, _, kernel_height, kernel_width = weight.shape
        padded_height = height + pads[0] + pads[2]
        padded_width = width + pads[1] + pads[3]
        if kernel_height > padded_height or kernel_width > padded_width:
            raise ValidationError(f"{label}: conv2d kernel exceeds the padded input")
        output_height = (padded_height - kernel_height) // strides[0] + 1
        output_width = (padded_width - kernel_width) // strides[1] + 1
        expected = TensorType(
            source.dtype,
            (source.shape[0], output_channels, output_height, output_width),
            source.layout,
            source.device,
        )
        if op.outputs[0].type != expected:
            raise ValidationError(f"{label}: conv2d output type is incorrect")
        return
    if op.opcode == "linear":
        _signature(op, inputs=3, outputs=1, attributes=set(), label=label)
        source, weight, bias = inputs
        if source.rank < 1 or weight.rank != 2 or bias.rank != 1 or not source.dtype.is_float:
            raise ValidationError(f"{label}: linear requires floating [...,I], [O,I], and [O]")
        if (
            weight.dtype != source.dtype
            or bias.dtype != source.dtype
            or weight.layout != source.layout
            or bias.layout != source.layout
            or weight.device != source.device
            or bias.device != source.device
        ):
            raise ValidationError(f"{label}: linear tensors must share dtype/layout/device")
        if source.shape[-1] != weight.shape[1] or bias.shape[0] != weight.shape[0]:
            raise ValidationError(f"{label}: linear in-feature or bias contract does not match")
        expected = TensorType(
            source.dtype,
            source.shape[:-1] + (weight.shape[0],),
            source.layout,
            source.device,
        )
        if op.outputs[0].type != expected:
            raise ValidationError(f"{label}: linear output type is incorrect")
        return
    if op.opcode == "reshape":
        _signature(op, inputs=1, outputs=1, attributes={"shape"}, label=label)
        shape = op.attribute("shape")
        if not isinstance(shape, tuple) or not shape or not all(
            isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
            for dimension in shape
        ):
            raise ValidationError(f"{label}: reshape shape must be a tuple of positive integers")
        source_numel = inputs[0].static_numel
        if source_numel is None:
            raise ValidationError(f"{label}: v1 reshape requires a static input shape")
        expected = TensorType(inputs[0].dtype, shape, inputs[0].layout, inputs[0].device)
        if expected.static_numel != source_numel or op.outputs[0].type != expected:
            raise ValidationError(f"{label}: reshape changes element count or declares the wrong output")
        return
    if op.opcode == "broadcast_in_dim":
        _signature(
            op,
            inputs=1,
            outputs=1,
            attributes={"shape", "broadcast_dimensions"},
            label=label,
        )
        shape = op.attribute("shape")
        dimensions = op.attribute("broadcast_dimensions")
        output = op.outputs[0].type
        if not isinstance(shape, tuple) or shape != output.shape:
            raise ValidationError(f"{label}: broadcast shape must exactly match the output shape")
        if (
            not isinstance(dimensions, tuple)
            or len(dimensions) != inputs[0].rank
            or any(not isinstance(axis, int) or isinstance(axis, bool) for axis in dimensions)
            or tuple(sorted(dimensions)) != dimensions
            or len(set(dimensions)) != len(dimensions)
            or any(axis < 0 or axis >= output.rank for axis in dimensions)
        ):
            raise ValidationError(f"{label}: broadcast_dimensions must map input axes in increasing order")
        if output.dtype != inputs[0].dtype or output.layout != inputs[0].layout or output.device != inputs[0].device:
            raise ValidationError(f"{label}: broadcast must preserve dtype/layout/device")
        for input_axis, output_axis in enumerate(dimensions):
            source_dimension = inputs[0].shape[input_axis]
            if source_dimension != 1 and source_dimension != output.shape[output_axis]:
                raise ValidationError(f"{label}: mapped broadcast dimensions are incompatible")
        return
    if op.opcode == "transpose":
        _signature(op, inputs=1, outputs=1, attributes={"permutation"}, label=label)
        permutation = op.attribute("permutation")
        if (
            not isinstance(permutation, tuple)
            or len(permutation) != inputs[0].rank
            or set(permutation) != set(range(inputs[0].rank))
        ):
            raise ValidationError(f"{label}: transpose permutation is invalid")
        expected = TensorType(
            inputs[0].dtype,
            tuple(inputs[0].shape[index] for index in permutation),
            inputs[0].layout,
            inputs[0].device,
        )
        if op.outputs[0].type != expected:
            raise ValidationError(f"{label}: transpose output type does not match its permutation")
        return
    if op.opcode in {"relu", "silu"}:
        _signature(op, inputs=1, outputs=1, attributes=set(), label=label)
        if not inputs[0].dtype.is_float or op.outputs[0].type != inputs[0]:
            raise ValidationError(f"{label}: activation requires an unchanged floating tensor type")
        return
    if op.opcode == "gelu":
        _signature(op, inputs=1, outputs=1, attributes={"approximation"}, label=label)
        if op.attribute("approximation") not in {"none", "tanh"}:
            raise ValidationError(f"{label}: GELU approximation must be none or tanh")
        if not inputs[0].dtype.is_float or op.outputs[0].type != inputs[0]:
            raise ValidationError(f"{label}: activation requires an unchanged floating tensor type")
        return
    if op.opcode == "sinusoidal_embedding":
        _signature(
            op,
            inputs=1,
            outputs=1,
            attributes={"dimension", "min_period", "max_period"},
            label=label,
        )
        dimension = op.attribute("dimension")
        minimum = op.attribute("min_period")
        maximum = op.attribute("max_period")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 or dimension % 2:
            raise ValidationError(f"{label}: sinusoidal dimension must be a positive even integer")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (minimum, maximum)
        ) or not 0 < minimum <= maximum:
            raise ValidationError(f"{label}: sinusoidal periods must satisfy 0 < min_period <= max_period")
        if inputs[0].rank != 1 or not inputs[0].dtype.is_float:
            raise ValidationError(f"{label}: sinusoidal input must be a rank-1 floating tensor")
        expected = TensorType(
            inputs[0].dtype,
            (inputs[0].shape[0], dimension),
            inputs[0].layout,
            inputs[0].device,
        )
        if op.outputs[0].type != expected:
            raise ValidationError(f"{label}: sinusoidal output type is incorrect")
        return
    if op.opcode == "state_read":
        _signature(op, inputs=0, outputs=1, attributes={"state"}, label=label)
        state = _state_attribute(op, states, label)
        if op.outputs[0].type != state.type:
            raise ValidationError(f"{label}: state_read output type does not match the state")
        return
    if op.opcode == "state_write":
        _signature(op, inputs=1, outputs=0, attributes={"state"}, label=label)
        state = _state_attribute(op, states, label)
        if state.access != StateAccess.READ_WRITE:
            raise ValidationError(f"{label}: cannot write read-only state {state.name}")
        if inputs[0] != state.type:
            raise ValidationError(f"{label}: state_write input type does not match the state")
        return
    raise AssertionError(f"unhandled verified opcode: {op.opcode}")


def _verify_call(
    op: Op,
    inputs: tuple[TensorType, ...],
    functions: dict[str, Function],
    verify_function: Callable[[str], tuple[TensorType, ...]],
    label: str,
) -> None:
    _signature(op, inputs=len(inputs), outputs=len(op.outputs), attributes={"callee", "repeat"}, label=label)
    callee = op.attribute("callee")
    repeat = op.attribute("repeat")
    if not isinstance(callee, str) or callee not in functions:
        raise ValidationError(f"{label}: call names an unknown callee")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
        raise ValidationError(f"{label}: call repeat must be a positive integer")
    callee_function = functions[callee]
    expected_inputs = tuple(value.type for value in callee_function.inputs)
    if inputs != expected_inputs:
        raise ValidationError(f"{label}: call input signature does not match callee {callee}")
    output_signature = verify_function(callee)
    if tuple(value.type for value in op.outputs) != output_signature:
        raise ValidationError(f"{label}: call output signature does not match callee {callee}")
    if repeat > 1 and expected_inputs != output_signature:
        raise ValidationError(f"{label}: repeated call callee signature is not iterable")


def _signature(
    op: Op,
    *,
    inputs: int,
    outputs: int,
    attributes: set[str],
    label: str,
) -> None:
    if len(op.inputs) != inputs or len(op.outputs) != outputs:
        raise ValidationError(f"{label}: expected {inputs} inputs and {outputs} outputs")
    actual_attributes = {name for name, _ in op.attributes}
    if actual_attributes != attributes:
        raise ValidationError(
            f"{label}: expected attributes {sorted(attributes)}, got {sorted(actual_attributes)}"
        )


def _state_attribute(op: Op, states: dict[str, State], label: str) -> State:
    name = op.attribute("state")
    if not isinstance(name, str) or name not in states:
        raise ValidationError(f"{label}: state attribute does not name a declared state")
    return states[name]


def _axis(value: object, rank: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= rank:
        raise ValidationError(f"{label}: axis must be a non-negative integer smaller than rank")
    return value


def _epsilon(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValidationError(f"{label}: epsilon must be a positive finite number")
    return float(value)


def _verify_norm_inputs(
    source: TensorType,
    weight: TensorType,
    bias: TensorType | None,
    output: TensorType,
    label: str,
    *,
    allow_f32_parameters: bool = False,
) -> None:
    if source.rank < 1 or not source.dtype.is_float:
        raise ValidationError(f"{label}: normalization source must be a rank-1+ floating tensor")
    allowed_dtypes = {source.dtype}
    if allow_f32_parameters and source.dtype in {DType.F16, DType.BF16}:
        allowed_dtypes.add(DType.F32)
    expected_shape = (source.shape[-1],)
    if (
        weight.dtype not in allowed_dtypes
        or weight.shape != expected_shape
        or weight.layout != source.layout
        or weight.device != source.device
        or (
            bias is not None
            and (
                bias.dtype != source.dtype
                or bias.shape != expected_shape
                or bias.layout != source.layout
                or bias.device != source.device
            )
        )
    ):
        raise ValidationError(f"{label}: normalization weight/bias must match the final dimension")
    if output != source:
        raise ValidationError(f"{label}: normalization output type must match its source")


def _integer_tuple(value: object, length: int, *, minimum: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != length
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= minimum
            for item in value
        )
    ):
        raise ValidationError(
            f"{label} must be a {length}-tuple of integers no smaller than {minimum}"
        )
    return value


def _verify_literal_values(values: tuple[object, ...], dtype: DType, label: str) -> None:
    if dtype == DType.BOOL:
        if not all(isinstance(value, bool) for value in values):
            raise ValidationError(f"{label}: bool constant contains a non-boolean")
        return
    if dtype == DType.I32:
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise ValidationError(f"{label}: i32 constant contains a non-integer")
        if not all(-(2**31) <= value < 2**31 for value in values):
            raise ValidationError(f"{label}: i32 constant is out of range")
        return
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        raise ValidationError(f"{label}: floating constant contains a non-number")


def _verify_tensor_type(tensor: object, declared_symbols: set[str], label: str) -> None:
    if not isinstance(tensor, TensorType):
        raise ValidationError(f"{label} does not have a TensorType")
    for dimension in tensor.shape:
        if isinstance(dimension, str) and dimension not in declared_symbols:
            raise ValidationError(f"{label} uses undeclared shape symbol {dimension}")


def _unique_by_name(items: Iterable[_Named], kind: str) -> dict[str, _Named]:
    result: dict[str, _Named] = {}
    for item in items:
        name = getattr(item, "name", None)
        if name in result:
            raise ValidationError(f"ProgramIR contains duplicate {kind} name {name}")
        result[name] = item
    return result
