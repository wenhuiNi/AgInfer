from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Mapping

from ..errors import ValidationError
from .model import DType, Device, Function, Op, Program, TensorType
from .verify import verify_program

Number = int | float


@dataclass(frozen=True)
class Tensor:
    dtype: DType
    shape: tuple[int, ...]
    data: tuple[Number, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise ValidationError("reference tensor dtype must be a ProgramIR DType")
        if not all(
            isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
            for dimension in self.shape
        ):
            raise ValidationError("reference tensor shape must contain positive integers")
        expected = math.prod(self.shape)
        if len(self.data) != expected:
            raise ValidationError(
                f"reference tensor has {len(self.data)} values but shape requires {expected}"
            )
        _validate_data(self.dtype, self.data)

    @staticmethod
    def from_values(dtype: DType, shape: tuple[int, ...], values: object) -> "Tensor":
        if not isinstance(values, (list, tuple)):
            raise ValidationError("reference tensor values must be a flat list or tuple")
        return _make_tensor(dtype, shape, tuple(values))


@dataclass(frozen=True)
class ExecutionResult:
    outputs: tuple[Tensor, ...]
    states: tuple[tuple[str, Tensor], ...]

    def state(self, name: str) -> Tensor:
        for state_name, value in self.states:
            if state_name == name:
                return value
        raise KeyError(name)


def execute(
    program: Program,
    inputs: Mapping[str, Tensor],
    *,
    states: Mapping[str, Tensor] | None = None,
    constants: Mapping[tuple[str, str], Tensor] | None = None,
    function: str | None = None,
) -> ExecutionResult:
    """Execute a verified small ProgramIR graph with a pure-Python CPU oracle."""

    verify_program(program)
    selected_name = program.entry if function is None else function
    selected = next((item for item in program.functions if item.name == selected_name), None)
    if selected is None:
        raise ValidationError(f"reference executor function is not defined: {selected_name}")
    expected_inputs = {value.value_id for value in selected.inputs}
    if set(inputs) != expected_inputs:
        raise ValidationError(
            f"reference inputs must be exactly {sorted(expected_inputs)}, got {sorted(inputs)}"
        )
    state_values = dict(states or {})
    expected_states = {state.name for state in program.states}
    if set(state_values) != expected_states:
        raise ValidationError(
            f"reference states must be exactly {sorted(expected_states)}, got {sorted(state_values)}"
        )

    symbols: dict[str, int] = {}
    entry_inputs: list[Tensor] = []
    for value in selected.inputs:
        tensor = inputs[value.value_id]
        _match_type(tensor, value.type, program, symbols, f"input {value.value_id}")
        entry_inputs.append(tensor)
    for state in program.states:
        _match_type(state_values[state.name], state.type, program, symbols, f"state {state.name}")

    output_values = _execute_function(
        program,
        selected,
        tuple(entry_inputs),
        state_values,
        symbols,
        constants or {},
    )
    return ExecutionResult(
        outputs=output_values,
        states=tuple((state.name, state_values[state.name]) for state in program.states),
    )


def _execute_function(
    program: Program,
    function: Function,
    inputs: tuple[Tensor, ...],
    states: dict[str, Tensor],
    symbols: dict[str, int],
    constants: Mapping[tuple[str, str], Tensor],
) -> tuple[Tensor, ...]:
    if len(inputs) != len(function.inputs):
        raise AssertionError("verified call has the wrong input arity")
    values: dict[str, Tensor] = {}
    for declared, tensor in zip(function.inputs, inputs):
        _match_type(tensor, declared.type, program, symbols, f"call input {declared.value_id}")
        values[declared.value_id] = tensor
    functions = {item.name: item for item in program.functions}
    for op in function.body.ops:
        if op.opcode != "call":
            _execute_op(op, values, states, program, symbols, constants)
            continue
        call_inputs = tuple(values[value_id] for value_id in op.inputs)
        callee = functions[str(op.attribute("callee"))]
        for _ in range(int(op.attribute("repeat"))):
            call_inputs = _execute_function(program, callee, call_inputs, states, symbols, constants)
        for output, tensor in zip(op.outputs, call_inputs):
            _store_value(output.value_id, output.type, tensor, values, program, symbols, "call")
    return tuple(values[value_id] for value_id in function.outputs)


def _execute_op(
    op: Op,
    values: dict[str, Tensor],
    states: dict[str, Tensor],
    program: Program,
    symbols: dict[str, int],
    constants: Mapping[tuple[str, str], Tensor],
) -> None:
    inputs = tuple(values[value_id] for value_id in op.inputs)
    if op.opcode == "constant":
        output_type = op.outputs[0].type
        shape = tuple(int(dimension) for dimension in output_type.shape)
        result = _make_tensor(output_type.dtype, shape, op.attribute("value"))
        _store(op, result, values, program, symbols)
        return
    if op.opcode == "constant_ref":
        key = (str(op.attribute("namespace")), str(op.attribute("name")))
        try:
            result = constants[key]
        except KeyError as exc:
            raise ValidationError(f"reference constant is unavailable: {key[0]}:{key[1]}") from exc
        _store(op, result, values, program, symbols)
        return
    if op.opcode == "cast":
        target_dtype = DType(op.attribute("dtype"))
        result = Tensor(
            target_dtype,
            inputs[0].shape,
            tuple(_cast_value(value, target_dtype) for value in inputs[0].data),
        )
        _store(op, result, values, program, symbols)
        return
    if op.opcode == "add":
        result = _make_tensor(
            inputs[0].dtype,
            inputs[0].shape,
            tuple(a + b for a, b in zip(inputs[0].data, inputs[1].data)),
        )
        _store(op, result, values, program, symbols)
        return
    if op.opcode == "mul":
        result = _make_tensor(
            inputs[0].dtype,
            inputs[0].shape,
            tuple(a * b for a, b in zip(inputs[0].data, inputs[1].data)),
        )
        _store(op, result, values, program, symbols)
        return
    if op.opcode == "logical_and":
        result = Tensor(
            DType.BOOL,
            inputs[0].shape,
            tuple(bool(a and b) for a, b in zip(inputs[0].data, inputs[1].data)),
        )
        _store(op, result, values, program, symbols)
        return
    if op.opcode == "cumulative_sum":
        source = inputs[0]
        axis = int(op.attribute("axis"))
        data: list[Number] = []
        strides = _contiguous_strides(source.shape)
        for flat_index in range(len(source.data)):
            coordinates = list(_coordinates(flat_index, source.shape))
            total: Number = 0 if source.dtype == DType.I32 else 0.0
            for axis_index in range(coordinates[axis] + 1):
                coordinates[axis] = axis_index
                source_index = sum(
                    coordinate * stride for coordinate, stride in zip(coordinates, strides)
                )
                total += source.data[source_index]
            data.append(total)
        _store(op, _make_tensor(source.dtype, source.shape, tuple(data)), values, program, symbols)
        return
    if op.opcode == "matmul":
        left, right = inputs
        rows, inner = left.shape
        _, columns = right.shape
        data: list[Number] = []
        for row in range(rows):
            for column in range(columns):
                data.append(
                    sum(
                        left.data[row * inner + offset] * right.data[offset * columns + column]
                        for offset in range(inner)
                    )
                )
        _store(op, _make_tensor(left.dtype, (rows, columns), tuple(data)), values, program, symbols)
        return
    if op.opcode == "concat":
        axis = int(op.attribute("axis"))
        outer = math.prod(inputs[0].shape[:axis])
        inner = math.prod(inputs[0].shape[axis + 1 :])
        data: list[Number] = []
        for outer_index in range(outer):
            for tensor in inputs:
                width = tensor.shape[axis] * inner
                start = outer_index * width
                data.extend(tensor.data[start : start + width])
        output_shape = list(inputs[0].shape)
        output_shape[axis] = sum(tensor.shape[axis] for tensor in inputs)
        _store(
            op,
            Tensor(inputs[0].dtype, tuple(output_shape), tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "slice":
        axis = int(op.attribute("axis"))
        start_index = int(op.attribute("start"))
        stop_index = int(op.attribute("stop"))
        outer = math.prod(inputs[0].shape[:axis])
        inner = math.prod(inputs[0].shape[axis + 1 :])
        source_width = inputs[0].shape[axis] * inner
        data: list[Number] = []
        for outer_index in range(outer):
            begin = outer_index * source_width + start_index * inner
            end = outer_index * source_width + stop_index * inner
            data.extend(inputs[0].data[begin:end])
        output_shape = list(inputs[0].shape)
        output_shape[axis] = stop_index - start_index
        _store(
            op,
            Tensor(inputs[0].dtype, tuple(output_shape), tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode in {"reduce_sum", "reduce_max"}:
        axes = set(int(axis) for axis in op.attribute("axes"))
        keepdims = bool(op.attribute("keepdims"))
        if keepdims:
            output_shape = tuple(1 if axis in axes else size for axis, size in enumerate(inputs[0].shape))
        else:
            output_shape = tuple(size for axis, size in enumerate(inputs[0].shape) if axis not in axes)
        output_size = math.prod(output_shape)
        reduced: list[Number | None]
        if op.opcode == "reduce_sum":
            zero: Number = 0 if inputs[0].dtype == DType.I32 else 0.0
            reduced = [zero] * output_size
        else:
            reduced = [None] * output_size
        output_strides = _contiguous_strides(output_shape)
        for flat_index, value in enumerate(inputs[0].data):
            coordinates = _coordinates(flat_index, inputs[0].shape)
            if keepdims:
                output_coordinates = tuple(0 if axis in axes else coordinate for axis, coordinate in enumerate(coordinates))
            else:
                output_coordinates = tuple(coordinate for axis, coordinate in enumerate(coordinates) if axis not in axes)
            output_index = sum(
                coordinate * stride for coordinate, stride in zip(output_coordinates, output_strides)
            )
            if op.opcode == "reduce_sum":
                reduced[output_index] = reduced[output_index] + value  # type: ignore[operator]
            elif reduced[output_index] is None or value > reduced[output_index]:
                reduced[output_index] = value
        _store(
            op,
            _make_tensor(inputs[0].dtype, output_shape, tuple(reduced)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "gather":
        table, indices = inputs
        rows, width = table.shape
        data: list[Number] = []
        for raw_index in indices.data:
            index = int(raw_index)
            if index < 0 or index >= rows:
                raise ValidationError(f"gather index {index} is outside [0, {rows})")
            data.extend(table.data[index * width : (index + 1) * width])
        _store(
            op,
            Tensor(table.dtype, indices.shape + (width,), tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode in {"rms_norm", "layer_norm"}:
        source = inputs[0]
        weight = inputs[1]
        bias = inputs[2] if op.opcode == "layer_norm" else None
        epsilon = float(op.attribute("epsilon"))
        width = source.shape[-1]
        rows = len(source.data) // width
        data: list[Number] = []
        for row in range(rows):
            row_values = tuple(float(value) for value in source.data[row * width : (row + 1) * width])
            if op.opcode == "rms_norm":
                mean = 0.0
                variance = sum(value * value for value in row_values) / width
            else:
                mean = sum(row_values) / width
                variance = sum((value - mean) ** 2 for value in row_values) / width
            inverse = 1.0 / math.sqrt(variance + epsilon)
            for column, value in enumerate(row_values):
                normalized = (value - mean) * inverse
                result = normalized * float(weight.data[column])
                if bias is not None:
                    result += float(bias.data[column])
                data.append(result)
        _store(
            op,
            _make_tensor(source.dtype, source.shape, tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "softmax":
        source = inputs[0]
        axis = int(op.attribute("axis"))
        outer = math.prod(source.shape[:axis])
        width = source.shape[axis]
        inner = math.prod(source.shape[axis + 1 :])
        data = [0.0] * len(source.data)
        for outer_index in range(outer):
            for inner_index in range(inner):
                indices = tuple(
                    (outer_index * width + column) * inner + inner_index
                    for column in range(width)
                )
                maximum = max(float(source.data[index]) for index in indices)
                exponentials = tuple(math.exp(float(source.data[index]) - maximum) for index in indices)
                denominator = sum(exponentials)
                for index, exponential in zip(indices, exponentials):
                    data[index] = exponential / denominator
        _store(
            op,
            _make_tensor(source.dtype, source.shape, tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "rope":
        source, cosine, sine, positions = inputs
        batch, heads, sequence, head_dim = source.shape
        half = head_dim // 2
        data = [0.0] * len(source.data)
        pairing = str(op.attribute("pairing"))
        for batch_index in range(batch):
            for sequence_index in range(sequence):
                position = int(positions.data[batch_index * sequence + sequence_index])
                if position < 0 or position >= cosine.shape[0]:
                    raise ValidationError(
                        f"RoPE position {position} is outside [0, {cosine.shape[0]})"
                    )
                for head_index in range(heads):
                    base = ((batch_index * heads + head_index) * sequence + sequence_index) * head_dim
                    for pair in range(half):
                        first = 2 * pair if pairing == "interleaved" else pair
                        second = first + 1 if pairing == "interleaved" else pair + half
                        cosine_value = float(cosine.data[position * half + pair])
                        sine_value = float(sine.data[position * half + pair])
                        left = float(source.data[base + first])
                        right = float(source.data[base + second])
                        data[base + first] = left * cosine_value - right * sine_value
                        data[base + second] = left * sine_value + right * cosine_value
        _store(
            op,
            _make_tensor(source.dtype, source.shape, tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "rope_default":
        source, positions = inputs
        batch, heads, sequence, head_dim = source.shape
        half = head_dim // 2
        theta = float(op.attribute("theta"))
        frequency_dtype = DType(op.attribute("frequency_dtype"))
        pairing = str(op.attribute("pairing"))
        frequencies = tuple(
            _round_float(theta ** (-(2.0 * pair) / head_dim), frequency_dtype)
            for pair in range(half)
        )
        data = [0.0] * len(source.data)
        for batch_index in range(batch):
            for head_index in range(heads):
                for sequence_index in range(sequence):
                    position = int(positions.data[batch_index * sequence + sequence_index])
                    base = (
                        ((batch_index * heads + head_index) * sequence + sequence_index)
                        * head_dim
                    )
                    for pair, frequency in enumerate(frequencies):
                        first = pair * 2 if pairing == "interleaved" else pair
                        second = first + 1 if pairing == "interleaved" else pair + half
                        angle = position * frequency
                        cosine = math.cos(angle)
                        sine = math.sin(angle)
                        left = float(source.data[base + first])
                        right = float(source.data[base + second])
                        data[base + first] = left * cosine - right * sine
                        data[base + second] = left * sine + right * cosine
        _store(
            op,
            _make_tensor(source.dtype, source.shape, tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "scaled_dot_product_attention":
        query, key, value, mask = inputs
        batch, heads, query_length, head_dim = query.shape
        key_length = key.shape[2]
        value_dim = value.shape[3]
        scale = float(op.attribute("scale"))
        group_size = int(op.attribute("kv_group_size"))
        kv_heads = key.shape[1]
        data: list[Number] = []
        for batch_index in range(batch):
            for head_index in range(heads):
                kv_head_index = head_index // group_size
                for query_index in range(query_length):
                    scores: list[tuple[int, float]] = []
                    for key_index in range(key_length):
                        mask_index = (
                            ((batch_index * heads + head_index) * query_length + query_index)
                            * key_length
                            + key_index
                        )
                        if not mask.data[mask_index]:
                            continue
                        query_base = (
                            ((batch_index * heads + head_index) * query_length + query_index)
                            * head_dim
                        )
                        key_base = (
                            ((batch_index * kv_heads + kv_head_index) * key_length + key_index)
                            * head_dim
                        )
                        score = sum(
                            float(query.data[query_base + dimension])
                            * float(key.data[key_base + dimension])
                            for dimension in range(head_dim)
                        ) * scale
                        scores.append((key_index, score))
                    if not scores:
                        # ``dtype_min`` is finite. An all-false row therefore has
                        # equal logits at every key and softmaxs to a uniform row.
                        scores = [(key_index, 0.0) for key_index in range(key_length)]
                    maximum = max(score for _, score in scores)
                    exponentials = tuple(math.exp(score - maximum) for _, score in scores)
                    denominator = sum(exponentials)
                    probabilities = tuple(value / denominator for value in exponentials)
                    for value_index in range(value_dim):
                        result = 0.0
                        for (key_index, _), probability in zip(scores, probabilities):
                            value_base = (
                                ((batch_index * kv_heads + kv_head_index) * key_length + key_index)
                                * value_dim
                            )
                            result += probability * float(value.data[value_base + value_index])
                        data.append(result)
        output_shape = (batch, heads, query_length, value_dim)
        _store(
            op,
            _make_tensor(query.dtype, output_shape, tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "conv2d":
        source, weight, bias = inputs
        batch, channels, height, width = source.shape
        output_channels, _, kernel_height, kernel_width = weight.shape
        stride_height, stride_width = tuple(int(value) for value in op.attribute("strides"))
        pad_top, pad_left, pad_bottom, pad_right = tuple(
            int(value) for value in op.attribute("pads")
        )
        output_height = (height + pad_top + pad_bottom - kernel_height) // stride_height + 1
        output_width = (width + pad_left + pad_right - kernel_width) // stride_width + 1
        data: list[Number] = []
        for batch_index in range(batch):
            for output_channel in range(output_channels):
                for output_y in range(output_height):
                    for output_x in range(output_width):
                        result = float(bias.data[output_channel])
                        for input_channel in range(channels):
                            for kernel_y in range(kernel_height):
                                input_y = output_y * stride_height + kernel_y - pad_top
                                if input_y < 0 or input_y >= height:
                                    continue
                                for kernel_x in range(kernel_width):
                                    input_x = output_x * stride_width + kernel_x - pad_left
                                    if input_x < 0 or input_x >= width:
                                        continue
                                    source_index = (
                                        ((batch_index * channels + input_channel) * height + input_y)
                                        * width
                                        + input_x
                                    )
                                    weight_index = (
                                        ((output_channel * channels + input_channel) * kernel_height + kernel_y)
                                        * kernel_width
                                        + kernel_x
                                    )
                                    result += float(source.data[source_index]) * float(
                                        weight.data[weight_index]
                                    )
                        data.append(result)
        output_shape = (batch, output_channels, output_height, output_width)
        _store(
            op,
            _make_tensor(source.dtype, output_shape, tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "linear":
        source, weight, bias = inputs
        input_width = source.shape[-1]
        output_width = weight.shape[0]
        rows = len(source.data) // input_width
        data: list[Number] = []
        for row in range(rows):
            for output_index in range(output_width):
                result = float(bias.data[output_index])
                result += sum(
                    float(source.data[row * input_width + input_index])
                    * float(weight.data[output_index * input_width + input_index])
                    for input_index in range(input_width)
                )
                data.append(result)
        output_shape = source.shape[:-1] + (output_width,)
        _store(
            op,
            _make_tensor(source.dtype, output_shape, tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "reshape":
        shape = tuple(int(dimension) for dimension in op.attribute("shape"))
        _store(op, Tensor(inputs[0].dtype, shape, inputs[0].data), values, program, symbols)
        return
    if op.opcode == "broadcast_in_dim":
        output_shape = _concrete_shape(op.outputs[0].type, symbols)
        dimensions = tuple(int(axis) for axis in op.attribute("broadcast_dimensions"))
        input_strides = _contiguous_strides(inputs[0].shape)
        data: list[Number] = []
        for flat_index in range(math.prod(output_shape)):
            output_coordinates = _coordinates(flat_index, output_shape)
            input_coordinates = tuple(
                0 if inputs[0].shape[input_axis] == 1 else output_coordinates[output_axis]
                for input_axis, output_axis in enumerate(dimensions)
            )
            input_index = sum(
                coordinate * stride for coordinate, stride in zip(input_coordinates, input_strides)
            )
            data.append(inputs[0].data[input_index])
        _store(op, Tensor(inputs[0].dtype, output_shape, tuple(data)), values, program, symbols)
        return
    if op.opcode == "transpose":
        permutation = tuple(int(index) for index in op.attribute("permutation"))
        output_shape = tuple(inputs[0].shape[index] for index in permutation)
        input_strides = _contiguous_strides(inputs[0].shape)
        data: list[Number] = []
        for flat_index in range(math.prod(output_shape)):
            output_coordinates = _coordinates(flat_index, output_shape)
            input_coordinates = [0] * len(permutation)
            for output_axis, input_axis in enumerate(permutation):
                input_coordinates[input_axis] = output_coordinates[output_axis]
            input_index = sum(
                coordinate * stride for coordinate, stride in zip(input_coordinates, input_strides)
            )
            data.append(inputs[0].data[input_index])
        _store(
            op,
            Tensor(inputs[0].dtype, output_shape, tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "relu":
        result = tuple(max(0.0, float(value)) for value in inputs[0].data)
        _store(op, _make_tensor(inputs[0].dtype, inputs[0].shape, result), values, program, symbols)
        return
    if op.opcode == "silu":
        result = tuple(_silu(float(value)) for value in inputs[0].data)
        _store(op, _make_tensor(inputs[0].dtype, inputs[0].shape, result), values, program, symbols)
        return
    if op.opcode == "gelu":
        if op.attribute("approximation") == "none":
            result = tuple(
                0.5 * float(value) * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
                for value in inputs[0].data
            )
        else:
            coefficient = math.sqrt(2.0 / math.pi)
            result = tuple(
                0.5
                * float(value)
                * (1.0 + math.tanh(coefficient * (float(value) + 0.044715 * float(value) ** 3)))
                for value in inputs[0].data
            )
        _store(op, _make_tensor(inputs[0].dtype, inputs[0].shape, result), values, program, symbols)
        return
    if op.opcode == "sinusoidal_embedding":
        dimension = int(op.attribute("dimension"))
        half = dimension // 2
        minimum = float(op.attribute("min_period"))
        maximum = float(op.attribute("max_period"))
        periods = tuple(
            minimum * (maximum / minimum) ** (index / (half - 1) if half > 1 else 0.0)
            for index in range(half)
        )
        data: list[Number] = []
        for value in inputs[0].data:
            angles = tuple(2.0 * math.pi * float(value) / period for period in periods)
            data.extend(math.sin(angle) for angle in angles)
            data.extend(math.cos(angle) for angle in angles)
        _store(
            op,
            _make_tensor(inputs[0].dtype, (inputs[0].shape[0], dimension), tuple(data)),
            values,
            program,
            symbols,
        )
        return
    if op.opcode == "state_read":
        _store(op, states[str(op.attribute("state"))], values, program, symbols)
        return
    if op.opcode == "state_write":
        states[str(op.attribute("state"))] = inputs[0]
        return
    raise AssertionError(f"verified opcode has no reference implementation: {op.opcode}")


def _store(
    op: Op,
    tensor: Tensor,
    values: dict[str, Tensor],
    program: Program,
    symbols: dict[str, int],
) -> None:
    output = op.outputs[0]
    _store_value(output.value_id, output.type, tensor, values, program, symbols, op.opcode)


def _store_value(
    value_id: str,
    expected: TensorType,
    tensor: Tensor,
    values: dict[str, Tensor],
    program: Program,
    symbols: dict[str, int],
    operation: str,
) -> None:
    _match_type(tensor, expected, program, symbols, f"op {operation} output {value_id}")
    values[value_id] = tensor


def _match_type(
    tensor: Tensor,
    expected: TensorType,
    program: Program,
    symbols: dict[str, int],
    label: str,
) -> None:
    if expected.device != Device.CPU:
        raise ValidationError(f"{label}: pure-Python executor only accepts CPU ProgramIR tensors")
    if tensor.dtype != expected.dtype or len(tensor.shape) != expected.rank:
        raise ValidationError(f"{label}: dtype or rank does not match ProgramIR")
    for actual, declared in zip(tensor.shape, expected.shape):
        if isinstance(declared, int):
            if actual != declared:
                raise ValidationError(f"{label}: static shape does not match ProgramIR")
            continue
        dimension_range = program.shape_domain.get(declared)
        if dimension_range is None:
            raise AssertionError(f"verified program lost shape symbol {declared}")
        if not dimension_range.minimum <= actual <= dimension_range.maximum:
            raise ValidationError(f"{label}: shape symbol {declared} is outside its domain")
        previous = symbols.setdefault(declared, actual)
        if previous != actual:
            raise ValidationError(f"{label}: shape symbol {declared} is inconsistent")


def _make_tensor(dtype: DType, shape: tuple[int, ...], values: object) -> Tensor:
    if not isinstance(values, tuple):
        raise ValidationError("reference op produced non-tuple data")
    if dtype in {DType.I32, DType.BOOL}:
        data = values
    else:
        data = tuple(float(value) for value in values)
    return Tensor(dtype, shape, data)


def _concrete_shape(tensor_type: TensorType, symbols: Mapping[str, int]) -> tuple[int, ...]:
    result: list[int] = []
    for dimension in tensor_type.shape:
        if isinstance(dimension, int):
            result.append(dimension)
            continue
        value = symbols.get(dimension)
        if value is None:
            raise ValidationError(f"reference executor has no concrete value for shape symbol {dimension}")
        result.append(value)
    return tuple(result)


def _validate_data(dtype: DType, values: tuple[Number, ...]) -> None:
    if dtype == DType.BOOL:
        if not all(isinstance(value, bool) for value in values):
            raise ValidationError("bool reference tensor contains invalid data")
        return
    if dtype == DType.I32:
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and -(2**31) <= value < 2**31
            for value in values
        ):
            raise ValidationError("i32 reference tensor contains invalid data")
        return
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        raise ValidationError("floating reference tensor contains invalid data")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    value = 1
    for dimension in reversed(shape):
        strides.append(value)
        value *= dimension
    return tuple(reversed(strides))


def _coordinates(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = index % shape[axis]
        index //= shape[axis]
    return tuple(result)


def _silu(value: float) -> float:
    if value >= 0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)


def _cast_value(value: Number, dtype: DType) -> Number:
    if dtype == DType.BOOL:
        return bool(value)
    if dtype == DType.I32:
        converted = int(value)
        if not -(2**31) <= converted < 2**31:
            raise ValidationError("explicit cast to i32 is out of range")
        return converted
    converted = float(value)
    if not math.isfinite(converted):
        raise ValidationError("explicit floating cast produced a non-finite value")
    return converted


def _round_float(value: float, dtype: DType) -> float:
    if dtype == DType.F32:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    if dtype == DType.F16:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    if dtype == DType.BF16:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        return struct.unpack("<f", struct.pack("<I", rounded & 0xFFFF0000))[0]
    raise AssertionError(f"non-floating generated frequency dtype: {dtype}")
