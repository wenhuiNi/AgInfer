from __future__ import annotations

import math
import unittest

from aginfer.errors import ValidationError
from aginfer.ir import (
    DType,
    DimensionRange,
    Function,
    Op,
    Program,
    Region,
    ShapeDomain,
    State,
    StateAccess,
    Tensor,
    TensorType,
    Value,
    attributes,
    dump_program,
    execute,
)
from tests.ir_fixtures import gated_mlp_program, vision_projection_program


class ReferenceExecutorTests(unittest.TestCase):
    def test_constant_ref_uses_explicit_reference_inputs(self) -> None:
        tensor_type = TensorType(DType.F32, (2,))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("trigger", tensor_type),),
                    ("weight",),
                    Region(
                        (
                            Op(
                                "constant_ref",
                                (),
                                (Value("weight", tensor_type),),
                                attributes(namespace="model", name="weight"),
                            ),
                        )
                    ),
                ),
            ),
        )
        constant = Tensor.from_values(DType.F32, (2,), [1.5, -2.0])
        self.assertEqual(
            execute(program, {"trigger": constant}, constants={("model", "weight"): constant}).outputs,
            (constant,),
        )
        with self.assertRaisesRegex(ValidationError, "reference constant is unavailable"):
            execute(program, {"trigger": constant})

    def test_explicit_broadcast_sinusoidal_embedding_and_tanh_gelu(self) -> None:
        source = TensorType(DType.F32, (2, 2))
        broadcast = TensorType(DType.F32, (2, 3, 2))
        timestep = TensorType(DType.F32, (2,))
        embedding = TensorType(DType.F32, (2, 4))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", source), Value("time", timestep)),
                    ("expanded", "time_embedding", "activated"),
                    Region(
                        (
                            Op(
                                "broadcast_in_dim",
                                ("x",),
                                (Value("expanded", broadcast),),
                                attributes(shape=(2, 3, 2), broadcast_dimensions=(0, 2)),
                            ),
                            Op(
                                "sinusoidal_embedding",
                                ("time",),
                                (Value("time_embedding", embedding),),
                                attributes(dimension=4, min_period=1.0, max_period=2.0),
                            ),
                            Op(
                                "gelu",
                                ("time_embedding",),
                                (Value("activated", embedding),),
                                attributes(approximation="tanh"),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {
                "x": Tensor.from_values(DType.F32, (2, 2), (1, 2, 3, 4)),
                "time": Tensor.from_values(DType.F32, (2,), (0, 1)),
            },
        )
        self.assertEqual(result.outputs[0].data, (1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4))
        expected_embedding = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0)
        for actual, expected in zip(result.outputs[1].data, expected_embedding):
            self.assertAlmostEqual(actual, expected, places=7)
        coefficient = math.sqrt(2.0 / math.pi)
        expected_gelu = tuple(
            0.5 * value * (1.0 + math.tanh(coefficient * (value + 0.044715 * value**3)))
            for value in expected_embedding
        )
        for actual, expected in zip(result.outputs[2].data, expected_gelu):
            self.assertAlmostEqual(actual, expected, places=7)

    def test_gated_mlp_matches_direct_math_and_is_repeatable(self) -> None:
        program = gated_mlp_program()
        input_tensor = Tensor.from_values(DType.F32, (2, 2), (1.0, 2.0, 3.0, 4.0))
        first = execute(program, {"x": input_tensor})
        second = execute(program, {"x": input_tensor})
        expected = (
            _silu(1.0) * 3.0,
            _silu(2.0) * -1.0,
            _silu(3.0) * 7.0,
            _silu(4.0) * -1.0,
        )
        self.assertEqual(first, second)
        for actual, wanted in zip(first.outputs[0].data, expected):
            self.assertAlmostEqual(actual, wanted, places=7)

    def test_vision_projection_matches_expected_whole_graph(self) -> None:
        pixels = Tensor.from_values(DType.F32, (2, 3), (1.0, -2.0, 3.0, 0.5, 1.0, -1.0))
        result = execute(vision_projection_program(), {"pixels": pixels})
        self.assertEqual(result.outputs[0].shape, (1, 4))
        self.assertEqual(result.outputs[0].data, (4.0, 1.0, 0.0, 0.0))

    def test_transpose_uses_row_major_logical_permutation(self) -> None:
        source = TensorType(DType.F32, (2, 3))
        transposed = TensorType(DType.F32, (3, 2))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("input", source),),
                    ("output",),
                    Region(
                        (
                            Op(
                                "transpose",
                                ("input",),
                                (Value("output", transposed),),
                                attributes(permutation=(1, 0)),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {"input": Tensor.from_values(DType.F32, (2, 3), (1, 2, 3, 4, 5, 6))},
        )
        self.assertEqual(result.outputs[0].data, (1.0, 4.0, 2.0, 5.0, 3.0, 6.0))

    def test_i32_inputs_do_not_silently_truncate_floats(self) -> None:
        with self.assertRaisesRegex(ValidationError, "i32 reference tensor"):
            Tensor.from_values(DType.I32, (2,), (1.5, 2.0))

    def test_cast_concat_slice_and_reductions_match_hand_calculation(self) -> None:
        source = TensorType(DType.F32, (2, 3))
        casted = TensorType(DType.F16, (2, 3))
        sliced = TensorType(DType.F16, (2, 2))
        concatenated = TensorType(DType.F16, (4, 2))
        column_sum = TensorType(DType.F16, (2,))
        row_max = TensorType(DType.F16, (4, 1))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("input", source),),
                    ("column_sum", "row_max"),
                    Region(
                        (
                            Op(
                                "cast",
                                ("input",),
                                (Value("casted", casted),),
                                attributes(dtype="f16"),
                            ),
                            Op(
                                "slice",
                                ("casted",),
                                (Value("sliced", sliced),),
                                attributes(axis=1, start=1, stop=3),
                            ),
                            Op(
                                "concat",
                                ("sliced", "sliced"),
                                (Value("concatenated", concatenated),),
                                attributes(axis=0),
                            ),
                            Op(
                                "reduce_sum",
                                ("concatenated",),
                                (Value("column_sum", column_sum),),
                                attributes(axes=(0,), keepdims=False),
                            ),
                            Op(
                                "reduce_max",
                                ("concatenated",),
                                (Value("row_max", row_max),),
                                attributes(axes=(1,), keepdims=True),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {"input": Tensor.from_values(DType.F32, (2, 3), (1, 2, 3, 4, 5, 6))},
        )
        self.assertEqual(result.outputs[0].dtype, DType.F16)
        self.assertEqual(result.outputs[0].data, (14.0, 18.0))
        self.assertEqual(result.outputs[1].data, (3.0, 6.0, 3.0, 6.0))

    def test_multi_axis_reduction_can_produce_rank_zero_tensor(self) -> None:
        matrix = TensorType(DType.F32, (2, 2))
        scalar = TensorType(DType.F32, ())
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("input", matrix),),
                    ("sum",),
                    Region(
                        (
                            Op(
                                "reduce_sum",
                                ("input",),
                                (Value("sum", scalar),),
                                attributes(axes=(0, 1), keepdims=False),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {"input": Tensor.from_values(DType.F32, (2, 2), (1, 2, 3, 4))},
        )
        self.assertEqual(result.outputs[0].shape, ())
        self.assertEqual(result.outputs[0].data, (10.0,))

    def test_gather_uses_i32_indices_and_rejects_out_of_range_rows(self) -> None:
        table_type = TensorType(DType.F32, (3, 2))
        indices_type = TensorType(DType.I32, (2,))
        output_type = TensorType(DType.F32, (2, 2))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("table", table_type), Value("indices", indices_type)),
                    ("selected",),
                    Region((Op("gather", ("table", "indices"), (Value("selected", output_type),)),)),
                ),
            ),
        )
        table = Tensor.from_values(DType.F32, (3, 2), (10, 11, 20, 21, 30, 31))
        result = execute(
            program,
            {"table": table, "indices": Tensor.from_values(DType.I32, (2,), (2, 0))},
        )
        self.assertEqual(result.outputs[0].data, (30.0, 31.0, 10.0, 11.0))
        with self.assertRaisesRegex(ValidationError, "outside"):
            execute(
                program,
                {"table": table, "indices": Tensor.from_values(DType.I32, (2,), (-1, 0))},
            )

    def test_norms_and_softmax_match_independent_formulas(self) -> None:
        source = TensorType(DType.F32, (2, 3))
        parameter = TensorType(DType.F32, (3,))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", source), Value("weight", parameter), Value("bias", parameter)),
                    ("rms", "layer", "probabilities"),
                    Region(
                        (
                            Op(
                                "rms_norm",
                                ("x", "weight"),
                                (Value("rms", source),),
                                attributes(epsilon=1e-5),
                            ),
                            Op(
                                "layer_norm",
                                ("x", "weight", "bias"),
                                (Value("layer", source),),
                                attributes(epsilon=1e-5),
                            ),
                            Op(
                                "softmax",
                                ("x",),
                                (Value("probabilities", source),),
                                attributes(axis=1),
                            ),
                        )
                    ),
                ),
            ),
        )
        values = (1.0, 2.0, 3.0, -1.0, 0.0, 1.0)
        weights = (1.0, 2.0, 0.5)
        biases = (0.5, -0.5, 1.0)
        result = execute(
            program,
            {
                "x": Tensor.from_values(DType.F32, (2, 3), values),
                "weight": Tensor.from_values(DType.F32, (3,), weights),
                "bias": Tensor.from_values(DType.F32, (3,), biases),
            },
        )
        expected_rms: list[float] = []
        expected_layer: list[float] = []
        expected_softmax: list[float] = []
        for row_index in range(2):
            row = values[row_index * 3 : (row_index + 1) * 3]
            rms_inverse = 1.0 / math.sqrt(sum(value * value for value in row) / 3 + 1e-5)
            expected_rms.extend(value * rms_inverse * weight for value, weight in zip(row, weights))
            mean = sum(row) / 3
            variance = sum((value - mean) ** 2 for value in row) / 3
            layer_inverse = 1.0 / math.sqrt(variance + 1e-5)
            expected_layer.extend(
                (value - mean) * layer_inverse * weight + bias
                for value, weight, bias in zip(row, weights, biases)
            )
            maximum = max(row)
            exponentials = [math.exp(value - maximum) for value in row]
            expected_softmax.extend(value / sum(exponentials) for value in exponentials)
        for actual, expected in zip(result.outputs[0].data, expected_rms):
            self.assertAlmostEqual(actual, expected, places=7)
        for actual, expected in zip(result.outputs[1].data, expected_layer):
            self.assertAlmostEqual(actual, expected, places=7)
        for actual, expected in zip(result.outputs[2].data, expected_softmax):
            self.assertAlmostEqual(actual, expected, places=7)

    def test_fixed_call_repeat_matches_manual_stateful_unroll(self) -> None:
        tensor = TensorType(DType.F32, (1,))
        program = Program(
            entry="main",
            states=(State("memory", tensor, StateAccess.READ_WRITE),),
            functions=(
                Function(
                    "main",
                    (Value("input", tensor),),
                    ("output",),
                    Region(
                        (
                            Op(
                                "call",
                                ("input",),
                                (Value("output", tensor),),
                                attributes(callee="step", repeat=3),
                            ),
                        )
                    ),
                ),
                Function(
                    "step",
                    (Value("current", tensor),),
                    ("updated",),
                    Region(
                        (
                            Op(
                                "state_read",
                                (),
                                (Value("memory_value", tensor),),
                                attributes(state="memory"),
                            ),
                            Op(
                                "add",
                                ("current", "memory_value"),
                                (Value("updated", tensor),),
                            ),
                            Op(
                                "state_write",
                                ("updated",),
                                (),
                                attributes(state="memory"),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {"input": Tensor.from_values(DType.F32, (1,), (1.0,))},
            states={"memory": Tensor.from_values(DType.F32, (1,), (1.0,))},
        )
        self.assertEqual(result.outputs[0].data, (8.0,))
        self.assertEqual(result.state("memory").data, (8.0,))
        self.assertIn(
            '%output: f32[1]<row_major,cpu> = call(%input) {"callee":"step","repeat":3}',
            dump_program(program),
        )

    def test_rope_pairings_use_explicit_positions(self) -> None:
        source_type = TensorType(DType.F32, (1, 1, 2, 4))
        table_type = TensorType(DType.F32, (2, 2))
        positions_type = TensorType(DType.I32, (1, 2))

        def program(pairing: str) -> Program:
            return Program(
                entry="main",
                functions=(
                    Function(
                        "main",
                        (
                            Value("x", source_type),
                            Value("cos", table_type),
                            Value("sin", table_type),
                            Value("positions", positions_type),
                        ),
                        ("out",),
                        Region(
                            (
                                Op(
                                    "rope",
                                    ("x", "cos", "sin", "positions"),
                                    (Value("out", source_type),),
                                    attributes(pairing=pairing),
                                ),
                            )
                        ),
                    ),
                ),
            )

        common_inputs = {
            "x": Tensor.from_values(DType.F32, (1, 1, 2, 4), (1, 2, 3, 4, 5, 6, 7, 8)),
            "cos": Tensor.from_values(DType.F32, (2, 2), (1, 1, 0, 0)),
            "sin": Tensor.from_values(DType.F32, (2, 2), (0, 0, 1, 1)),
            "positions": Tensor.from_values(DType.I32, (1, 2), (0, 1)),
        }
        interleaved = execute(program("interleaved"), common_inputs).outputs[0]
        split = execute(program("split_half"), common_inputs).outputs[0]
        self.assertEqual(interleaved.data, (1.0, 2.0, 3.0, 4.0, -6.0, 5.0, -8.0, 7.0))
        self.assertEqual(split.data, (1.0, 2.0, 3.0, 4.0, -7.0, -8.0, 5.0, 6.0))
        bad_inputs = dict(common_inputs)
        bad_inputs["positions"] = Tensor.from_values(DType.I32, (1, 2), (0, 2))
        with self.assertRaisesRegex(ValidationError, "RoPE position"):
            execute(program("interleaved"), bad_inputs)

    def test_masked_attention_matches_independent_softmax(self) -> None:
        tensor_type = TensorType(DType.F32, (1, 1, 2, 2))
        mask_type = TensorType(DType.BOOL, (1, 1, 2, 2))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (
                        Value("q", tensor_type),
                        Value("k", tensor_type),
                        Value("v", tensor_type),
                        Value("mask", mask_type),
                    ),
                    ("out",),
                    Region(
                        (
                            Op(
                                "scaled_dot_product_attention",
                                ("q", "k", "v", "mask"),
                                (Value("out", tensor_type),),
                                attributes(scale=1.0, kv_group_size=1, mask_fill="dtype_min"),
                            ),
                        )
                    ),
                ),
            ),
        )
        inputs = {
            "q": Tensor.from_values(DType.F32, (1, 1, 2, 2), (1, 0, 0, 1)),
            "k": Tensor.from_values(DType.F32, (1, 1, 2, 2), (1, 0, 0, 1)),
            "v": Tensor.from_values(DType.F32, (1, 1, 2, 2), (10, 0, 0, 20)),
            "mask": Tensor.from_values(DType.BOOL, (1, 1, 2, 2), (True, False, True, True)),
        }
        result = execute(program, inputs).outputs[0]
        denominator = 1.0 + math.e
        expected = (10.0, 0.0, 10.0 / denominator, 20.0 * math.e / denominator)
        for actual, wanted in zip(result.data, expected):
            self.assertAlmostEqual(actual, wanted, places=7)
        bad_inputs = dict(inputs)
        bad_inputs["mask"] = Tensor.from_values(
            DType.BOOL, (1, 1, 2, 2), (False, False, True, True)
        )
        fully_masked = execute(program, bad_inputs).outputs[0]
        self.assertEqual(fully_masked.data[:2], (5.0, 10.0))
        for actual, wanted in zip(fully_masked.data[2:], expected[2:]):
            self.assertAlmostEqual(actual, wanted, places=7)

    def test_grouped_query_attention_reuses_explicit_kv_heads(self) -> None:
        query = TensorType(DType.F32, (1, 2, 1, 2))
        key_value = TensorType(DType.F32, (1, 1, 2, 2))
        mask = TensorType(DType.BOOL, (1, 2, 1, 2))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("q", query), Value("k", key_value), Value("v", key_value), Value("mask", mask)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "scaled_dot_product_attention",
                                ("q", "k", "v", "mask"),
                                (Value("out", query),),
                                attributes(scale=1.0, kv_group_size=2, mask_fill="dtype_min"),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {
                "q": Tensor.from_values(DType.F32, (1, 2, 1, 2), (1, 0, 0, 1)),
                "k": Tensor.from_values(DType.F32, (1, 1, 2, 2), (1, 0, 0, 1)),
                "v": Tensor.from_values(DType.F32, (1, 1, 2, 2), (10, 0, 0, 20)),
                "mask": Tensor.from_values(DType.BOOL, (1, 2, 1, 2), (True, True, True, True)),
            },
        ).outputs[0]
        denominator = 1.0 + math.e
        expected = (10.0 * math.e / denominator, 20.0 / denominator, 10.0 / denominator, 20.0 * math.e / denominator)
        for actual, wanted in zip(result.data, expected):
            self.assertAlmostEqual(actual, wanted, places=7)

    def test_conv2d_patch_projection_matches_hand_summed_patches(self) -> None:
        source = TensorType(DType.F32, (1, 1, 4, 4))
        weight = TensorType(DType.F32, (1, 1, 2, 2))
        bias = TensorType(DType.F32, (1,))
        output = TensorType(DType.F32, (1, 1, 2, 2))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("image", source), Value("weight", weight), Value("bias", bias)),
                    ("patches",),
                    Region(
                        (
                            Op(
                                "conv2d",
                                ("image", "weight", "bias"),
                                (Value("patches", output),),
                                attributes(strides=(2, 2), pads=(0, 0, 0, 0)),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {
                "image": Tensor.from_values(DType.F32, (1, 1, 4, 4), tuple(range(1, 17))),
                "weight": Tensor.from_values(DType.F32, (1, 1, 2, 2), (1, 1, 1, 1)),
                "bias": Tensor.from_values(DType.F32, (1,), (0.5,)),
            },
        )
        self.assertEqual(result.outputs[0].data, (14.5, 22.5, 46.5, 54.5))

    def test_rank3_linear_bias_and_exact_gelu_match_formulas(self) -> None:
        source = TensorType(DType.F32, (1, 2, 3))
        weight = TensorType(DType.F32, (2, 3))
        bias = TensorType(DType.F32, (2,))
        output = TensorType(DType.F32, (1, 2, 2))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", source), Value("weight", weight), Value("bias", bias)),
                    ("projected", "activated"),
                    Region(
                        (
                            Op(
                                "linear",
                                ("x", "weight", "bias"),
                                (Value("projected", output),),
                            ),
                            Op(
                                "gelu",
                                ("projected",),
                                (Value("activated", output),),
                                attributes(approximation="none"),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {
                "x": Tensor.from_values(DType.F32, (1, 2, 3), (1, 2, 3, 4, 5, 6)),
                "weight": Tensor.from_values(DType.F32, (2, 3), (1, 0, -1, 0.5, 0.5, 0.5)),
                "bias": Tensor.from_values(DType.F32, (2,), (0.5, -1.0)),
            },
        )
        expected_linear = (-1.5, 2.0, -1.5, 6.5)
        self.assertEqual(result.outputs[0].data, expected_linear)
        expected_gelu = tuple(
            0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))
            for value in expected_linear
        )
        for actual, wanted in zip(result.outputs[1].data, expected_gelu):
            self.assertAlmostEqual(actual, wanted, places=7)

    def test_symbolic_shape_domain_accepts_and_rejects_real_bounds(self) -> None:
        tensor_type = TensorType(DType.F32, ("batch", 2))
        program = Program(
            entry="main",
            shape_domain=ShapeDomain((DimensionRange("batch", 1, 2, 4),)),
            functions=(
                Function(
                    "main",
                    (Value("left", tensor_type), Value("right", tensor_type)),
                    ("sum",),
                    Region((Op("add", ("left", "right"), (Value("sum", tensor_type),)),)),
                ),
            ),
        )
        left = Tensor.from_values(DType.F32, (3, 2), (1, 2, 3, 4, 5, 6))
        right = Tensor.from_values(DType.F32, (3, 2), (6, 5, 4, 3, 2, 1))
        self.assertEqual(execute(program, {"left": left, "right": right}).outputs[0].data, (7.0,) * 6)
        outside = Tensor.from_values(DType.F32, (5, 2), tuple(range(10)))
        with self.assertRaisesRegex(ValidationError, "outside its domain"):
            execute(program, {"left": outside, "right": outside})

    def test_state_read_write_is_explicit_and_returns_new_state(self) -> None:
        tensor_type = TensorType(DType.F32, (2,))
        program = Program(
            entry="main",
            states=(State("memory", tensor_type, StateAccess.READ_WRITE),),
            functions=(
                Function(
                    "main",
                    (Value("delta", tensor_type),),
                    ("updated",),
                    Region(
                        (
                            Op(
                                "state_read",
                                (),
                                (Value("previous", tensor_type),),
                                attributes(state="memory"),
                            ),
                            Op("add", ("previous", "delta"), (Value("updated", tensor_type),)),
                            Op("state_write", ("updated",), (), attributes(state="memory")),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {"delta": Tensor.from_values(DType.F32, (2,), (0.5, -1.0))},
            states={"memory": Tensor.from_values(DType.F32, (2,), (2.0, 3.0))},
        )
        self.assertEqual(result.outputs[0].data, (2.5, 2.0))
        self.assertEqual(result.state("memory"), result.outputs[0])
        with self.assertRaisesRegex(ValidationError, "states must be exactly"):
            execute(program, {"delta": Tensor.from_values(DType.F32, (2,), (0.0, 0.0))})

    def test_prefix_mask_cumsum_and_mixed_rms_norm_match_independent_math(self) -> None:
        mask_type = TensorType(DType.BOOL, (2, 3))
        position_type = TensorType(DType.I32, (2, 3))
        activation_type = TensorType(DType.BF16, (1, 2))
        parameter_type = TensorType(DType.F32, (2,))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (
                        Value("left", mask_type),
                        Value("right", mask_type),
                        Value("positions", position_type),
                        Value("activation", activation_type),
                        Value("weight", parameter_type),
                    ),
                    ("mask", "cumulative", "normalized"),
                    Region(
                        (
                            Op(
                                "logical_and",
                                ("left", "right"),
                                (Value("mask", mask_type),),
                            ),
                            Op(
                                "cumulative_sum",
                                ("positions",),
                                (Value("cumulative", position_type),),
                                attributes(axis=1),
                            ),
                            Op(
                                "rms_norm",
                                ("activation", "weight"),
                                (Value("normalized", activation_type),),
                                attributes(epsilon=1e-6),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {
                "left": Tensor.from_values(DType.BOOL, (2, 3), (True, True, False, True, False, True)),
                "right": Tensor.from_values(DType.BOOL, (2, 3), (True, False, True, True, True, False)),
                "positions": Tensor.from_values(DType.I32, (2, 3), (1, 0, 1, 3, 4, 5)),
                "activation": Tensor.from_values(DType.BF16, (1, 2), (3.0, 4.0)),
                "weight": Tensor.from_values(DType.F32, (2,), (2.0, 0.5)),
            },
        )
        self.assertEqual(result.outputs[0].data, (True, False, False, True, False, False))
        self.assertEqual(result.outputs[1].data, (1, 1, 2, 3, 7, 12))
        inverse_rms = 1.0 / math.sqrt((3.0**2 + 4.0**2) / 2.0 + 1e-6)
        expected = (3.0 * inverse_rms * 2.0, 4.0 * inverse_rms * 0.5)
        for actual, wanted in zip(result.outputs[2].data, expected):
            self.assertAlmostEqual(actual, wanted, places=7)

    def test_adaptive_rms_norm_and_gated_residual_match_independent_math(self) -> None:
        bf16_tokens = TensorType(DType.BF16, (1, 2, 2))
        f32_tokens = TensorType(DType.F32, (1, 2, 2))
        condition_type = TensorType(DType.F32, (1, 2))
        modulation_type = TensorType(DType.F32, (1, 6))
        component_type = TensorType(DType.F32, (1, 2))
        norm_weight_type = TensorType(DType.F32, (2,))
        dense_weight_type = TensorType(DType.F32, (6, 2))
        dense_bias_type = TensorType(DType.F32, (6,))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (
                        Value("hidden", bf16_tokens),
                        Value("condition", condition_type),
                        Value("branch", bf16_tokens),
                        Value("norm_weight", norm_weight_type),
                        Value("dense_weight", dense_weight_type),
                        Value("dense_bias", dense_bias_type),
                        Value("ones", f32_tokens),
                    ),
                    ("normalized", "residual"),
                    Region(
                        (
                            Op(
                                "cast",
                                ("hidden",),
                                (Value("hidden_f32", f32_tokens),),
                                attributes(dtype="f32"),
                            ),
                            Op(
                                "rms_norm",
                                ("hidden_f32", "norm_weight"),
                                (Value("unit_norm", f32_tokens),),
                                attributes(epsilon=1e-6),
                            ),
                            Op(
                                "linear",
                                ("condition", "dense_weight", "dense_bias"),
                                (Value("modulation", modulation_type),),
                            ),
                            Op(
                                "slice",
                                ("modulation",),
                                (Value("scale_flat", component_type),),
                                attributes(axis=1, start=0, stop=2),
                            ),
                            Op(
                                "slice",
                                ("modulation",),
                                (Value("shift_flat", component_type),),
                                attributes(axis=1, start=2, stop=4),
                            ),
                            Op(
                                "slice",
                                ("modulation",),
                                (Value("gate_flat", component_type),),
                                attributes(axis=1, start=4, stop=6),
                            ),
                            Op(
                                "broadcast_in_dim",
                                ("scale_flat",),
                                (Value("scale", f32_tokens),),
                                attributes(shape=(1, 2, 2), broadcast_dimensions=(0, 2)),
                            ),
                            Op(
                                "broadcast_in_dim",
                                ("shift_flat",),
                                (Value("shift", f32_tokens),),
                                attributes(shape=(1, 2, 2), broadcast_dimensions=(0, 2)),
                            ),
                            Op(
                                "broadcast_in_dim",
                                ("gate_flat",),
                                (Value("gate", f32_tokens),),
                                attributes(shape=(1, 2, 2), broadcast_dimensions=(0, 2)),
                            ),
                            Op("add", ("ones", "scale"), (Value("scale_plus_one", f32_tokens),)),
                            Op("mul", ("unit_norm", "scale_plus_one"), (Value("scaled", f32_tokens),)),
                            Op("add", ("scaled", "shift"), (Value("shifted", f32_tokens),)),
                            Op(
                                "cast",
                                ("shifted",),
                                (Value("normalized", bf16_tokens),),
                                attributes(dtype="bf16"),
                            ),
                            Op(
                                "cast",
                                ("gate",),
                                (Value("gate_bf16", bf16_tokens),),
                                attributes(dtype="bf16"),
                            ),
                            Op(
                                "mul",
                                ("branch", "gate_bf16"),
                                (Value("gated_branch", bf16_tokens),),
                            ),
                            Op(
                                "add",
                                ("hidden", "gated_branch"),
                                (Value("residual", bf16_tokens),),
                            ),
                        )
                    ),
                ),
            ),
        )
        result = execute(
            program,
            {
                "hidden": Tensor.from_values(DType.BF16, (1, 2, 2), (3.0, 4.0, 0.0, 5.0)),
                "condition": Tensor.from_values(DType.F32, (1, 2), (2.0, -1.0)),
                "branch": Tensor.from_values(DType.BF16, (1, 2, 2), (0.5, 1.0, 2.0, -1.0)),
                "norm_weight": Tensor.from_values(DType.F32, (2,), (1.0, 1.0)),
                "dense_weight": Tensor.from_values(
                    DType.F32,
                    (6, 2),
                    (1, 0, 0, 1, 0.5, 0, 0, 0.5, 1, 1, -1, 1),
                ),
                "dense_bias": Tensor.from_values(DType.F32, (6,), (0, 0, 0, 0, 0, 0)),
                "ones": Tensor.from_values(DType.F32, (1, 2, 2), (1, 1, 1, 1)),
            },
        )
        inverse_rms = 1.0 / math.sqrt(12.5 + 1e-6)
        expected_normalized = (3.0 * inverse_rms * 3.0 + 1.0, -0.5, 1.0, -0.5)
        for actual, wanted in zip(result.outputs[0].data, expected_normalized):
            self.assertAlmostEqual(actual, wanted, places=6)
        self.assertEqual(result.outputs[1].data, (3.5, 1.0, 2.0, 8.0))

    def test_default_rope_uses_explicit_positions_pairing_and_theta(self) -> None:
        source_type = TensorType(DType.F32, (1, 1, 2, 4))
        position_type = TensorType(DType.I32, (1, 2))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", source_type), Value("positions", position_type)),
                    ("split", "interleaved"),
                    Region(
                        (
                            Op(
                                "rope_default",
                                ("x", "positions"),
                                (Value("split", source_type),),
                                attributes(
                                    pairing="split_half",
                                    theta=10_000.0,
                                    frequency_dtype="bf16",
                                ),
                            ),
                            Op(
                                "rope_default",
                                ("x", "positions"),
                                (Value("interleaved", source_type),),
                                attributes(
                                    pairing="interleaved",
                                    theta=10_000.0,
                                    frequency_dtype="f32",
                                ),
                            ),
                        )
                    ),
                ),
            ),
        )
        source = Tensor.from_values(DType.F32, (1, 1, 2, 4), (1, 2, 3, 4, 1, 2, 3, 4))
        result = execute(
            program,
            {
                "x": source,
                "positions": Tensor.from_values(DType.I32, (1, 2), (0, 1)),
            },
        )
        self.assertEqual(result.outputs[0].data[:4], source.data[:4])
        self.assertEqual(result.outputs[1].data[:4], source.data[:4])
        bf16_frequency = 0.010009765625
        split_expected = (
            math.cos(1.0) - 3.0 * math.sin(1.0),
            2.0 * math.cos(bf16_frequency) - 4.0 * math.sin(bf16_frequency),
            math.sin(1.0) + 3.0 * math.cos(1.0),
            2.0 * math.sin(bf16_frequency) + 4.0 * math.cos(bf16_frequency),
        )
        interleaved_expected = (
            math.cos(1.0) - 2.0 * math.sin(1.0),
            math.sin(1.0) + 2.0 * math.cos(1.0),
            3.0 * math.cos(0.01) - 4.0 * math.sin(0.01),
            3.0 * math.sin(0.01) + 4.0 * math.cos(0.01),
        )
        for actual, wanted in zip(result.outputs[0].data[4:], split_expected):
            self.assertAlmostEqual(actual, wanted, places=7)
        for actual, wanted in zip(result.outputs[1].data[4:], interleaved_expected):
            self.assertAlmostEqual(actual, wanted, places=7)


def _silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


if __name__ == "__main__":
    unittest.main()
