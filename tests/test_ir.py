from __future__ import annotations

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
    TensorType,
    Value,
    dump_program,
    attributes,
    verify_program,
)
from tests.ir_fixtures import gated_mlp_program, vision_projection_program


class ProgramIRTests(unittest.TestCase):
    def test_two_unrelated_subgraphs_share_generic_ops(self) -> None:
        gated = gated_mlp_program()
        vision = vision_projection_program()
        verify_program(gated)
        verify_program(vision)
        self.assertEqual(
            {op.opcode for op in gated.functions[0].body.ops}
            & {op.opcode for op in vision.functions[0].body.ops},
            {"constant", "matmul"},
        )

    def test_dump_is_stable_and_canonical(self) -> None:
        program = vision_projection_program()
        expected = """program schema=1.0 entry=@vision_projection
func @vision_projection(%pixels: f32[2x3]<row_major,cpu>) -> (%tokens) {
  %weight: f32[3x2]<row_major,cpu> = constant() {"value":[1.0,0.0,0.0,1.0,1.0,1.0]}
  %projected: f32[2x2]<row_major,cpu> = matmul(%pixels, %weight)
  %activated: f32[2x2]<row_major,cpu> = relu(%projected)
  %tokens: f32[1x4]<row_major,cpu> = reshape(%activated) {"shape":[1,4]}
  return %tokens
}
"""
        self.assertEqual(dump_program(program), expected)
        self.assertEqual(dump_program(program), dump_program(program))

    def test_rejects_undefined_and_duplicate_ssa_values(self) -> None:
        tensor = TensorType(DType.F32, (2,))
        undefined = _single_op_program(
            Op("add", ("x", "missing"), (Value("out", tensor),)),
            tensor,
        )
        with self.assertRaisesRegex(ValidationError, "not defined before use"):
            verify_program(undefined)
        duplicate = _single_op_program(
            Op("relu", ("x",), (Value("x", tensor),)),
            tensor,
        )
        with self.assertRaisesRegex(ValidationError, "duplicate SSA"):
            verify_program(duplicate)

    def test_rejects_bad_op_contract_unknown_op_and_future_schema(self) -> None:
        left = TensorType(DType.F32, (2, 3))
        right = TensorType(DType.F32, (4, 2))
        output = TensorType(DType.F32, (2, 2))
        bad_matmul = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("left", left), Value("right", right)),
                    ("out",),
                    Region((Op("matmul", ("left", "right"), (Value("out", output),)),)),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "contracting dimensions"):
            verify_program(bad_matmul)
        tensor = TensorType(DType.F32, (2,))
        with self.assertRaisesRegex(ValidationError, "unsupported opcode"):
            verify_program(_single_op_program(Op("mystery", ("x",), (Value("out", tensor),)), tensor))
        future = Program(
            functions=gated_mlp_program().functions,
            entry="gated_mlp",
            schema_minor=1,
        )
        with self.assertRaisesRegex(ValidationError, "unsupported ProgramIR schema"):
            verify_program(future)

    def test_constant_ref_has_a_logical_address_not_embedded_payload(self) -> None:
        tensor = TensorType(DType.BF16, (2, 3))
        program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("trigger", TensorType(DType.F32, (1,))),),
                    ("weight",),
                    Region(
                        (
                            Op(
                                "constant_ref",
                                (),
                                (Value("weight", tensor),),
                                attributes(namespace="model", name="encoder.weight"),
                            ),
                        )
                    ),
                ),
            ),
        )
        verify_program(program)
        self.assertIn('constant_ref() {"name":"encoder.weight","namespace":"model"}', dump_program(program))
        bad = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("trigger", TensorType(DType.F32, (1,))),),
                    ("weight",),
                    Region((Op("constant_ref", (), (Value("weight", tensor),), attributes(namespace="", name="x")),)),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "non-empty strings"):
            verify_program(bad)

    def test_rejects_undeclared_symbol_and_read_only_state_write(self) -> None:
        symbolic = TensorType(DType.F32, ("batch", 2))
        program = _single_op_program(
            Op("relu", ("x",), (Value("out", symbolic),)),
            symbolic,
        )
        with self.assertRaisesRegex(ValidationError, "undeclared shape symbol"):
            verify_program(program)

        tensor = TensorType(DType.F32, (2,))
        state_program = Program(
            entry="main",
            states=(State("memory", tensor, StateAccess.READ_ONLY),),
            functions=(
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("x",),
                    Region((Op("state_write", ("x",), (), attributes(state="memory")),)),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "cannot write read-only"):
            verify_program(state_program)

    def test_rejects_invalid_transform_and_reduction_contracts(self) -> None:
        tensor = TensorType(DType.F32, (2, 3))
        bad_cast = _single_op_program(
            Op("cast", ("x",), (Value("out", tensor),), attributes(dtype="complex64")),
            tensor,
        )
        with self.assertRaisesRegex(ValidationError, "cast dtype is unknown"):
            verify_program(bad_cast)

        symbolic = TensorType(DType.F32, ("batch", 3))
        sliced = TensorType(DType.F32, (1, 3))
        symbolic_slice = Program(
            entry="main",
            shape_domain=ShapeDomain((DimensionRange("batch", 1, 2, 4),)),
            functions=(
                Function(
                    "main",
                    (Value("x", symbolic),),
                    ("out",),
                    Region(
                        (
                            Op(
                                "slice",
                                ("x",),
                                (Value("out", sliced),),
                                attributes(axis=0, start=0, stop=1),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "requires a static sliced dimension"):
            verify_program(symbolic_slice)

        bad_reduction = _single_op_program(
            Op(
                "reduce_sum",
                ("x",),
                (Value("out", tensor),),
                attributes(axes=(), keepdims=True),
            ),
            tensor,
        )
        with self.assertRaisesRegex(ValidationError, "non-empty tuple"):
            verify_program(bad_reduction)

        wrong_concat_output = TensorType(DType.F32, (2, 5))
        concat_program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("left", tensor), Value("right", tensor)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "concat",
                                ("left", "right"),
                                (Value("out", wrong_concat_output),),
                                attributes(axis=1),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "concat output type"):
            verify_program(concat_program)

        broadcast_output = TensorType(DType.F32, (2, 4, 3))
        bad_broadcast = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("out",),
                    Region(
                        (
                            Op(
                                "broadcast_in_dim",
                                ("x",),
                                (Value("out", broadcast_output),),
                                attributes(shape=(2, 4, 3), broadcast_dimensions=(1, 2)),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "mapped broadcast dimensions"):
            verify_program(bad_broadcast)

    def test_rejects_invalid_gather_norm_and_softmax_contracts(self) -> None:
        table = TensorType(DType.F32, (4, 3))
        bad_indices = TensorType(DType.F32, (2,))
        gathered = TensorType(DType.F32, (2, 3))
        gather_program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("table", table), Value("indices", bad_indices)),
                    ("out",),
                    Region((Op("gather", ("table", "indices"), (Value("out", gathered),)),)),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "I32 indices"):
            verify_program(gather_program)

        source = TensorType(DType.F32, (2, 3))
        wrong_weight = TensorType(DType.F32, (2,))
        bad_norm = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", source), Value("weight", wrong_weight)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "rms_norm",
                                ("x", "weight"),
                                (Value("out", source),),
                                attributes(epsilon=0.0),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "weight/bias"):
            verify_program(bad_norm)

        weight = TensorType(DType.F32, (3,))
        zero_epsilon = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", source), Value("weight", weight)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "rms_norm",
                                ("x", "weight"),
                                (Value("out", source),),
                                attributes(epsilon=0.0),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "epsilon must be"):
            verify_program(zero_epsilon)

        bad_softmax = _single_op_program(
            Op("softmax", ("x",), (Value("out", source),), attributes(axis=2)),
            source,
        )
        with self.assertRaisesRegex(ValidationError, "smaller than rank"):
            verify_program(bad_softmax)

    def test_rejects_unknown_recursive_and_non_iterable_calls(self) -> None:
        tensor = TensorType(DType.F32, (2,))
        unknown = _single_op_program(
            Op(
                "call",
                ("x",),
                (Value("out", tensor),),
                attributes(callee="missing", repeat=1),
            ),
            tensor,
        )
        with self.assertRaisesRegex(ValidationError, "unknown callee"):
            verify_program(unknown)

        recursive = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("out",),
                    Region(
                        (
                            Op(
                                "call",
                                ("x",),
                                (Value("out", tensor),),
                                attributes(callee="main", repeat=1),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "call graph is recursive"):
            verify_program(recursive)

        zero_repeat = Program(
            entry="main",
            functions=(
                Function(
                    "identity",
                    (Value("value", tensor),),
                    ("value",),
                    Region(()),
                ),
                Function(
                    "main",
                    (Value("x", tensor),),
                    ("out",),
                    Region(
                        (
                            Op(
                                "call",
                                ("x",),
                                (Value("out", tensor),),
                                attributes(callee="identity", repeat=0),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "repeat must be a positive"):
            verify_program(zero_repeat)

        non_iterable = Program(
            entry="main",
            functions=(
                Function(
                    "merge",
                    (Value("left", tensor), Value("right", tensor)),
                    ("left",),
                    Region(()),
                ),
                Function(
                    "main",
                    (Value("x", tensor), Value("y", tensor)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "call",
                                ("x", "y"),
                                (Value("out", tensor),),
                                attributes(callee="merge", repeat=2),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "signature is not iterable"):
            verify_program(non_iterable)

    def test_rejects_invalid_rope_and_attention_contracts(self) -> None:
        source = TensorType(DType.F32, (1, 1, 2, 3))
        table = TensorType(DType.F32, (4, 1))
        positions = TensorType(DType.I32, (1, 2))
        rope_program = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", source), Value("cos", table), Value("sin", table), Value("pos", positions)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "rope",
                                ("x", "cos", "sin", "pos"),
                                (Value("out", source),),
                                attributes(pairing="unknown"),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "pairing must be"):
            verify_program(rope_program)

        query = TensorType(DType.F32, (1, 1, 2, 2))
        wrong_mask = TensorType(DType.F32, (1, 1, 2, 2))
        attention = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (
                        Value("q", query),
                        Value("k", query),
                        Value("v", query),
                        Value("mask", wrong_mask),
                    ),
                    ("out",),
                    Region(
                        (
                            Op(
                                "scaled_dot_product_attention",
                                ("q", "k", "v", "mask"),
                                (Value("out", query),),
                                attributes(scale=0.0, kv_group_size=1, mask_fill="dtype_min"),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "scale must be"):
            verify_program(attention)

        key = TensorType(DType.F32, (1, 1, 2, 2))
        mask = TensorType(DType.BOOL, (1, 1, 2, 2))
        bad_gqa = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("q", query), Value("k", key), Value("v", key), Value("mask", mask)),
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
        with self.assertRaisesRegex(ValidationError, "query heads must equal"):
            verify_program(bad_gqa)

    def test_rejects_invalid_conv2d_contracts(self) -> None:
        source = TensorType(DType.F32, (1, 2, 4, 4))
        weight = TensorType(DType.F32, (3, 1, 2, 2))
        bias = TensorType(DType.F32, (3,))
        output = TensorType(DType.F32, (1, 3, 2, 2))

        def program(
            strides: tuple[int, int],
            selected_weight: TensorType = weight,
        ) -> Program:
            return Program(
                entry="main",
                functions=(
                    Function(
                        "main",
                        (Value("x", source), Value("weight", selected_weight), Value("bias", bias)),
                        ("out",),
                        Region(
                            (
                                Op(
                                    "conv2d",
                                    ("x", "weight", "bias"),
                                    (Value("out", output),),
                                    attributes(strides=strides, pads=(0, 0, 0, 0)),
                                ),
                            )
                        ),
                    ),
                ),
            )

        with self.assertRaisesRegex(ValidationError, "channel or bias"):
            verify_program(program((2, 2)))
        matching_weight = TensorType(DType.F32, (3, 2, 2, 2))
        with self.assertRaisesRegex(ValidationError, "no smaller than 1"):
            verify_program(program((0, 2), matching_weight))

    def test_rejects_invalid_linear_contracts(self) -> None:
        scalar = TensorType(DType.F32, ())
        weight = TensorType(DType.F32, (2, 3))
        bias = TensorType(DType.F32, (2,))
        output = TensorType(DType.F32, (2,))
        rank_zero = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", scalar), Value("weight", weight), Value("bias", bias)),
                    ("out",),
                    Region((Op("linear", ("x", "weight", "bias"), (Value("out", output),)),)),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "requires floating"):
            verify_program(rank_zero)

        source = TensorType(DType.F32, (1, 3))
        wrong_bias = TensorType(DType.F32, (3,))
        bad_bias = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", source), Value("weight", weight), Value("bias", wrong_bias)),
                    ("out",),
                    Region((Op("linear", ("x", "weight", "bias"), (Value("out", TensorType(DType.F32, (1, 2))),)),)),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "bias contract"):
            verify_program(bad_bias)

    def test_rejects_invalid_sinusoidal_and_implicit_gelu_semantics(self) -> None:
        timestep = TensorType(DType.F32, (2,))
        embedding = TensorType(DType.F32, (2, 3))
        odd = _single_op_program(
            Op(
                "sinusoidal_embedding",
                ("x",),
                (Value("out", embedding),),
                attributes(dimension=3, min_period=0.1, max_period=1.0),
            ),
            timestep,
        )
        with self.assertRaisesRegex(ValidationError, "positive even"):
            verify_program(odd)
        implicit_gelu = _single_op_program(Op("gelu", ("x",), (Value("out", timestep),)), timestep)
        with self.assertRaisesRegex(ValidationError, "attributes"):
            verify_program(implicit_gelu)

    def test_prefix_primitives_fail_closed_on_invalid_types_and_attributes(self) -> None:
        floats = TensorType(DType.F32, (2, 3))
        bad_logical = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("left", floats), Value("right", floats)),
                    ("out",),
                    Region((Op("logical_and", ("left", "right"), (Value("out", floats),)),)),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "identical BOOL"):
            verify_program(bad_logical)

        booleans = TensorType(DType.BOOL, (2, 3))
        bad_cumsum = _single_op_program(
            Op(
                "cumulative_sum",
                ("x",),
                (Value("out", booleans),),
                attributes(axis=1),
            ),
            booleans,
        )
        with self.assertRaisesRegex(ValidationError, "numeric tensor"):
            verify_program(bad_cumsum)

        rope_source = TensorType(DType.BF16, (1, 2, 3, 4))
        positions = TensorType(DType.I32, (1, 3))
        bad_rope = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", rope_source), Value("positions", positions)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "rope_default",
                                ("x", "positions"),
                                (Value("out", rope_source),),
                                attributes(
                                    pairing="split_half",
                                    theta=0.0,
                                    frequency_dtype="bf16",
                                ),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "positive finite"):
            verify_program(bad_rope)

        bad_frequency = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", rope_source), Value("positions", positions)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "rope_default",
                                ("x", "positions"),
                                (Value("out", rope_source),),
                                attributes(
                                    pairing="split_half",
                                    theta=10_000.0,
                                    frequency_dtype="bool",
                                ),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "frequency_dtype must be floating"):
            verify_program(bad_frequency)

        norm_source = TensorType(DType.BF16, (1, 4))
        wrong_parameter = TensorType(DType.F16, (4,))
        bad_norm = Program(
            entry="main",
            functions=(
                Function(
                    "main",
                    (Value("x", norm_source), Value("weight", wrong_parameter)),
                    ("out",),
                    Region(
                        (
                            Op(
                                "rms_norm",
                                ("x", "weight"),
                                (Value("out", norm_source),),
                                attributes(epsilon=1e-6),
                            ),
                        )
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValidationError, "weight/bias"):
            verify_program(bad_norm)


def _single_op_program(op: Op, input_type: TensorType) -> Program:
    output_id = op.outputs[0].value_id if op.outputs else "x"
    return Program(
        entry="main",
        functions=(
            Function("main", (Value("x", input_type),), (output_id,), Region((op,))),
        ),
    )


if __name__ == "__main__":
    unittest.main()
