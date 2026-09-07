from __future__ import annotations

from aginfer.ir import DType, Function, Op, Program, Region, TensorType, Value, attributes


def gated_mlp_program() -> Program:
    matrix = TensorType(DType.F32, (2, 2))
    return Program(
        entry="gated_mlp",
        functions=(
            Function(
                name="gated_mlp",
                inputs=(Value("x", matrix),),
                outputs=("out",),
                body=Region(
                    (
                        Op(
                            "constant",
                            (),
                            (Value("gate_weight", matrix),),
                            attributes(value=(1.0, 0.0, 0.0, 1.0)),
                        ),
                        Op(
                            "constant",
                            (),
                            (Value("up_weight", matrix),),
                            attributes(value=(1.0, 1.0, 1.0, -1.0)),
                        ),
                        Op("matmul", ("x", "gate_weight"), (Value("gate", matrix),)),
                        Op("silu", ("gate",), (Value("activated", matrix),)),
                        Op("matmul", ("x", "up_weight"), (Value("up", matrix),)),
                        Op("mul", ("activated", "up"), (Value("out", matrix),)),
                    )
                ),
            ),
        ),
    )


def vision_projection_program() -> Program:
    pixels = TensorType(DType.F32, (2, 3))
    weight = TensorType(DType.F32, (3, 2))
    projected = TensorType(DType.F32, (2, 2))
    flattened = TensorType(DType.F32, (1, 4))
    return Program(
        entry="vision_projection",
        functions=(
            Function(
                name="vision_projection",
                inputs=(Value("pixels", pixels),),
                outputs=("tokens",),
                body=Region(
                    (
                        Op(
                            "constant",
                            (),
                            (Value("weight", weight),),
                            attributes(value=(1.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
                        ),
                        Op("matmul", ("pixels", "weight"), (Value("projected", projected),)),
                        Op("relu", ("projected",), (Value("activated", projected),)),
                        Op(
                            "reshape",
                            ("activated",),
                            (Value("tokens", flattened),),
                            attributes(shape=(1, 4)),
                        ),
                    )
                ),
            ),
        ),
    )
