# AgInfer Kernel ABI 1.0

The Kernel ABI defines how an AIM Plan v1 launch record invokes a function in
the selected architecture's CUBIN bundle. Runtime ABI 1 accepts Kernel ABI 1.

## CUBIN requirements

- The bundle is a 64-bit, little-endian NVIDIA CUDA ELF image.
- ELF target flags must match the AIM variant exactly.
- Every plan launch names an `extern "C" __global__` symbol in the bundle.
- PTX text and relocatable PTX fallback are forbidden.
- A kernel must not allocate memory, discover shapes, select tactics, or invoke
  runtime compilation.

The Runtime loads the image with `cuModuleLoadData`, resolves symbols with
`cuModuleGetFunction`, and submits launches with `cuLaunchKernel` on the
caller's stream. Launches are asynchronous after one-time Session resource
initialization.

## Execution plan

The compiler converts `plan.json` into the binary `AIMPLN1\0` format embedded
in each architecture variant. All integers are little-endian.

| Record | Size | Purpose |
|---|---:|---|
| Plan header | 128 bytes | Schema, exact arch, table counts/offsets, arena and workspace sizes |
| Profile | 64 bytes | Tensor and launch ranges for one exact shape dispatch |
| Tensor | 160 bytes | Name, dtype, location, I/O kind, byte size, shape, and stride |
| Launch | 64 bytes | Symbol, profile, argument range, grid, block, and dynamic shared memory |
| Argument | 32 bytes | Tensor pointer, scalar, arena pointer, or packed-weight pointer |
| String table | variable | NUL-terminated UTF-8 profile, tensor, and Kernel names |

Tables are contiguous and in canonical order. AIM's per-plan SHA-256 protects
the complete binary plan. The Runtime additionally validates every range,
record, tensor extent, launch dimension, argument reference, reserved field,
and string before loading CUDA resources.

## Kernel argument kinds

Arguments appear in the exact order declared by the CUDA Kernel signature.

| JSON form | CUDA ABI value | Description |
|---|---|---|
| `{"tensor": "name"}` | 64-bit device pointer | Caller-provided input or output buffer |
| `{"scalar_u32": N}` | 32-bit unsigned value | Compile-time scalar |
| `{"scalar_f32": X}` | 32-bit IEEE 754 value | Compile-time scalar |
| `{"arena_offset": N}` | 64-bit device pointer | Session arena base plus validated byte offset |
| `{"weights_offset": N}` | 64-bit device pointer | Packed-weight base plus validated byte offset |

Host tensors cannot be passed to a CUDA Kernel. Arena and weight offsets must
be inside their declared allocations. Runtime does not reinterpret, cast, or
repack arguments.

## Tensor contract

Plan v1 supports the following native tensor types:

| Plan dtype | C++ `DType` | Bytes |
|---|---|---:|
| `F32` | `kFp32` | 4 |
| `F16` | `kFp16` | 2 |
| `BF16` | `kBf16` | 2 |
| `F8_E4M3` | `kFp8E4M3` | 1 |
| `I32` | `kInt32` | 4 |

Before launch, each `TensorView` must match the selected profile's name,
dtype, memory location, rank, shape, and stride. Its buffer must be non-null
and large enough for the plan's byte-size requirement. Out-of-profile tensors
return `INVALID_ARGUMENT`; the Runtime does not create a new dispatch.

## Example launch

```json
{
  "kernel": "vector_add_f32",
  "grid": [1, 1, 1],
  "block": [256, 1, 1],
  "shared_bytes": 0,
  "arguments": [
    {"tensor": "left"},
    {"tensor": "right"},
    {"tensor": "output"},
    {"scalar_u32": 256}
  ]
}
```

The corresponding Kernel signature is:

```cuda
extern "C" __global__ void vector_add_f32(
    const float* left,
    const float* right,
    float* output,
    unsigned int count);
```

