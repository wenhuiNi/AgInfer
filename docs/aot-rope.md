# AOT CUDA split-half RoPE provider v1

The built-in AOT CUDA provider exposes four exact BF16 RoPE regions:

```text
input          [1, sequence, heads, 256]  row-major BF16 CUDA (BSHD)
positions      [1, sequence]              row-major I32 CUDA
output         [1, heads, sequence, 256]  row-major BF16 CUDA (BHSD)

sequence       968 or 50
heads          8 or 1
theta          10000
frequency      BF16
pairing        split-half
target         SM120
```

Each command fuses the exclusive input BSHD-to-BHSD transpose with RoPE. The
128 inverse frequencies are rounded to BF16 before the angle is formed, which
matches the source model contract. The rotation uses the first and second
128-element halves as pairs; it is not the adjacent-element, interleaved RoPE
form used by some fused QKV kernels.

`kernels/rope.cu` exports one fixed symbol for each sequence/head combination.
One thread handles one split-half pair, computes the angle and sine/cosine in
F32, rounds sine/cosine to BF16, and performs two separately rounded BF16
products followed by a BF16 sum for each output. These intermediate roundings
are part of the source semantics; an F32 rotation with only a final BF16 cast
is not equivalent. The four kernels are part of the same
target-specific, PTX-free CUBIN as the other built-in AOT kernels.

The 192-byte numeric payload binds the variant, input/output layouts, dtypes,
shape, theta, frequency and pairing conventions, launch geometry, alignments,
tensor byte sizes, CUBIN byte size, and CUBIN SHA-256. It contains no pointers
or symbol strings.

The CUBIN digest distinguishes corrected arithmetic from earlier builds with
final-only rounding. A receipt for an earlier module cannot validate the new
module merely by replacing its digest; numerical evidence must be rerun.

`AotRopeCommand::Prepare` validates the payload, active architecture,
caller-owned module identity, all three tensor bindings, and output
non-aliasing, then resolves the fixed symbol. `Execute` performs one driver
launch on the caller stream. It does not allocate, synchronize, load modules,
select a tactic, convert dtypes, or fall back.

Other sequence lengths, head counts, head dimensions, frequency dtypes,
position dtypes, theta values, pairing conventions, layouts, in-place forms,
and architectures remain unresolved at compile time. A transpose is fused
only when the lowering proves that it has the exact BSHD-to-BHSD permutation
and its result has no other consumer.
