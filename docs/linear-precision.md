# Experimental BF16 large-linear islands

`select-algorithms --bf16-large-linears` and `compile --bf16-large-linears`
must be used together. This opt-in compiler precision form is **not an accepted
deployment optimization**: it passed the initial real observation but failed
the changed-image/noise parity gate. The default remains unchanged.

Matching requires CUDA row-major F32 linear boundaries, at least 128 rows and
N/K at least 256, with weight and bias originating from explicit widening of
immutable BF16 constants. It reuses those BF16 parameters, shares sibling input
rounding, executes BF16 GEMM with F32 accumulation, and restores the original F32
output interface. True F32 parameters are not silently narrowed. Norm, attention,
patch projection, camera count, token length and denoising steps are unchanged.
This is not full BF16 vision or FP8 inference. Extra boundary casts are a known
cost; a later fused producer/consumer form must earn its own correctness gate.

On the local SM120 ten-step model, 163 static sites execute 489 times across
three views. The four new BF16 GEMM shapes use M=256 with (N,K) equal to
(1152,1152), (4304,1152), (1152,4304), and (2048,1152). Shared baseline tactics
and attention payloads were unchanged.

The original observation passed cosine/max/p99 gates, but jointly negating the
normalized first image and initial noise produced cosine about 0.93585 and
max error about 1.92443 versus the reference-precision native path. This
perturbation is a sensitivity control, not held-out distribution coverage or a
closed-loop evaluation. It prevents a blanket parity claim. Testing stopped at
that failure; restore validation did not run. Do not treat the initial-sample
speed measurement as an accepted model speedup. Next work must isolate sensitive
regions and compare additional real observations against the source host before
promoting a mixed-precision policy; do not relax gates to accept this candidate.
