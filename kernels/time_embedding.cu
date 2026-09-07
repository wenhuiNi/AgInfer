#include <cuda_runtime.h>

#include <cstdint>

// Exact PI0.5/OpenPI envelope.  The source implementation constructs the
// geometric period grid in float64, evaluates sin/cos in float64, then casts
// the concatenated result to the float32 timestep dtype.
extern "C" __global__ void aginfer_time_embedding_f32_d1024(
    const float* timestep, float* output) {
  constexpr std::uint32_t kHalf = 512;
  constexpr double kMinimumPeriod = 0.004;
  constexpr double kPeriodRatio = 1000.0;
  constexpr double kTwoPi = 6.283185307179586476925286766559005768;

  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kHalf) return;
  const double fraction = static_cast<double>(index) / 511.0;
  const double period = kMinimumPeriod * pow(kPeriodRatio, fraction);
  const double scale = (1.0 / period) * kTwoPi;
  const double angle = scale * static_cast<double>(timestep[0]);
  double sine = 0.0;
  double cosine = 0.0;
  sincos(angle, &sine, &cosine);
  output[index] = static_cast<float>(sine);
  output[kHalf + index] = static_cast<float>(cosine);
}
