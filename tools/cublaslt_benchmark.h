// Private implementation included in the offline selector's anonymous namespace.
// Synthetic inputs are a tactic prefilter, never a model accuracy reference.
bool BenchmarkEligible(const Request& r) {
  const auto& a = r.layouts[0]; const auto& b = r.layouts[1]; const auto& c = r.layouts[2];
  return r.dtype == 2 && r.compute == 1 && r.bias == 1 && r.batch == 1 &&
      r.trans_a == 1 && r.trans_b == 0 && b.cols >= 2 && b.cols <= 128 &&
      a.rows >= 256 && a.cols >= 256 && a.ld == a.rows && b.ld == b.rows && c.ld == c.rows &&
      r.alignments == std::array<std::uint32_t, 3>{256,256,256} &&
      2 * (a.rows*a.cols + b.rows*b.cols + c.rows*c.cols + c.rows) + r.workspace <= (128LL << 20);
}
void Cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct BenchResources {
  std::array<void*, 5> buffers{}; // A, B, D=C (beta=0), bias, workspace
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, end = nullptr;
  std::array<cudaGraph_t, 4> graphs{};
  std::array<cudaGraphExec_t, 4> execs{};
  ~BenchResources() {
    if (stream) cudaStreamSynchronize(stream);
    for (auto x : execs) if (x) cudaGraphExecDestroy(x);
    for (auto x : graphs) if (x) cudaGraphDestroy(x);
    for (auto x : buffers) if (x) cudaFree(x);
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
    if (stream) cudaStreamDestroy(stream);
  }
};
double Median(std::array<float, 3> times) {
  std::sort(times.begin(), times.end()); return times[1];
}
std::size_t Benchmark(Resources& ctx, const Request& r, const std::vector<Candidate>& candidates,
                      std::string& receipt) {
  constexpr int repeats = 8;
  BenchResources b;
  Cuda(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking));
  Cuda(cudaEventCreate(&b.start)); Cuda(cudaEventCreate(&b.end));
  std::array<std::size_t, 5> sizes{};
  for (int i = 0; i < 3; ++i) sizes[i] = r.layouts[i].rows * r.layouts[i].cols * 2;
  sizes[3] = r.layouts[2].rows * 2; sizes[4] = r.workspace;
  std::uint32_t seed = 0x125ad4e7;
  for (int i = 0; i < 5; ++i) {
    if (!sizes[i]) continue;
    Cuda(cudaMalloc(&b.buffers[i], sizes[i]));
    if (i == 4) continue;
    std::vector<std::uint16_t> data(sizes[i]/2);
    for (auto& x : data) {
      seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
      const float value = (static_cast<int>(seed % 2049) - 1024) / 4096.f;
      std::uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
      x = static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
    }
    Cuda(cudaMemcpyAsync(b.buffers[i], data.data(), sizes[i], cudaMemcpyHostToDevice, b.stream));
    Cuda(cudaStreamSynchronize(b.stream));
  }
  Check(cublasLtMatmulDescSetAttribute(ctx.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                      &b.buffers[3], sizeof(void*)));
  const float alpha = 1.f, beta = 0.f;
  auto launch = [&](std::size_t i) {
    return cublasLtMatmul(ctx.handle, ctx.op, &alpha, b.buffers[0], ctx.layouts[0],
        b.buffers[1], ctx.layouts[1], &beta, b.buffers[2], ctx.layouts[2], b.buffers[2],
        ctx.layouts[2], &candidates[i].algo, b.buffers[4], r.workspace, b.stream);
  };
  std::vector<std::uint16_t> reference(sizes[2]/2), output(reference.size());
  auto read = [&] {
    Cuda(cudaMemcpyAsync(output.data(), b.buffers[2], sizes[2], cudaMemcpyDeviceToHost, b.stream));
    Cuda(cudaStreamSynchronize(b.stream));
    for (auto x : output) Require((x & 0x7f80) != 0x7f80, "nonfinite tactic output");
  };
  std::array<bool, 4> matches{};
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    Cuda(cudaMemsetAsync(b.buffers[2], 0xff, sizes[2], b.stream));
    Check(launch(i)); read();
    if (i == 0) reference = output;
    matches[i] = output == reference;
    if (!matches[i]) continue;
    Cuda(cudaStreamBeginCapture(b.stream, cudaStreamCaptureModeThreadLocal));
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    for (int j = 0; j < repeats && status == CUBLAS_STATUS_SUCCESS; ++j) status = launch(i);
    const auto capture = cudaStreamEndCapture(b.stream, &b.graphs[i]);
    Check(status); Cuda(capture);
    Cuda(cudaGraphInstantiateWithFlags(&b.execs[i], b.graphs[i], 0));
    Cuda(cudaMemsetAsync(b.buffers[2], 0xff, sizes[2], b.stream));
    Cuda(cudaGraphLaunch(b.execs[i], b.stream)); read();
    Require(output == reference, "tactic Graph replay differs from eager baseline");
  }
  std::array<std::array<float, 3>, 4> times{};
  // Reverse order between rounds so the baseline is not always timed first.
  for (int round = 0; round < 3; ++round) {
    for (std::size_t step = 0; step < candidates.size(); ++step) {
      const auto i = round % 2 ? candidates.size()-1-step : step;
      if (!matches[i]) continue;
      Cuda(cudaEventRecord(b.start, b.stream));
      Cuda(cudaGraphLaunch(b.execs[i], b.stream));
      Cuda(cudaEventRecord(b.end, b.stream)); Cuda(cudaEventSynchronize(b.end));
      float ms = 0; Cuda(cudaEventElapsedTime(&ms, b.start, b.end));
      Require(std::isfinite(ms) && ms > 0, "invalid tactic timing");
      times[i][round] = ms * 1000.f / repeats;
    }
  }
  std::size_t chosen = 0;
  for (std::size_t i = 1; i < candidates.size(); ++i) {
    if (!matches[i] || Median(times[i]) >= .98 * Median(times[0])) continue;
    bool wins = true;
    for (int j = 0; j < 3; ++j) wins &= times[i][j] < times[0][j];
    if (wins && Median(times[i]) < Median(times[chosen])) chosen = i;
  }
  std::ostringstream out; out << std::setprecision(9);
  out << "{\"repeats\":8,\"input\":\"synthetic-bf16-v1\",\"candidates\":[";
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    out << (i ? "," : "") << "{\"rank\":" << candidates[i].rank
        << ",\"matches_baseline\":" << (matches[i] ? "true" : "false") << ",\"times_us\":[";
    if (matches[i]) for (int j = 0; j < 3; ++j) out << (j ? "," : "") << times[i][j];
    out << "]}";
  }
  receipt = out.str() + "]}";
  return chosen;
}
