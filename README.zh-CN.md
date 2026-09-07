<div align="center">

# AgInfer

**面向 VLA/VLM 目标相关 AOT 部署的实验性基础实现**

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=cplusplus&logoColor=white)](CMakeLists.txt)

[English](README.md) · 简体中文

</div>

AgInfer 正在实现离线模型编译器和独立的 C++/CUDA Runtime。目标部署 artifact
将包含 lowered program、packed weights、目标相关实现和经过验证的执行元数据，
使部署进程不依赖 Python、PyTorch、ONNX Runtime 或 TensorRT。

## 项目状态

当前仓库仍处于基础实现阶段，尚未提供可发布的模型 importer、优化器、量化器、
生产级 artifact schema 或端到端支持模型。因此 GR00T 和 LingBot 不再作为已支持
模型对外声明。

目前实际实现的能力包括：

- 有边界检查的 safetensors header 读取，以及 pickle checkpoint 拒绝；
- 支持单文件/分片 safetensors 的 namespace-aware source manifest、有界常量访问和
  fail-closed recipe coverage；
- 实验性的 mmap-friendly AIM 容器与范围检查；完整 SHA-256 检查在离线 verify 或
  Debug/显式校验 Runtime 构建中执行，部署构建默认不扫描，详见
  [加载校验策略](docs/runtime-contract.md#load-time-checksum-policy)；
- 实验性v1 kernel-launch与v2 tagged-provider二进制执行计划；
- 离线source-to-AIM候选编译器、显式固定算法、内容寻址build record和v2可执行payload校验器；
- 最小 typed-SSA ProgramIR verifier、稳定文本 dump，以及覆盖首批通用 op 的
  纯 Python CPU reference executor；
- 使用 opaque handle、numeric port、显式 prepare/bind/enqueue 的 C ABI，
  以及 C++17 RAII 薄封装；
- 已由真实目标架构 fixture 覆盖的 CUDA Driver launch 路径；
- 默认 CUDA 构建会生成目标架构专用、无 PTX 的 AOT
  cast/pointwise/exact tanh-GELU/SiLU/F32 vision attention/LayerNorm/RMSNorm/
  split-half RoPE/denoise KV-pack/prefix KV-store/denoise metadata/prefix-input
  assembly/patchify CUBIN，
  并包含 native prepared command 与 exact cuBLASLt linear provider；
- 面向 SM120、PI0.5 类 BF16 GQA denoise region 的 exact FlashInfer FA2 command，
  直接消费 dense 2-D BOOL mask 并融合输出 BSHD；
- 面向 SM120、PI0.5 类 BF16 GQA prefix region 的 exact FlashInfer FA2 command，
  直接消费共享 1-D pad mask，保留 finite-mask fully-masked-row 语义，并融合四个
  mask op 与输出 BSHD；
- 面向 SM120、PI0.5 类 F32 vision-attention region 的自有 exact AOT command，
  融合输入/输出 transpose 与 scalar-mask broadcast；
- 面向 SM120、F32 `[1,256,1152]` vision 边界的自有 exact affine LayerNorm
  AOT command；
- 面向 SM120、F32 `[1,50,1024]` 与 BF16 activation/F32 weight
  `[1,968,2048]` 边界的自有 exact RMSNorm AOT command；
- 面向 SM120、BF16 `[1,50,1024]` hidden 与 F32 `[1,3072]`
  scale/shift/gate modulation 的自有 exact fused adaptive RMSNorm command，
  用一次 launch替代独占的 cast/norm/slice/broadcast/pointwise链；
- 面向 SM120 的 exact denoise KV-pack command，用一次 launch 拼接
  prefix/current BF16 K/V，并吸收单 KV head 的 BSHD→BHSD V transpose；
- 面向 SM120 的 exact prefix KV state-store command，用一次 launch写入两项
  persistent state，并由内存规划把 singleton-axis row-major transpose证明为view；
- 面向 SM120 的 exact denoise suffix-metadata command，用一次 launch扩展
  prefix pad mask并构造50项position ID，同时逐项保留mask中的空洞；
- 面向 SM120 的 exact prefix-input command，用一次 launch完成语言 embedding
  gather/scale、三路投影图像拼接以及共享 pad mask/position ID 构造；
- 面向 SM120 的 exact patch-projection command，将 AOT NCHW patchify 与锁定的
  cuBLASLt F32 projection组合，并直接写出NHWC；
- 面向 SM120 的 exact F32 time sinusoidal-embedding command，用一次launch保留
  源模型的Float64周期与三角函数语义；
- 面向 SM120 的 exact terminal F32 action-slice command，从每个32-wide row
  复制 `[0,7)`，不把跨row非连续输出伪装成alias；
- 将静态 BOOL/I32/F32/BF16 literal broadcast 确定性物化为经校验、按内容去重的
  constant blob，并对只插入 singleton 轴的 broadcast 做零拷贝内存规划；
- 面向 SM120、sequence 968/50、heads 8/1 的自有 exact split-half BF16 RoPE
  command，并融合各自独占的 BSHD→BHSD 输入 transpose；
- 面向 SM120 三个已交付 BF16/F32 activation shape 的自有 exact tanh-GELU
  command，以及用于 `[1,1024]` 的 F32 SiLU command；
- artifact 损坏、目标不匹配、tensor contract 和 launch plan 测试。

当前 AIM schema 和 Runtime ABI 均为实验接口，不提供兼容性承诺。SHA-256 只能检测
意外损坏，不能认证来自不可信来源的 artifact。

已经实现的 ownership、版本和执行规则见
[实验性 Runtime contract](docs/runtime-contract.md)。
独立版本的语义层见 [实验性 ProgramIR contract](docs/program-ir.md)。
离线 checkpoint 清单与常量寻址规则见
[实验性 source contract](docs/source-contract.md)。
首个仅覆盖 source 的 recipe 审计见
[PI0.5 source inventory](docs/pi05-source-inventory.md)；这不代表端到端模型支持。
当前三个 attention region 分别见
[FlashInfer contract](docs/flashinfer-attention.md) 与
[自有 F32 vision-attention contract](docs/aot-vision-attention.md)。
当前 normalization 边界见
[自有 F32 LayerNorm contract](docs/aot-layer-norm.md) 与
[自有 RMSNorm contract](docs/aot-rms-norm.md)。融合 denoise normalization
region 见 [adaptive RMSNorm contract](docs/aot-adaptive-rms-norm.md)。
exact denoise K/V 物化边界见 [KV-pack contract](docs/aot-kv-pack.md)。
成对 persistent prefix-cache 写入见
[prefix KV state-store contract](docs/aot-prefix-kv-store.md)。
denoise mask/position 构造边界见
[suffix-metadata contract](docs/aot-suffix-metadata.md)。
prefix embedding/mask/position assembly 边界见
[prefix-input contract](docs/aot-prefix-input.md)。
非重叠vision patch projection见
[patch-projection contract](docs/patch-projection.md)。
固定Float64语义的timestep展开见
[time-embedding contract](docs/aot-time-embedding.md)。
tensor-only terminal输出边界见
[action-slice contract](docs/aot-action-slice.md)。
离线 literal broadcast 编码与 memory-plan 集成见
[compile-time materialization contract](docs/literal-materialization.md)。
固定的 split-half rotary region 见
[自有 RoPE contract](docs/aot-rope.md)。
固定 activation region 见
[自有 activation contract](docs/aot-activation.md)。

## 构建与测试

无需安装 package 即可运行 Python contract tests：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests
```

构建默认 CUDA Runtime、native providers 和目标架构 AOT kernel bundle，然后运行
C++ contract tests。默认目标为 `sm120`：

```bash
git submodule update --init third_party/flashinfer
git -C third_party/flashinfer submodule update --init 3rdparty/cccl
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

构建只消费固定版本的 FlashInfer 与 CCCL 源码，部署 Runtime 不依赖 Python/JIT。
精确支持边界见 [attention provider contract](docs/flashinfer-attention.md)。

可用 `-DAGINFER_CUDA_ARCH=89` 或 `110` 显式选择其他支持的部署架构。如只需
不链接 CUDA Toolkit/provider 的 contract build，可显式关闭 CUDA：

```bash
cmake -S . -B build-contract -DAGINFER_ENABLE_CUDA=OFF
cmake --build build-contract -j
ctest --test-dir build-contract --output-on-failure
```

仓库测试不代表模型级正确性或性能结论。

在受支持的 GPU 上可以显式开启直接 CUDA 执行 cell；架构必须与当前设备一致：

```bash
cmake -S . -B build-gpu -DAGINFER_BUILD_CUDA_TESTS=ON \
  -DAGINFER_CUDA_ARCH=120
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
```

## Artifact 检查

安装 Python package 后可以使用 artifact 检查命令：

```bash
python3 -m pip install .
aginfer inspect model.aim
```

`aginfer inspect` 会先验证实验性容器，再显示 platform、CUDA variants、manifest、
graph metadata、tensor 数量、文件大小和 digest。实验性 `aginfer select-algorithms`、`aginfer compile` 与
`aginfer verify` 的使用方式见[候选编译说明](docs/candidate-compilation.md)。
原生AlgoCheck工具可生成显式离线算法选择；编译生成尚未数值验收的候选。
选择器不计时、不宣称最快；这不是生产模型支持承诺。

可以在不加载 tensor payload 或框架代码的情况下检查本地 checkpoint：

```bash
aginfer source-manifest /path/to/checkpoint --offline --output /tmp/source.json
aginfer source-contract /path/to/checkpoint --offline
```

第二个命令报告有类型的模型 IO、有序 processor steps、引用的 state assets，以及
tokenizer 等尚未内置的外部资产需求。

## Runtime 边界

部署 Runtime 只消费已经 lower 的执行 artifact。模型导入、图重写、校准、量化、
tactic 选择、权重变换和性能 qualification 全部属于离线构建阶段。Runtime 不得
静默 fallback、JIT 编译、自动调优或重新 pack 权重。

## 许可证

AgInfer 使用 [MIT License](LICENSE)。
