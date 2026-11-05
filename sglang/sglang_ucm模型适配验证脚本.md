# SGLang + UCM 模型适配检测需求

## 1. 目标与依据

以 UCM 主干 `develop`（`4e47254`）的 SGLang 适配为依据，设计一个
**不启动服务、不加载权重**的检测工具，回答两个问题：

- 模型配置和结构能否被当前 SGLang 识别、构建？
- 该模型需要的缓存能否被 UCM 正确保存、查询和恢复？

结论只适用于被测的模型、框架版本、平台和缓存配置，不代表模型一定能正常推理。
下文为设计方案及接口示例，不表示主干已经提供全部检测能力。

## 2. 正常启动与检测流程对照

常规服务的主要启动路径如下；部分步骤会随平台和配置交错执行。

```text
读取参数和模型配置
    ↓
启动调度进程，初始化设备和通信
    ↓
建立模型结构 → 加载权重 → 权重后处理
    ↓
分配设备缓存（L1），初始化 Attention Backend、图捕获和预热
    ↓
建立缓存管理；启用 HiCache 时分配 Host 缓存（L2）
    ↓
配置 UCM 时创建存储后端、注册缓存池
    ↓
启动 HTTP 接口，完成服务预热并就绪
```

| 正常服务环节 | 检测脚本的对应方式 |
| --- | --- |
| 读取配置、识别模型 | inspect 复用 SGLang 的配置解析和模型识别逻辑 |
| 建立模型结构 | inspect 尝试使用 meta 模式，只建立参数形状，不存放权重数据 |
| 加载权重、执行算子、图捕获和服务预热 | 跳过 |
| 创建缓存 | roundtrip 在 CPU 上创建少量真实格式的缓存；复杂类型复用原生缓存构造逻辑 |
| 注册 UCM、保存和恢复缓存 | roundtrip 调用生产接口，比较写入前与读回后的内容 |

代码入口：SGLang 的 `http_server.py::launch_server()`、`engine.py::_launch_subprocesses()`、
`loader.py::DefaultModelLoader.load_model()`、`scheduler.py::init_model_worker()`。

## 3. 实现方案

```text
读取配置 → 确定缓存格式 → 创建并注册真实缓存池 → 写入、查询、读回并比较
```

1. **读取配置**：识别模型与缓存类型；可选 meta 模型构造。meta 不支持时，明确记录结构未验证。
2. **确定格式**：优先复用当前 SGLang 的生产建池方法，在 CPU 上创建小容量原生池；混合层规划和测试窗口仍按配置生成。
3. **注册缓存**：使用真实 Host Pool，调用 `UnifiedCacheStore` 的生产注册接口，检查布局和页面组件。
4. **验证存取**：填入已知测试数据，写入 Posix，查询命中边界，读到独立位置，逐组件比较字节内容。

测试数值可以随机生成，但池类型、数据格式和恢复规则必须真实匹配模型。
只检查 v2 方法存在，或用普通 KV 池替代专用池，不能判定适配成功。

### 覆盖方式

| 缓存类型 | 重点验证 |
| --- | --- |
| MHA/GQA/MQA、MLA | 主 KV 的布局、页面大小和数据恢复 |
| SWA | 实际层划分、尾窗页数及缺页时的恢复边界 |
| Mamba/循环状态 | temporal、conv 等异构组件及状态快照恢复 |
| 纯 SWA、纯循环状态 | 逻辑主池不存数据，必须检查所有必要附加池 |
| DSA/indexer、MiniMax sparse | 原生索引池、scale，或集成在主池中的索引数据 |
| DeepSeek V4 | 实际启用的压缩池、索引、scale、RoPE、状态池及各自页面单位 |

复杂池清单来自 SGLang 原生对象和装配逻辑，不使用模型名称白名单，也不把所有枚举成员当作必需池。
未知版本、特殊组合或依赖真实设备才能构造的类型，返回具体的“无法判定”原因。

### 必须遵守的规则

- 主干 UCM 接受 `page_first`、`page_first_direct`；页面组件大小必须符合生产适配要求。
- 连续页策略检查完整前缀；尾窗策略只检查恢复位置需要的窗口或状态。
- 多池取共同可恢复位置，不能只取各池最大边界的最小值。例如 `{4, 8}` 与 `{6}` 没有共同边界。
- 各池按自己的页大小生成索引，不能把 Mamba 状态槽或 C128 物理页当成主 KV token。
- 设置内存预算和超时，使用独立测试目录；主干 `clear()` 不负责删除对象，残留应明确报告。

生产适配依据：`ucm/integration/sglang/unifiedcache_store.py` 和 `ucm_connector.py`。

## 4. 使用方式

`inspect` 检查配置和结构，不写存储；`roundtrip` 进一步检查实际缓存注册和存取。
默认模式为 `inspect`，建议显式指定 `--mode` 便于区分；提供存储目录不会自动切换模式。

```bash
# 配置、结构检查
ucm-toolkit run sglang-model-check --model /models/model --mode inspect

# Host 存储验证：其余参数采用默认值或自动选择
ucm-toolkit run sglang-model-check --model /models/model \
  --mode roundtrip --storage-backends /mnt/ucm-test

# DeepSeek V4 / 昇腾示例
ucm-toolkit run sglang-model-check --model /models/deepseek-v4 \
  --platform ascend --mode roundtrip --page-size 256 --c128-page-size 16 \
  --max-host-mib 2048 --storage-backends /mnt/ucm-dsv4-test --verbose
```

### 配置参数

以下为当前检测命令的参数，默认值以参数定义为准。
常用只需配置 `--model`、`--mode`，roundtrip 再提供 `--storage-backends`。
平台、模型/缓存 dtype 和主页面大小默认自动选择；布局默认 `page_first`，
测试页数为 2，Host 缓冲预算为 512 MiB。默认页数不保证覆盖所有稀疏边界，
需要时使用 `--pages 5` 等高级选项。验证特定服务时应与其非默认配置保持一致。
`--help` 仅展示常用参数，`--help-all` 展示全部高级选项；参数功能均保留。

| 参数 | 默认值 | 用途与限制 |
| --- | --- | --- |
| --model | 必填 | 本地模型目录或 Hugging Face 模型 ID |
| --mode | inspect | inspect 检查配置和结构；roundtrip 增加缓存存取测试 |
| --dtype | auto | 模型参数的数据类型，由 SGLang 解析 |
| --skip-meta-model | 关闭 | 跳过 meta 构造；报告注明模型结构未验证 |
| --trust-remote-code / --no-trust-remote-code | 开启 | 是否允许加载模型自定义代码 |
| --platform | auto | auto、cuda、rocm、ascend、xpu、cpu |
| --device-id | 单 rank 为 0；多 rank 自动选择 | 多 rank 默认使用继承的可见设备列表前 TP×PP 张，未设置时使用 0..TP×PP-1；可显式传逗号分隔的设备列表 |
| --tp-size / --pp-size | 1 / 1 | 单机 MHA/MLA 及常规 SWA/Mamba 混合池的 TP/PP 检测；大于 1 时自动通过 torchrun 启动，要求 roundtrip 且不跳过 meta 构造 |
| --distributed-timeout | 300 | 多 rank worker 和通信超时，单位秒 |
| --layout | page_first | UCM 接受 page_first、page_first_direct |
| --page-size | 自动选择 | 主缓存页大小；显式指定时必须大于 0 |
| --pages | 2 | 测试页数，必须大于 0；建议 5 页覆盖多个状态边界 |
| --max-host-mib | 512 | Host 测试内存预算，单位 MiB；必须大于 0，不是进程内存硬限制 |
| --kv-cache-dtype | auto | auto、bfloat16、float16、float32、fp8_e4m3；显式指定目前要求原生缓存描述路径 |
| --dsv4-attn-backend | auto | DeepSeek V4 后端：auto、flashmla、flashmla_mega、trtllm |
| --fp4-indexer | 关闭 | 检测 DeepSeek V4 FP4 indexer 的数据和 scale |
| --c128-page-size | 16 | NPU C128 物理页大小，必须是正的 16 倍数 |
| --storage-backends | 无 | 存储目录，多个用冒号分隔；roundtrip 必填，使用专用测试目录 |
| --connector-name | UcmPipelineStore | UCM connector 名称 |
| --connector-module-path | 无 | 自定义 connector 模块路径 |
| --output | 无 | 同时将 JSON 结果写入指定文件 |
| --verbose | 关闭 | 输出详细检测信息和异常堆栈 |
| -h / --help | — | 显示常用参数帮助 |
| --help-all | — | 显示全部参数，包括高级和模型专用选项 |

模型 dtype 和缓存 dtype 分别配置；后端、量化参数及相关 SGLang 环境变量应与计划部署一致。
默认仍为单 rank。Linux 上 MHA/MLA 及常规 SWA/Mamba 混合模型可增加 `--tp-size 2 --pp-size 2`，
验证真实 NCCL/HCCL 通信、meta 模型的阶段层范围、各 rank 缓存池及 UCM 回环。
MLA 按 SGLang writer 规则写入，但所有 rank 读回；测试内存预算按每个 worker 计算。
混合池按 PP 阶段过滤层，按 TP 上下文获得本地形状，逐池遵循 writer 规则并读回比较。
MLA 主 KV 与 Mamba 状态可能有不同的复制/分片语义；旧版缺少必要 writer API 时返回无法判定。
原生 DSA/indexer、DSA＋Mamba、DeepSeek V4 和已支持的 MiniMax sparse 布局也复用原生池构造器，
按真实 PP 范围和 TP 上下文建池。保留全模型压缩配置，由 SGLang 发现本阶段实际物理区域；
原生 API 缺失、捕获了其他阶段的数据或不支持的共享布局时返回无法判定。
DeepSeek V4 按 SGLang 的复制型 MLA 规则处理 writer/key，不按其 MHA 标签推断分片。
本阶段固定关闭 DP/CP/DCP，MLA SWA 仍需专用描述；不执行真实 forward 或设备缓存拷贝。
真实设备测试通过前，不应仅凭离线单元测试宣布目标环境支持。

## 5. 验证边界与结果

| 可以验证 | 仍需另外验证 |
| --- | --- |
| 配置识别、可执行的 meta 构造 | 权重完整性、真实 forward、量化算子 |
| 真实 Host Pool 注册和 Posix 字节存取 | CUDA/NPU 设备容量及 L1↔L2 拷贝 |
| 给定页面/窗口描述下的多池查询交集 | 引擎生成 transfer、原始 token 到压缩页的映射 |
| 被测配置的页面组件与物理格式 | prefetch、write-through、layerwise 调度和跨 rank 共识 |

结果分为 `compatible`（被测范围通过）、`incompatible`（发现明确冲突或数据错误）、
`inconclusive`（证据不足）。混合模型仅通过 inspect，不能宣布存储适配成功。

报告应记录版本、平台、实际池清单、每池页大小、dtype、比较字节数及未验证项。
模拟存储单元测试与目标环境的真实 SGLang/Posix 集成测试必须分别记录。

## 6. 验收重点

1. 不启动服务、不加载权重；生产 SGLang 和 UCM 适配逻辑保持原样。
2. 覆盖完整命中、缺页、稀疏状态边界和逻辑主池场景。
3. 源与目标位置独立，所有必需组件读回后逐字节一致。
4. 不支持的布局、组件格式及版本给出明确原因，不把“未验证”当成“不兼容”。
5. 通过结论限定在实际执行的结构、通信和存储范围，不扩展为真实多卡推理可用性证明。

## 7. 文件职责

核心实现位于 `toolkit/ucm_toolkit/tools/sglang_model_check/`。

| 文件 | 主要职责 |
| --- | --- |
| `adapter.py` | 命令行入口：解析和校验参数，设置子进程环境；单 rank 启动 runner，多 rank 通过 torchrun 启动 distributed |
| `config.py` | 统一保存模型、模式、页面、内存预算、TP/PP 等配置，并传递给检测子进程 |
| `runner.py` | 单 rank 主流程：读取配置、meta 构造、创建和注册 Host 池，执行 UCM 写入、查询、读回和数据比较 |
| `compatibility.py` | 将模型特征转换为池和 v1/v2 接口需求，检查 UCM 接口与布局；接口存在不等于适配验证通过 |
| `pool_probe.py` | 规划普通混合池的层划分、几何和尾窗/checkpoint；计算测试持久化位置与预期可恢复前缀，作为查询结果的对照 |
| `native_probe.py` | 构造 DSA、DeepSeek V4、MiniMax sparse 等原生 CPU 池，复用真实 buffer 描述和 Host builder，管理内存预算与资源释放 |
| `distributed.py` | 初始化真实 TP/PP 通信组，验证通信、阶段层划分、各 rank 存储读写和查询边界 MIN 共识，汇总结果 |
| `result.py` | 定义报告状态、错误码和验证范围，生成默认精简输出或 verbose 详细输出 |
| `sglang_compat/inspector.py` | 集中兼容 SGLang 版本差异，提取模型/混合结构信息，提供离线构造所需的运行时配置上下文 |
| `__init__.py`、`sglang_compat/__init__.py` | 导出工具类和兼容辅助函数，不承担检测流程 |

职责划分：`pool_probe.py` 负责“需要哪些池、预期结果是什么”，`native_probe.py`
负责“利用 SGLang 原生实现构造专用池”，`runner.py` / `distributed.py` 负责执行验证。

DSA（含 DSA＋Mamba 的注意力子池）优先复用 SGLang 的 `prepare_host_pool_config`
与 `build_host_pool_group`，由原生声明完成校验、依赖排序和层映射；检测器仅提供
小容量 CPU 主 Host 池并关闭 indexer pinning。旧版使用各声明的 builder，verbose
报告中的 `native_provider.host_pool_assembly` 标明实际路径。
普通 SWA/Mamba 和 DeepSeek V4 保留现有规划/区域装配路径，不模拟完整 HiCache 控制器。

普通 MHA/MLA（含配置描述的 SWA 子池）、MiniMax sparse 和 DeepSeek V4 也复用
`KVCacheConfigurator` 的生产建池方法，由框架选择平台、格式和底层参数。
Mamba 使用原生 `MambaPool` 创建状态，再将真实状态 buffer 交给 Host 池，兼容新旧 Host 接口；
不再手工创建 temporal/conv 张量。小容量建池上下文由检测器共享提供，不运行完整配置器。
SWA/Mamba 层规划、测试 checkpoint 和 V4 Host 区域装配仍保留必要描述，不视为完整服务装配证明。

DSA 的设备缓存构造复用当前安装版本的 `KVCacheConfigurator` 方法：普通 Ascend DSA
走原生 NPU MLA 建池路径；DSA＋Mamba 在各平台均走原生混合池路径，再取出真实注意力子池。
检测器不再手写 `index_size` 等底层构造参数，也不通过丢弃未知参数掩盖接口变化。
仅提供小容量、CPU 位置和必要上下文，不执行完整 `configure()`、请求分配或 forward/transfer。
verbose 的 `native_provider.device_pool_builder` 记录实际使用的方法。
此复用消除已知的检测器参数/路径不匹配，但不承诺任意未来版本均可离线验证；
未实际验证的接口和运行时依赖仍应如实返回无法判定，不能改报为模型不兼容。

### 测试、接入与文档

| 文件 | 主要职责 |
| --- | --- |
| `toolkit/tests/test_sglang_model_check_toolkit.py` | 验证参数、工具入口、配置、能力判断和报告输出 |
| `toolkit/tests/test_sglang_model_check_pools.py` | 验证混合池规划、尾窗/checkpoint、联合查询和数据回环；包含可选真实环境集成测试 |
| `toolkit/tests/test_sglang_model_check_native.py` | 验证原生池构造与装配编排、版本兼容、内存预算和层映射；不执行真实设备 kernel |
| `toolkit/tests/test_sglang_model_check_distributed.py` | 验证 TP/PP 启动、rank 映射、分区和错误传播；包含可选真实多卡集成测试 |
| `toolkit/ucm_toolkit/registry.py` | 将检测器注册为 `ucm-toolkit run sglang-model-check` 命令 |
| 检测器目录下的 `README.md` | 使用方法、参数、覆盖范围、限制与测试方式 |
| 本文档 `requirements_analysis_zh.md` | 需求目标、实现方案、文件职责和验收边界 |
| `toolkit/README.md` | 在 toolkit 总说明中提供检测工具简介 |

单元测试与真实环境集成测试的结论必须区分。文件中涉及的结构、通信和 Host/UCM
存储检查，均不替代权重加载、推理 kernel、设备缓存传输及完整服务启动验证。
