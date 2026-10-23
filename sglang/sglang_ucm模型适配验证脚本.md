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
2. **确定格式**：普通池复用配置推导；复杂池在 CPU 上创建小容量 SGLang 原生缓存对象，获取实际布局和池清单。
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

```bash
# 配置、结构检查
ucm-toolkit run sglang-model-check --model /models/model --mode inspect

# Host 存储验证：页大小省略时按缓存类型和平台选择
ucm-toolkit run sglang-model-check --model /models/model \
  --mode roundtrip --pages 5 --max-host-mib 2048 \
  --storage-backends /mnt/ucm-test --output result.json

# DeepSeek V4 / 昇腾示例
ucm-toolkit run sglang-model-check --model /models/deepseek-v4 \
  --platform ascend --mode roundtrip --page-size 256 --c128-page-size 16 \
  --max-host-mib 2048 --storage-backends /mnt/ucm-dsv4-test --verbose
```

### 配置参数

以下为当前检测命令的参数，默认值以参数定义为准。

| 参数 | 默认值 | 用途与限制 |
| --- | --- | --- |
| --model | 必填 | 本地模型目录或 Hugging Face 模型 ID |
| --mode | inspect | inspect 检查配置和结构；roundtrip 增加缓存存取测试 |
| --dtype | auto | 模型参数的数据类型，由 SGLang 解析 |
| --skip-meta-model | 关闭 | 跳过 meta 构造；报告注明模型结构未验证 |
| --trust-remote-code / --no-trust-remote-code | 开启 | 是否允许加载模型自定义代码 |
| --platform | auto | auto、cuda、rocm、ascend、xpu、cpu |
| --device-id | 0 | 设置子进程可见设备，不代表验证多卡 |
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
| -h / --help | — | 显示命令帮助 |

模型 dtype 和缓存 dtype 分别配置；后端、量化参数及相关 SGLang 环境变量应与计划部署一致。
当前没有 TP/PP 数量或超时配置参数，检测仍以单 rank 为范围；超时属于后续执行管理需求。

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
5. 通过结论限定在实际执行的检查范围，不扩展为推理可用性或多卡适配证明。
