# vLLM PCP / DCP 特性与 UCM 适配分析

## PCP、DCP 是什么？

大模型推理通常分两步：**Prefill（预填充）**一次处理输入提示词，并生成历史 token 的 KV Cache（注意力计算需要复用的 Key/Value 张量）；**Decode（解码）**利用历史 KV，逐步生成后续 token。Context Parallel（上下文并行）沿 token/序列维度，让多张卡分担计算或缓存。

- **PCP（Prefill Context Parallel，预填充上下文并行）**：把长提示词的 prefill 计算分给多张卡。例如输入有 32K token，PCP=4 时，各卡处理分配给自己的 query 片段，再通过通信完成必要同步。主要目标是缩短首 token 等待时间；实际可以采用首尾配对，并非简单连续四等分。
- **DCP（Decode Context Parallel，解码上下文并行）**：把历史 KV 按序列位置分散保存在多张卡上。生成新 token 时，各卡读取本地 KV、计算局部 attention，再合并结果。主要目标是减少每卡历史 KV 的存储和读取压力；它也影响带历史缓存的 prefill。

可以概括为：**PCP 主要分担输入阶段的计算，DCP 主要分担历史 KV 的存储与访问。** 当前实现中，PCP 增加计算进程，DCP 复用已有 TP/PCP 进程；叠加时不能直接相乘来计算 KV 分片数。

## 1. 范围与结论

本文基于 2026-11-16 的本地工作区静态检查，未进行多卡运行验证。“当前拓扑”指本地代码，不表示已核对远端最新版本。

| 仓库 | 路径 | HEAD 基线 |
| --- | --- | --- |
| vLLM | `C:\github项目\vllm` | `3494bb04178506d66d62c1e522dd97b73f7555f4` |
| Ascend | `C:\github项目\vllm-ascend` | `9e3cc3dc223e91b7892edc787f1e15e47d007ff0` |
| UCM | `C:\github项目\lc-unified\unified-cache-management` | `d7c453780c3ce2491eeb825d9f6f1721003256f9` |

工作区可能存在未提交修改。核心结论：UCM 已有专用 CP Connector，但仍将 PCP×DCP 当作独立序列分片维度。当前引擎的 PCP 扩展计算，DCP 决定序列 KV 分片，两者可复用同一批进程。适配应转向真实 KV 所有权描述，而非只修改并行度乘积。

## 2. PCP 与 DCP 特性

| 项目 | PCP | DCP |
| --- | --- | --- |
| 全称 | Prefill Context Parallel | Decode Context Parallel |
| 目的 | 分摊长输入 prefill 计算，降低 TTFT | 分摊历史 KV 容量、访问和 attention 计算 |
| 进程数 | 增加 | 不增加，复用 TP/PCP 进程 |
| 序列 KV 分片 | 自身不增加 | 由实际 DCP group 决定 |
| 主要代价 | token 分配、恢复顺序、缓存同步 | 查询或局部结果通信与合并 |

### 2.1 PCP 的计算与缓存

Ascend 使用首尾配对 DualChunkSwap 平衡 causal attention：输入切为 2P 个 chunk，rank r 处理 r 和 2P−1−r。PCP=4 时各 rank 分别处理 (0,7)、(1,6)、(2,5)、(3,4)。

PCP>1、DCP=1 时，各 PCP rank 保留其 TP 本地 KV heads 对应的完整序列缓存。不能根据局部 query 片段推断其只保存局部 KV。

PCP-only decode 可按请求轮转分配 query owner，缓存更新跨 PCP 同步。无本地 query 的 rank 仍可能参与通信。全局请求顺序、局部 query 顺序和 gathered KV 顺序必须区分。

### 2.2 DCP 缓存布局

DCP 沿序列分片 KV，各 rank 计算局部 attention，再合并结果和 softmax 统计量。cached/chunked prefill 同样需要访问分布式历史 KV。

设物理 block 容量 B，DCP 大小 D，interleave 大小 I。标准 DCP attention 虚拟块覆盖 B×D 个 token：

```text
virtual_block = token_position // (B * D)
offset        = token_position %  (B * D)
owner_rank    = (offset // I) % D
local_offset  = (offset // (I * D)) * I + offset % I
```

B 应能被 I 整除。I=1 为逐 token 交错，I=B 为逐 block 交错。PCP 不额外放大虚拟块。indexer、窗口、压缩状态等应读取各自实际 spec，不能直接套此规则。

### 2.3 当前拓扑

worker 数按 DP×PP×PCP×TP 扩展，不再乘 DCP。

| 配置 | DCP 组织 | 说明 |
| --- | --- | --- |
| PCP=1 | TP 内分组 | 通用配置要求 TP%DCP=0，后端可能更严格 |
| PCP>1、DCP=1 | 无序列 DCP 分片 | PCP 间存在缓存副本 |
| PCP>1、DCP=PCP | 沿 PCP 轴 | 同组具有相同 TP head 分区 |
| PCP>1、DCP=TP×PCP | 覆盖完整二维网格 | Ascend MLA/SFA 支持，不能泛化到所有模型 |

TP=2、PCP=2 示例：

```text
                TP坐标0  TP坐标1
PCP坐标0           0        1
PCP坐标1           2        3
TP组：             [0,1]、[2,3]
PCP组：            [0,2]、[1,3]
DCP=2：            [0,2]、[1,3]
DCP=4：            [0,2,1,3]（逻辑DCP顺序）
```

应直接读取实际通信组 rank。当前 Ascend GQA 堆叠仅支持 DCP=PCP，其专用 GQA DCP 已进入弃用流程；DSA/V4.1 PCP 使用复制缓存，不能据此宣称支持 DCP 堆叠。引擎支持某组合不代表 UCM 已支持。

## 3. UCM 当前适配

主要实现：`ucm/integration/vllm/ucm_connector.py`。

1. 普通分支在 PCP×DCP>1 时选择 UCMCPConnector；FAWA、命中率模拟等分支有更高优先级。
2. 初始化使用 `cp_world_size=PCP*DCP`、`current_rank=DCP*pcp_rank+dcp_rank`，并将 block_size 乘 cp_world_size。
3. worker 绑定元数据时，对 UCM block ID 使用 `[current_rank::cp_world_size]` 切片。
4. TP rank 按 DCP 做除法，非 MLA 构造哈希时临时将 TP size 除以 DCP。哈希元数据包含 DCP size。
5. CP Connector 默认支持 layerwise，也保留整块读写路径。
6. `_dcp_replicated_scale` 已识别部分 replicated tensor：一个逻辑块覆盖 D 个物理行时，同时放大拷贝大小与 block stride。SFA indexer 等有对应补充。

显式限制：CP Connector 拒绝 PP>1，禁止启用现有 rank consistency manager；Ascend V4.1 FAWA 路径要求 PP=PCP=DCP=1。

## 4. 现存问题

| 问题 | 原因 | 可能影响 |
| --- | --- | --- |
| PCP-only 被当作分片 | 实际复制 KV，却按 PCP 切块 | 错误命中粒度或读写归属 |
| 堆叠重复计数 | DCP 已跨 PCP，仍乘 PCP | 逻辑 token 范围及对象数量不符 |
| 联合 rank 失真 | PCP 与 DCP rank 不再独立 | block 切片不代表实际 owner |
| TP 折算不通用 | DCP 不一定切分 TP 轴 | 哈希标识与 head 内容错配 |
| 组件布局差异 | KV 分片而 indexer 等复制 | 部分恢复、错误拷贝范围 |
| 请求视图不一致 | PCP 重排 query 和 decode owner | 元数据绑定到错误请求或区间 |
| CP 一致性缺口 | 专用路径禁用现有管理器 | 需要重新保证完整命中与失败回退 |
| 格式隔离需审计 | 有 DCP 字段不等于覆盖全部布局差异 | interleave、dtype、组件变化可能误复用 |

以上运行后果为静态风险推断，未实验复现。纯 DCP 下乘积退化为 DCP，基础假设较接近当前实现，但仍需验证模型布局和生命周期。

## 5. 新拓扑下的调整方案

### 5.1 描述真实缓存所有权

建立统一描述，包含实际 DCP group/rank、KV head 分区、内容相同的副本组、逻辑 token 范围、物理布局、interleave、dtype、组件类型、stage/layer。MLA latent 不能机械套普通 head 分片。

删除统一的 PCP×DCP 分片和 TP//DCP 推导，按实际内容映射。必须同步审计 hasher、lookup、命中 token 换算、元数据绑定、worker 拷贝及所有 cp_world_size 依赖，不能只改初始化。

### 5.2 分离内容身份与物理格式

缓存身份描述模型/权重、token 前缀及实际缓存内容。直接存储设备布局时，增加足够的格式指纹，覆盖实际 head 分区、DCP、interleave、KV dtype/量化、组件、版本等；已有字段应复用。

PCP 副本数不应无条件进入内容身份，但是否可复用须检查实际组件布局。不要简单删除 DCP 字段实现跨拓扑复用。

### 5.3 用读写计划替换列表切片

```text
全局请求前缀 → 逻辑token块 → 必需组件及head/序列分片
             → UCM对象 → 本地物理块、offset、长度
```

第一阶段要求 interleave=block_size，使用块级映射，符合当前 Ascend KV transfer 配置要求。细粒度交错只有在引擎允许且确有收益时再实现 gather/scatter。

### 5.4 副本去重及加载

PCP-only 同一 TP 坐标的相同缓存可选一个副本保存；复制的 indexer 等独立判断，不能把整个 worker 一概视为副本。

先实现各副本独立读取同一对象，保证正确性；再优化为单副本读取后分发。历史命中 KV 必须在首次消费前进入所有需要的副本，不能依赖后续针对新增 token 的 attention 同步自动补齐。分发须有明确完成事件与 collective 参与规则。

### 5.5 命中与失败一致性

只有全部必需序列分片、head 分区和组件可用，才能报告连续前缀命中。应按统一 token 单位聚合可用前缀，部分 rank 命中不能放行。

增加完整性检查或完成记录：多对象写入未全部成功时不发布完整命中；加载失败统一反馈，消费前回退重算。禁止使用部分或未初始化缓存。

异步任务完成前不得复用 block；取消、抢占、重试后旧任务不得覆盖新请求；副本保存者失败应允许其他副本重试。空 query rank 仍执行必要读写和通信。

### 5.6 分层时序与复杂状态

```text
加载/分发完成 → attention或历史KV gather
本层缓存写入 → 必要同步完成 → 保存 → 发布完成
```

按后端审计 hook。fused prolog 可能先写本地行再同步，不能默认 save_kv_layer 时副本已完整。

SFA attention KV、indexer、scale 分别适配。DSA/V4.1 需确认窗口、压缩和其他必要状态是否完整恢复；仅恢复长期 KV 不足以证明能够跳过前缀计算。

PP 后续按 stage/layer 编址并聚合完整前缀，不能直接删除拒绝检查。草稿模型使用自身拓扑，不能继承目标模型的 CP 映射。

## 6. 改造后收益与边界

| 能力 | 解决问题 | 条件 |
| --- | --- | --- |
| PCP-only 正确恢复 | 复制 KV 被误当分片 | 正确副本与全局请求映射 |
| 两种 MLA/SFA 堆叠 | 重复计数、错误 TP 折算 | 引擎支持对应模型组合 |
| 同布局缓存复用 | 恢复正确本地 DCP 分片 | 完整性及格式校验 |
| 去重保存 | 副本重复外部写入 | 数据相同且已完整 |
| 单副本读取后分发 | 重复外部读取 | 分发通信成本合适 |
| 分层预取 | 读取与计算串行等待 | 正确层依赖与生命周期 |
| 组件级恢复 | 辅助状态缺失 | 独立格式与恢复协议 |
| 统一失败回退 | 部分成功误当命中 | 跨 worker 状态一致 |

组件存在 R 份相同副本时，去重保存相对全部副本独立写入，理论外部写入可降至约 1/R。只适用于复制部分，不表示整体缓存或端到端性能提升 R 倍；还取决于底层去重、通信和重叠情况。

PCP 数量变化但实际 KV 布局不变时，可以共享缓存。DCP、TP、interleave 改变通常需要 token/head 重排：可加载时转换源布局，或引入统一存储格式。两者都有缓冲与转换成本，压缩状态还需专用处理，建议后续单独设计。

## 7. 实施顺序与验证

| 阶段 | 内容 | 验收目标 |
| --- | --- | --- |
| 1 | 拓扑、布局指纹、读写映射 | 纯 PCP、纯 DCP、MLA/SFA 两种堆叠同布局正确 |
| 2 | 完整命中、失败回退、生命周期 | 缺分片不误命中，无死锁与旧任务覆盖 |
| 3 | 去重保存、分层重叠 | 外部写入减少且结果正确 |
| 4 | 加载分发、复杂状态、PP | 逐模型逐组合验证 |
| 5 | 跨 DCP/TP/interleave 转换 | 明确复用收益与成本 |

测试选择引擎允许的 TP=2/PCP=2/DCP=1、2、4 等组合，以及纯 DCP。非法组合明确报错。覆盖：

- 无外部缓存、冷缓存、热缓存的 logits/数值容差及生成结果。
- 每个 rank 的 KV token/head 内容，不能仅比较最终文本。
- chunked prefill、混合 batch、短请求、空 query rank。
- SFA indexer、scale 等辅助组件完整恢复。
- 缺失对象、单 rank 失败、取消、抢占、未完成异步保存。
- 同布局跨 PCP 副本数复用及不同格式隔离。
- 外部字节数、对象数、TTFT、吞吐、通信耗时。

首个交付建议：同布局纯 PCP、纯 DCP、MLA/SFA 两种堆叠，完整命中与失败回退，副本去重保存。复杂状态、PP、跨布局转换分阶段开放。

## 8. 源码索引

路径相对各自仓库；行号为检查时位置，后续可能变化。

| 来源 | 文件位置 | 内容 |
| --- | --- | --- |
| vLLM | `vllm/config/parallel.py:170` | PCP 定义 |
| vLLM | `vllm/config/parallel.py:454` | DCP 与 interleave |
| vLLM | `vllm/config/parallel.py:660` | 拓扑校验 |
| Ascend | `docs/source/developer_guide/Design_Documents/context_parallel.md` | 拓扑、缓存、请求生命周期 |
| Ascend | `docs/source/user_guide/feature_guide/context_parallel.md` | 模型、transfer 约束 |
| UCM | `ucm/integration/vllm/ucm_connector.py:129` | DCP 哈希字段 |
| UCM | `ucm/integration/vllm/ucm_connector.py:584` | replicated tensor 缩放 |
| UCM | `ucm/integration/vllm/ucm_connector.py:3370` | CP 初始化及限制 |
| UCM | `ucm/integration/vllm/ucm_connector.py:3445` | 元数据切片 |
| UCM | `ucm/integration/vllm/ucm_connector.py:3848` | Connector 选择 |
| UCM | `ucm/integration/vllm/hma_connector.py:1193` | Ascend V4.1 限制 |

## 9. 图解：模型数据分布与 UCM 修改前后的区别

本节补充于 2026-11-18，描述工作区中两轮改造后的逻辑。前文“现存问题”和“PP 尚不支持”等表述描述修改前基线；应结合本节理解当前代码状态。13项 CPU 回归测试已通过，但 PP+PCP+DCP 的 NPU 执行、量化/MTP组合和 UCM 外部热加载精度尚未实机验证。

### 9.1 先分清四种数据和四个维度

```text
模型权重：训练好的矩阵，推理时加载，不是请求的KV缓存
激活数据：本次计算产生的中间结果，跨PP stage传递
attention KV：请求历史token的缓存，DCP主要分片的对象
辅助缓存：SFA indexer、scale、压缩或窗口状态，可能有独立布局

PP  → 沿模型层切分，每个stage只负责部分层
TP  → 在stage内部切分权重/heads，部分张量仍复制
PCP → 增加计算rank，拆prefill query；可对特定权重进一步分片
DCP → 复用已有rank，沿历史序列分片attention KV
```

DP=1 时：总卡数=PP×TP×PCP。DCP 不额外增加卡数，也不是全部权重的额外切分倍数。

### 9.2 二维网格：行是 TP 组，列是 PCP 组

下面使用 TP2、PCP2、PP1，共4卡。编号仅为示意；实际 DCP rank 必须读取通信组，不能从全局编号猜测。

```text
                         TP方向
                     TP0        TP1
PCP坐标0              卡0        卡1     ← TP组A
PCP坐标1              卡2        卡3     ← TP组B
                      ↕          ↕
                   PCP组0      PCP组1

TP组：  [卡0,卡1]、[卡2,卡3]
PCP组： [卡0,卡2]、[卡1,卡3]
```

“PCP坐标0的一行”不叫一个 PCP 组。此前口语中的“每组 PCP 一套模型”，准确说是：不同 PCP 坐标上的 TP 组，会复制普通 TP 权重分片；支持 PCP 权重分片的模块除外。

普通 TP 权重 W 的放置：

```text
卡0：W的TP分片0          卡1：W的TP分片1
卡2：W的TP分片0（副本）  卡3：W的TP分片1（副本）
```

若某模块启用 TP×PCP 权重分片，则四卡持有四个更小分片。当前 Ascend 的 Embedding/LM Head，以及满足条件的 SFA/DSA O-projection 可以采用这类路径；量化方法和实际开关可能使其回退。MoE 专家权重还受 EP 支配，不能仅凭这张 TP 图推算 GLM 的全部权重。DCP 本身不把 W 再切一遍。

### 9.3 MHA/GQA：head 分区与 token 分区是两回事

下面假设 KV heads 足够，TP2 分为 H0、H1；实际 head 数少于并行度时可能引入 KV head 复制。图表示一种清晰的 head 分片实例，不是所有 GQA 配置。

PCP2、DCP1：计算query不同，但每个TP head分区的完整历史在PCP间复制。

```text
                  TP0                   TP1
PCP0        H0 × 全序列KV          H1 × 全序列KV
PCP1        H0 × 全序列KV          H1 × 全序列KV
              ↑同列为副本             ↑同列为副本
```

PCP2、DCP2=PCP：每列内部沿token分片，head分区保持不同。

```text
                  TP0                   TP1
PCP0        H0 × token分片S0       H1 × token分片S0
PCP1        H0 × token分片S1       H1 × token分片S1

每列S0+S1恢复对应heads的完整序列。
```

S0/S1 是交错布局的示意，不一定为连续前半/后半段。当前 Ascend GQA 堆叠仅支持 DCP=PCP，且专用 GQA DCP 已进入弃用流程；不能将 MLA/SFA 的全网格 DCP 直接用于 GQA。

修改后的 UCM 对非 MLA 使用独立 head namespace：PCP 堆叠且 DCP=PCP 时保留 TP head 身份，而不是错误地做 TP/DCP。注意：`is_mla` 来自引擎配置，模型名字中包含 MoE/DSA 不等于 UCM 一定按非 MLA 分支处理。

### 9.4 MLA：latent KV 通常不按 query heads 切分

MLA 的历史缓存主要是压缩 latent 及相关位置分量，不能照普通 MHA 将其理解为不同 query heads 的独立 KV。

TP2、PCP2 的典型分布：

```text
DCP1：
卡0：完整latent KV      卡1：完整latent KV
卡2：完整latent KV      卡3：完整latent KV

DCP2=PCP：
卡0：latent序列S0       卡1：latent序列S0
卡2：latent序列S1       卡3：latent序列S1
     两列DCP组之间可能保留相同latent分片

DCP4=TP×PCP：
卡0：S[r0]              卡1：S[r2]
卡2：S[r1]              卡3：S[r3]
     四个实际DCP rank共同覆盖完整历史
```

这里 r0～r3 表示按 PCP 优先逻辑顺序展示的 DCP 位置，实际运行以初始化组顺序为准。当前引擎可能为 collective 顺序调整 rank 排列。

UCM 对 MLA 使用统一 latent namespace，再通过 DCP rank 映射外部块。新的 CP 保存逻辑不再使用“TP副本轮流只保存部分块”的规则：DCP rank 已拥有不同序列内容，不能再丢掉其中部分块。相同副本可能重复写同一键，专用副本去重和leader分发尚未实现。

### 9.5 GLM SFA：大 KV 分片，小 indexer 仍复制

你当前日志的后端为 ASCEND_SFA/ASCEND_SFA_INDEXER，使用 replicated indexer。以 TP2、PCP2、DCP4 为例：

```text
              attention KV             indexer KV
卡0          本地序列分片S[r0]          完整序列索引缓存
卡2          本地序列分片S[r1]          完整序列索引缓存
卡1          本地序列分片S[r2]          完整序列索引缓存
卡3          本地序列分片S[r3]          完整序列索引缓存

attention：各卡只读自己的历史分片，再合并结果
indexer：需要全历史视图，选择一致的稀疏位置
```

因此，“DCP消除了复制”只能用于相应的attention KV，不能用于整个缓存包。UCM 的 replicated tensor 缩放逻辑会对符合布局条件的 indexer 同时放大拷贝长度和物理步长，使一份逻辑块覆盖应有物理行；scale 等分量也需要随对应布局保存。

GLM 的 MoE 专家权重、attention 权重、indexer 缓存和 attention KV 是不同对象；不能由 replicated indexer 推断所有权重也复制。

DSA/V4.1 的压缩、窗口和循环状态则更复杂：本次组合改造未解除其专用 Connector 的限制，也不能仅恢复长期 KV 就认为完整前缀状态已经恢复。

### 9.6 PP：复制的是计算网格，不是整个模型层集合

以80层、TP2、PCP2、PP2为示意，总计8卡。层数与分配仅为示例，不代表 GLM 实际配置。

```text
PP stage0：第0～39层                PP stage1：第40～79层
             TP0    TP1                         TP0    TP1
PCP0         卡0    卡1             PCP0        卡4    卡5
PCP1         卡2    卡3             PCP1        卡6    卡7
             TP×PCP网格                        TP×PCP网格
                    └────── 中间激活 ──────→┘

stage0只持有前40层权重及这些层的KV
stage1只持有后40层权重及这些层的KV
```

DCP 在每个 stage 内建立，不能跨 PP stage 把不同层误认为同一 KV 分片。若每个 stage 的 DCP=4，则两边分别有4个序列分片，并不是一个跨8卡的 DCP8组。

普通 TP 权重仍在同一 stage 的 PCP 坐标之间复制；跨 stage 保存的是不同层。Embedding/LM Head、专家及 PCP 权重分片遵循各模块实现。

### 9.7 修改前后：TP4/PCP2/DCP8/PP2 对照

这个组合共16卡，每个 PP stage 8卡。

```text
物理拓扑（改造不改变引擎的这张图）：

stage0，负责前半模型层          stage1，负责后半模型层
PCP0 [0 1 2 3]                 PCP0 [ 8  9 10 11]
PCP1 [4 5 6 7]                 PCP1 [12 13 14 15]
内部DCP8                        内部DCP8
```

修改前：

```text
PP>1且CP启用 → Connector直接拒绝启动

即使简单删掉拒绝检查，旧CP仍认为：
序列分片数 = PCP×DCP = 2×8 = 16（实际每stage只有8份）
逻辑块范围 = B×16（实际为B×8）
owner编号  = DCP×pcp_rank+dcp_rank（把相关坐标再次叠加）
非MLA TP折算 = TP/DCP = 4//8 = 0（不合法）
```

修改后：

```text
PP stage隔离：stage0 store             stage1 store
序列分片数：      DCP8                     DCP8
逻辑块范围：      B×8                      B×8
本地归属：        实际DCP rank             实际DCP rank

调度器：等待各stage握手
        → 根据真实层范围/块字节数建立stores
        → 查询所有stage和必需head分区
        → 只报告全部完整的连续前缀
worker：只加载/保存自己stage的层和本地DCP分片
```

对于 TP4/PCP2/DCP2/PP2，分片数相应为每 stage 2，而非旧式4。非 MLA 的 TP head 分区仍为4；MLA latent 使用统一内容namespace。

### 9.8 一块缓存如何映射到外部对象

示例：B=128、DCP=4、interleave=128，一个引擎虚拟块覆盖512 token。

```text
全局token范围：       0～511
                     │
       ┌─────────────┼──────────────┐
DCP0：0～127    DCP1：128～255  DCP2：256～383  DCP3：384～511
       │                 │              │              │
UCM： K0                K1             K2             K3

worker拿到的本地物理block容量仍为128 token，
各rank的block ID数值即使相同，也指向各自设备上的不同缓存。
```

改造后的列表映射：

```text
调度器keys： [K0 K1 K2 K3 | K4 K5 K6 K7 | ...]
DCP0读取：   [K0          | K4          | ...]
DCP1读取：   [   K1       |    K5       | ...]
DCP2读取：   [      K2    |       K6    | ...]
DCP3读取：   [         K3 |          K7 | ...]
```

PCP 不再次增加该列表的每组对象数。对 replicated indexer，一份对应对象还需要包含其完整的相关物理行；上图仅画 attention KV token 分布。对象键是请求语义链生成的标识，不等于简单地对图中局部token片段单独哈希。

PP 加入后，每个 stage 在自己的目录写这些键，内容为不同层：

```text
共享Posix根目录
├─ pp_0_l<起始层>-<层数>/  K0、K1、K2、K3……（stage0层数据）
└─ pp_1_l<起始层>-<层数>/  K0、K1、K2、K3……（stage1层数据）
```

同名K不表示数据相同，必须结合stage目录和格式身份解释。

### 9.9 修改前后的命中和保存

```text
缓存查询
                 stage0: K0 K1 K2 K3 都存在
请求某前缀 ────→ stage1: K0 K1 缺K2 K3
                               ↓
修改后：该虚拟块不能算完整命中；只保留此前完整连续前缀
```

已有 PP 层本来就提供跨stage最短前缀聚合，本次不是新发明这项能力，而是让CP初始化正确接入它，并在握手前返回未命中；CP还会查询必需的非MLA head namespace。

```text
保存路径
修改前的通用MLA规则：TP副本按块位置分担保存
                     ↓ 套到DCP后可能丢掉独有序列分片
修改后的CP规则：每个worker保留全部本地DCP块进行保存
                     ↓ 正确性优先，副本可能重复写
```

本次仍使用现有任务等待/完成接口，没有新增跨卡广播和分布式事务协议。CPU测试证明映射及控制逻辑，不证明所有worker完成事件在所有后端组合下均正确。

### 9.10 当前可测试范围与尚未完成事项

| 组合 | 当前代码路径 | 验证状态 |
| --- | --- | --- |
| TP+纯DCP、PP1 | 原有CP映射基础，保留纯DCP哈希格式 | 用户已报告GLM TP16/DCP16精度正常；最新修改仍需回归 |
| TP+PCP、DCP1 | PCP副本读取完整对应缓存 | CPU映射测试通过，NPU待测 |
| TP+PCP+DCP、PP1 | DCP取实际组，不再乘PCP | CPU映射测试通过，NPU待测 |
| TP+PP、无CP | 既有layerwise PP路径，资源逻辑回归 | 本次CPU测试覆盖部分控制逻辑 |
| TP+PP+DCP | stage存储隔离+stage内DCP | 引擎允许的模型组合下，NPU待测 |
| TP+PP+PCP及三者组合 | layerwise模式接入PP握手与CP分片 | CPU组合测试通过，NPU待测 |
| PP+CP整块模式 | 明确拒绝 | 未实现 |
| FAWA/V4.1及特殊混合状态 | 保留原有专用限制 | 不因本次修改自动支持 |
| 跨TP/DCP布局复用、leader加载分发 | 未实现 | 需要另行设计 |

第一版使用共享Posix、use_layerwise=true。组合PCP+DCP要求interleave=block_size；当前Ascend传输场景也要求该设置。PCP使用Model Runner V2。不要删除模型后端自身的合法性校验来强行运行不支持组合。

推荐验证 TP4/PCP2/DCP2/PP2 和 TP4/PCP2/DCP8/PP2，先关闭UCM建立引擎基线，再使用新存储目录冷保存，重启后确认外部命中和实际加载，最后比较精度。PP层数可自动分配，手动分层时必须使用当前模型真实层数。

本节对应代码：`ucm/integration/vllm/ucm_connector.py` 中的 `CPTopology`、`UCMCPConnector`、`UCMLayerWiseConnector._configure_pp_store`、`set_xfer_handshake_metadata_pp_aware`、`_lookup_all_stages`；回归测试为 `test/test_cp_topology.py`。未新增生产拓扑文件。
