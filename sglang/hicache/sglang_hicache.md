# SGLang HiCache V1/V2 与 Hybrid KV Cache

> 基于当前本地 SGLang 代码整理。重点关注 L3 接口、多 Pool 数据布局，以及 L1/L2 分配与淘汰。

## 1. 先理解缓存层次

```text
Radix/Hybrid Cache：维护前缀、引用和淘汰状态
        │
        ├── node.value       → L1 Device Pool 索引
        └── node.host_value  → L2 Host Pool 索引

L1：GPU/NPU HBM，模型直接访问
L2：Host DRAM，负责缓存下沉与恢复
L3：外部存储，通过 HiCacheStorage 接口接入
```

Radix Tree 不保存真实 KV Tensor，只保存 token 前缀到 Pool 索引的映射。物理数据由 Device/Host Pool 保存。

## 2. V1 与 V2 的核心区别

| 维度 | V1 | V2 |
|---|---|---|
| 接口 | `batch_get_v1`、`batch_set_v1` | `batch_get_v2`、`batch_set_v2`、`batch_exists_v2` |
| 输入 | `keys + host_indices` | `List[PoolTransfer]` |
| Host Pool | 一个主 Pool | 多个命名 Pool |
| 适用场景 | 普通 MHA/GQA、MLA | Full+Mamba、Full+SWA、DSA、MTP 等 |
| 命中判断 | 主 KV 的连续前缀 | 所有必要 Pool 的可恢复前缀交集 |
| 返回结果 | 每个 Key 成功与否 | 每个 Pool、每个 Key 成功与否 |

核心结论：

> V1 以单一 KV Pool 为中心；V2 把一次逻辑缓存操作扩展为多个异构 Pool 的组合。

## 3. V1 使用方式

注册一个 Host Pool：

```python
storage.register_mem_pool_host(mem_pool_host)
```

写入 L3：

```python
results = storage.batch_set_v1(keys, host_indices, extra_info)
```

从 L3 加载到已分配的 L2 Page：

```python
results = storage.batch_get_v1(keys, host_indices, extra_info)
```

对应关系是：

```text
keys[i] ↔ host_indices[i] ↔ 主 Host Pool 的一个 Page
```

V1 的优点是接口简单、Page I/O 直接；局限是不能自然表达形状、粒度和生命周期不同的多个 Pool。

## 4. V2 与 PoolTransfer

V2 先按名称注册多个 Host Pool：

```python
storage.register_mem_host_pool_v2(kv_pool, PoolName.KV)
storage.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)
storage.register_mem_host_pool_v2(swa_pool, PoolName.SWA)
```

一次操作由多个 `PoolTransfer` 构成：

```python
transfers = [
    PoolTransfer(name=PoolName.KV, ...),
    PoolTransfer(name=PoolName.MAMBA, ...),
]

storage.batch_set_v2(transfers)
storage.batch_get_v2(transfers)
```

### PoolTransfer 字段

| 字段 | 含义 |
|---|---|
| `name` | Pool 类型，如 KV、Mamba、SWA、Indexer、Draft |
| `host_indices` | 数据在 L2 Host Pool 中的索引 |
| `device_indices` | 数据在 L1 Device Pool 中的索引 |
| `keys` | 数据在 L3 中的逻辑 Key |
| `hit_policy` | 该 Pool 如何参与前缀命中判断 |
| `nodes_to_load` | 本次连续传输覆盖的 Radix `NodeId` 列表，用于完成后将 Buffer 拆分回写到对应节点 |
| `indices_from_pool` | 当前 Sidecar Pool 是否复用另一个 Pool 的索引 |

`PoolTransfer` 本身不存放数据，它是一份传输说明：

```text
name            → 去哪个 Pool
device_indices  → L1 地址
host_indices    → L2 地址
keys            → L3 对象
```

### 三段数据关系

```text
L1 ↔ L2：device_indices + host_indices
L2 ↔ L3：host_indices + keys
逻辑树更新：nodes_to_load
```

## 5. V2 的命中策略

### ALL_PAGES

候选前缀内的每个 Page 都必须存在：

```text
P0  P1  P2  P3  P4
✓   ✓   ✓   ✗   ✓

连续命中只能到 P2
```

适用于：

- Full Attention KV；
- DSA Indexer；
- 与主 KV 逐页对应的数据。

### TRAILING_PAGES

只要求候选前缀尾部所需的 Page/State 存在：

```text
Full KV：P0 P1 P2 P3 P4
Mamba：               State@P4
SWA：            P3 P4
```

适用于：

- Mamba Checkpoint；
- SWA 尾部窗口；
- 只描述前缀尾部状态的数据。

最终命中不是简单比较数量，而是寻找所有必要 Pool 共同支持的最长合法前缀终点。

## 6. Hybrid 模型的多 Pool 布局

### Full Attention + Mamba

```text
L1 Device
├── Full KV Pool
│   └── 按 token/page 保存完整 Attention KV
└── Mamba Pool
    ├── Conv State
    └── Temporal/SSM State

L2 Host
├── MHA/MLA Host Pool
└── Mamba Host Pool
```

两类 Pool 必须分开，因为：

| Full Attention | Mamba |
|---|---|
| 每个 token 都产生 KV | 历史被压缩进 recurrent state |
| 按 token/page 分配 | 按请求或检查点 slot 分配 |
| 容量随前缀长度增长 | 单个 State 大小基本固定 |
| 需要连续前缀 Page | 通常只需尾部有效 checkpoint |

`MambaPool` 中 Conv 与 Temporal 是不同 Tensor，但相同 slot 共同表示一个 Mamba 状态。

### Full Attention + SWA

```text
L1 Device
├── Full KV Pool：保存完整前缀
└── SWA KV Pool：只保存最近窗口

L2 Host
├── Full Host Pool
└── SWA Host Pool
```

Full KV 和 SWA KV 使用独立物理 Pool，通过 Full→SWA Index Mapping 保持逻辑关联。

### Sidecar Pool

如果辅助数据与主 KV 一一对应，可以设置：

```python
indices_from_pool=PoolName.KV
```

它表示复用主 KV 的索引，适合 Indexer 等 Sidecar 数据。Mamba/SWA 的数量和生命周期不同，通常需要独立分配。

## 7. HostPoolGroup

Hybrid 模型的多个 Host Pool 由 `HostPoolGroup` 统一协调：

```text
HostPoolGroup
├── Anchor Pool：主 KV
├── Mamba/SWA Pool
├── Indexer Pool
└── Draft Pool
```

它负责：

- 按 Pool 名称分配和释放；
- 为独立 Side Pool 分配自己的索引；
- 处理 `indices_from_pool`；
- 任一必要 Pool 分配失败时回滚本次全部分配。

这样可防止出现“Full KV 已分配，但 Mamba/SWA 状态缺失”的不可恢复缓存。

## 8. L1 分配与普通淘汰

### 分配

```text
请求需要新 token/page
        ↓
Device Allocator.alloc()
        ↓
空间不足时触发 Prefix Cache 淘汰
        ↓
释放足够的 slot/page 后重试
```

- Full Attention：Token/Page Allocator；
- SWA：窗口/Page Allocator；
- Mamba：请求或检查点 State Slot Allocator。

### 普通 Radix 淘汰

Radix Cache 从可淘汰叶节点中选择牺牲者：

```text
叶节点
+ 引用计数为 0
+ 符合 LRU/Priority 等策略
```

选中后，将 `node.value` 对应的 Device slot/page 归还 L1 Allocator。活跃请求锁定的节点和祖先不会被淘汰。

## 9. HiCache 的 L1/L2 淘汰

### L1 已有 L2 备份

```text
释放 node.value
保留 node.host_value
节点变成 L2-only
```

后续命中时，为它重新分配 L1 Page，再从 L2 Load-back。

### L1 没有 L2 备份

- Write-through：一般直接删除未备份的 L1 叶节点；
- Write-back：尝试先分配 L2 Page 并执行 D2H，成功后释放 L1；
- L2 也没有空间时，可能彻底丢弃节点或子树。

### L2 淘汰

L2 主要淘汰：

```text
L1 已经不存在的 L2-only 节点
+ Host 引用计数为 0
+ 位于可淘汰叶集合
```

淘汰后释放 `node.host_value`，必要时删除最终的缓存树节点。

## 10. Hybrid 模型淘汰详解

### 10.1 为什么不能只用一套 LRU

Full、SWA 与 Mamba 的一次命中所消费的数据不同：

- Full Attention 会使用整条命中前缀的 KV；
- SWA 只使用当前窗口；
- Mamba 通常只使用某个前缀边界上的 checkpoint。

因此它们的热度、锁定范围和可回收时间不同，不能只用主 KV 的 LRU 代替全部状态管理。

### 10.2 Full + Mamba

Mamba Radix 节点分别保存：

```text
value              → Full KV L1 索引
mamba_value        → Mamba L1 State Slot
host_value         → Full KV L2 索引
mamba_host_value   → Mamba L2 State Slot
```

同时维护：

```text
full_lock_ref       mamba_lock_ref
Full LRU            Mamba LRU
Host Full 引用      Host Mamba 引用
```

关键不变量是：

```text
mamba_lock_ref > 0 ⇒ full_lock_ref > 0
```

原因是恢复 Mamba State 时，仍需保证与它对应的 Full 前缀边界有效。

#### Full KV 淘汰

```text
从 Full LRU 选择无锁节点
        ↓
有 L2 备份：释放 Full L1 Page，保留树和 Host 索引
无 L2 备份：Write-back 或删除相应前缀
```

#### Mamba State 淘汰

```text
从 Mamba LRU 选择无 mamba 引用的 State
        ↓
释放 mamba_value 对应的 State Slot
        ↓
保留 Full KV，或在两者均不存在时清理节点
```

一次前缀访问会刷新整段 Full 路径，但通常只刷新实际消费的 Mamba checkpoint，因此两套 LRU 顺序可能不同。

#### L2 淘汰

Full Host Page 和 Mamba Host State 分别受自己的引用计数保护。加载中的 State 不可淘汰；只有相关 Host 引用为 0 时才能释放。

删除最终节点前必须确认：

```text
Full L1 不存在
Mamba L1 不存在
Full L2 不存在
Mamba L2 不存在
没有活动引用
```

### 10.3 Full + SWA

SWA 同时存在两类回收：

#### 窗口淘汰

当请求长度超过窗口 `W`：

```text
当前位置 N
保留范围 [N-W, N]
更早的 SWA KV 可立即释放
```

这是由模型语义决定的确定性回收，不是容量 LRU。此时相同 token 的 Full KV 仍可保留。

#### 容量淘汰

SWA Radix Cache 分别维护：

```text
full_lock_ref / swa_lock_ref
Full LRU / SWA LRU
```

当 SWA KV 已被窗口机制释放，但 Radix 节点的 Full KV 仍存在时，节点通过 `swa_tombstone` 标记“逻辑节点存在、SWA 数据已释放”。

因此可能出现：

```text
Radix Node 存在
Full KV 存在
SWA KV 不存在（tombstone）
```

Full KV 的淘汰仍按前缀叶和引用计数执行；SWA KV 则可以因滑动窗口更早释放。

### 10.4 多 Pool 分配失败与淘汰一致性

Hybrid Cache 的恢复条件不是“主 KV 存在”即可，而是所有必要状态在同一个前缀边界上可用。

```text
Full KV 命中
Mamba/SWA 缺失
        ↓
不能把该边界视为完整命中
```

因此：

- 分配时，多 Pool 必须一起成功，否则回滚；
- 查询时，取各 Pool 可恢复前缀的交集；
- 淘汰时，各 Pool 可以按自身生命周期释放；
- 发布可恢复状态时，必须重新验证所需 Pool 是否齐全。

## 11. 最简记忆

```text
V1 = 一个 Key 序列 + 一个 Host Pool

V2 = 一个逻辑缓存操作 + 多个 PoolTransfer
     ├── KV
     ├── Mamba/SWA
     ├── Indexer
     └── Draft
```

```text
L1 淘汰：
Radix/Hybrid Cache 选择无引用的冷节点
→ 有 L2 备份则只释放 Device 数据
→ 无备份则 Write-back 或彻底删除

L2 淘汰：
选择无 Host 引用的 L2-only 冷节点
→ 释放 Host Pool 数据
→ 必要时删除最终树节点
```

> Hybrid 的核心不是把不同状态 Padding 成同一种 Block，而是分 Pool 保存、分别分配和淘汰，再通过 V2 在逻辑上计算共同可恢复的最长前缀。

## 12. 关键代码位置

- V1/V2 与 `PoolTransfer`：[hicache_storage.py](/Users/cici/Documents/学习/GitHub项目/sglang/python/sglang/srt/mem_cache/hicache_storage.py)
- Host 多 Pool 管理：[group.py](/Users/cici/Documents/学习/GitHub项目/sglang/python/sglang/srt/mem_cache/pool_host/group.py)
- 普通 HiCache 淘汰：[hiradix_cache.py](/Users/cici/Documents/学习/GitHub项目/sglang/python/sglang/srt/mem_cache/hiradix_cache.py)
- Mamba Hybrid 缓存：[mamba_radix_cache.py](/Users/cici/Documents/学习/GitHub项目/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py)
- SWA Hybrid 缓存：[swa_radix_cache.py](/Users/cici/Documents/学习/GitHub项目/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py)
- Device KV/Mamba Pool：[memory_pool.py](/Users/cici/Documents/学习/GitHub项目/sglang/python/sglang/srt/mem_cache/memory_pool.py)

## 13. 各类 Pool 的物理连续性

判断“是否连续”要区分三层：

1. 单个 Tensor 的 Storage 是否连续；
2. 同一个 Page/Slot 在单个 Tensor 内是否连续；
3. K/V、Conv/SSM 或多个 Side Pool 之间是否属于同一块连续内存。

通常只有前两项成立，不能据此推断多个 Tensor 在物理地址上相邻。

### 13.1 Mamba Device Pool

Mamba 保存的不是逐 token K/V，而是每个状态槽的 recurrent state：

```text
Mamba State Slot i
├── Conv State 0
├── Conv State 1（如果模型定义了多类 Conv 状态）
└── Temporal/SSM State
```

#### 默认布局

CUDA 默认路径大致为：

```text
conv_state[j]：
[num_mamba_layers, num_slots + 1, *conv_shape_j]

temporal_state：
[num_mamba_layers, num_slots + 1, *temporal_shape]
```

含义是：

- 每一种 Conv State 是一个独立 Tensor；
- SSM/Temporal State 是另一个独立 Tensor；
- 单个 Tensor 内通常连续；
- 同一 Tensor 中，一个 Layer、一个 Slot 对应的状态区域通常连续；
- Conv 与 SSM 由不同 Tensor 分配，地址不保证相邻，不能作为一整块连续内存直接搬运。

逻辑上同一个 Slot 的状态分散在：

```text
conv_state[0][layer, slot]
conv_state[1][layer, slot]
temporal_state[layer, slot]
```

因此默认模式下，一次完整 Mamba State 传输通常需要多个 Buffer/Segment。

#### Envelope/Page-major 布局

部分 CUDA 路径支持 `envelope_layout`：

```text
一个连续 uint8 _raw buffer
├── Slot 0 Envelope
│   ├── Conv[0] × 所有 Mamba Layers
│   ├── Conv[1] × 所有 Mamba Layers
│   └── SSM/Temporal × 所有 Mamba Layers
├── Slot 1 Envelope
│   ├── Conv[0] × 所有 Mamba Layers
│   ├── Conv[1] × 所有 Mamba Layers
│   └── SSM/Temporal × 所有 Mamba Layers
└── ...
```

Mamba State 的 `page_size == 1`，因此一个 Page 就是一个 Slot。对 Slot `s`：

```text
slot_start = raw_base + s × entry_bytes
slot_end   = slot_start + entry_bytes
```

`[slot_start, slot_end)` 是该 Slot 完整 Conv + SSM 状态的连续字节区间，下一个 Slot 紧接在其后。当前 builder 按各组件字节数累加 offset，并要求 offset 能被对应 dtype 对齐；不会为 Conv 和 Temporal 分配独立 Storage。

Conv 和 Temporal Tensor 此时是从同一个 `_raw` Buffer 构造出的 `torch.as_strided()` View：

```text
_raw
 ├── conv[0] view
 ├── conv[1] view
 └── temporal view
```

这些完整 View 本身通常不是 contiguous：从 `conv[0]` 的 Slot 0 跳到 Slot 1 时，需要跨过 Slot 0 中的其他 Conv 和 Temporal 区域。因此必须区分：

```text
单个 Slot Envelope   → 物理连续
整个 Conv/Temporal View → Strided，不一定 contiguous
```

其核心价值是：

- 一个 Mamba Slot/Page 有明确的连续字节 Envelope；
- 便于 Page 级分配、注册和整体传输；
- 减少跨多个独立 Tensor 组织 I/O 的复杂度。

所以结论是：

> 默认 Mamba 的 Conv 与 SSM 分开分配，不保证整体连续；只有 Envelope/Page-major 模式才显式提供跨状态组件的统一连续 Page。

NPU 和 CPU Kernel 可能调整 Conv/Temporal 的维度或转置，因此分析传输地址时必须以实际 Tensor 的 `shape`、`stride` 和 `data_ptr` 为准。

#### Conv/SSM 容量比例与 Slot 数

Conv 和 SSM/Temporal **不按固定百分比切分内存**。两者拥有相同数量的 Slot，各自占用多少字节由模型 shape、dtype 和 Mamba 层数自然决定。

设 Mamba 层数为 `L`：

```text
Conv bytes/slot
= L × Σ(prod(conv_shape[i]) × conv_dtype.itemsize)

SSM bytes/slot
= L × prod(temporal_shape) × temporal_dtype.itemsize

entry_bytes
= Conv bytes/slot + SSM bytes/slot
```

因此：

```text
Conv 比例 = Conv bytes/slot ÷ entry_bytes
SSM 比例  = SSM bytes/slot  ÷ entry_bytes
```

不会出现 Conv 分配 1,000 个 Slot、SSM 分配 2,000 个 Slot 的情况；它们必须一一对应，共同构成完整 Mamba checkpoint。

### 13.2 Mamba Host Pool

Mamba L2 当前主要使用 `page_first`/`page_first_direct`：

```text
temporal_buffer：
[slot, num_mamba_layers, 1, *temporal_shape]

conv_buffer[j]：
[slot, num_mamba_layers, 1, *conv_shape_j]
```

这样，同一种状态中一个 Slot 的所有 Mamba Layer 数据是连续或 Page 集中的：

```text
temporal_buffer[slot] → 该 Slot 的全部 SSM Layer
conv_buffer[j][slot]  → 该 Slot 的全部第 j 类 Conv Layer
```

但 `temporal_buffer` 与各 `conv_buffer[j]` 仍是独立 Tensor，不保证跨 Buffer 连续。一个完整 Mamba Host State 逻辑上属于同一个 Slot，物理上可能仍对应多个内存段。

Host Mamba Pool 的 Slot 数由 HiCache 配置决定：

- 使用 `--hicache-ratio R` 时，`host_slots ≈ device_slots × R`；
- 使用 `--hicache-size G` 时，先按各 Device Pool 的实际字节数在 Full/SWA/Mamba 之间比例切分 `G`，再用 `Mamba Host bytes ÷ entry_bytes` 计算可容纳的完整 Slot 数。

无论哪种方式，每新增一个 Host Slot，都会同时为全部 Conv 和 SSM/Temporal 状态分配空间，不会在 Mamba Pool 内再配置一个 Conv/SSM 百分比。

当这些 Host Buffer 写入 UCM L3 时，当前实现通过 `get_data_page(flat=True)` 将同一 Slot 的多个 Host 内存段拼成一个 `entry_bytes` 大小的连续 staging page，以一个 Mamba block 保存。

### 13.3 普通 MHA/GQA Pool

默认 NHD Device 布局：

```text
K[layer]：[slot, kv_head, head_dim]
V[layer]：[slot, kv_head, v_head_dim]
```

连续性：

- 每层 K 是独立 Tensor；
- 每层 V 是独立 Tensor；
- 同层、同 Slot 的 Head×Dim 通常连续；
- K 与 V 不保证连续；
- 不同 Layer 也不保证连续；
- 一个逻辑 KV Page 跨所有层通常对应 `2 × layer_num` 个数据区域。

HND 或 Vectorized 5D 会改变 Page、Head、Token 的物理顺序，但不会自动把所有层的 K/V 合成一个连续 Buffer。

MHA Host Pool 则由 Layout 决定连续方向：

| Host Layout | 更连续的访问方向 | 典型用途 |
|---|---|---|
| `layer_first` | 同一 Layer 的连续 token/page | 逐层 H2D |
| `page_first` | 同一 token/page 的所有 Layer | 整页下沉到 L3 |
| `page_first_direct` | 显式 Page 内的 Layer 和 token | Page 级直接 I/O |
| `page_head` | 同一 Page/Head 的 token 和 Layer | 特定 Kernel/传输后端 |

### 13.4 MLA Pool

MLA 将 latent KV 与 RoPE K 合并在同一行：

```text
kv_buffer[layer]：
[slot, 1, kv_lora_rank + qk_rope_head_dim]

单行：
[compressed latent KV | RoPE K]
```

因此同一 Layer、同一 Slot 的 MLA 数据是一个连续 combined row。它没有传统 MHA 中独立的 K Tensor 和 V Tensor；V 是 combined row 前部的逻辑视图。

但是：

- 不同 Layer 仍是独立 Tensor；
- 一个 Page 跨所有 Layer 时仍可能需要多个地址；
- Host `page_first` 布局可进一步让同一 Page 的多层 combined row 集中存放。

MLA 的主要优势是单 token 数据紧凑，且不需要在 L3 中分别维护 K/V 两个组件。

### 13.5 SWA Pool

Full+SWA 模型物理上使用两个子 Pool：

```text
SWAKVPool
├── full_kv_pool
└── swa_kv_pool
```

两者不保证连续，也不应强制连续，因为：

- Full Pool 保存完整前缀；
- SWA Pool 只保存最近窗口；
- 两者容量不同；
- SWA 数据会比 Full KV 更早释放；
- Full→SWA Mapping 只建立逻辑索引关系，不表示地址相邻。

每个子 Pool 内部仍可采用自己的 MHA NHD/HND/Page-major 布局。

### 13.6 Indexer、Scale 与 Draft Side Pool

这些辅助数据通常位于独立 Tensor 或独立 Pool：

```text
主 KV Pool
Indexer Pool
Indexer Scale Pool
Draft KV Pool
Draft SWA Pool
```

`indices_from_pool=PoolName.KV` 只代表 Sidecar 与主 KV 复用逻辑 Page 编号和生命周期，不代表两块数据在物理内存中连续。

例如：

```text
KV Page 10       → 主 KV Buffer 的第 10 页
Indexer Page 10  → Indexer Buffer 的第 10 页
```

两者编号一致，但 `data_ptr` 属于不同分配。

Draft Pool 同样通常独立保存投机模型的 KV。即使 Host Pool 为减少管理开销将 Target/Draft Layer 组织成 packed layer 维度，也应依据实际 Buffer 元数据判断物理连续性，不能仅根据相同 index 推断。

### 13.7 总结表

| Pool | 同一最小数据单元内 | 跨组件是否连续 | 分配/淘汰关系 |
|---|---|---|---|
| MHA Device | 同层单 Slot 的 K 或 V 连续 | K/V、Layer 间不保证 | 共用 token/page index |
| MLA Device | 同层单 Slot combined KV 连续 | Layer 间不保证 | 按 token/page |
| Mamba Device 默认 | 单个 Conv/SSM row 连续 | Conv 与 SSM 不连续 | 按 State Slot |
| Mamba Envelope | 一个 Slot 的 Envelope 连续 | 组件为同一 raw Buffer 的 View | 按 Envelope Page |
| SWA Hybrid | 各子 Pool 内按自身 Layout | Full/SWA 不连续 | 映射关联、独立生命周期 |
| Sidecar/Indexer | 自身 Page 内连续 | 与主 KV 不连续 | 可复用逻辑 index |
| Host page-first | 同一 Pool 的单 Page 多层数据集中 | 不同 Pool/组件不保证 | 每个 Pool 独立分配 |

最重要的判断原则：

> “使用同一个 logical index”只说明数据属于同一个逻辑缓存单元，不代表其物理地址连续；是否能够一次连续 I/O，必须检查它们是否来自同一 Storage、同一 `data_ptr` 区间且 stride 可合并。

## 14. Hybrid 组件如何统一挂树、分配与复用

### 14.1 `device_indices`、`host_indices` 与 `keys`

| 字段 | 含义 | 来源 |
|---|---|---|
| `device_indices` | L1 Device Pool 的逻辑槽位 | Device allocator 分配；分页模式下由 `page_id × page_size + offset` 展开 |
| `host_indices` | L2 Host Pool 的逻辑槽位 | 对应 Host Pool 从自己的空闲槽位中分配 |
| `keys` | L3 中 Page 的内容身份 | 由本页 token 与前一页 hash 链式计算 |

三者相互独立：

```text
keys：说明“是什么数据”，迁移后保持稳定
device_indices：说明“当前在 L1 哪里”，重新加载后可变化
host_indices：说明“当前在 L2 哪里”，重新加载后可变化
```

Full/SWA 通常一个 Key 对应 `page_size` 个 token slot；Mamba 通常是一个 checkpoint Key 对应一个 State Slot。

### 14.2 同一棵 Radix Tree 上的多组件

Radix 节点代表一段 token 前缀，真实 Tensor 仍在各 Pool 中。节点只保存组件索引与状态：

```text
Radix Node
├── component_data[FULL]  → Full L1/L2 indices
├── component_data[SWA]   → SWA L1/L2 indices，可为 tombstone
├── component_data[MAMBA] → Mamba checkpoint slot，可为空
└── hash/lock/LRU 等元数据
```

不是每个节点都有所有数据：字段存在但 `value=None` 表示该组件在 L1 不存在；`value=None、host_value!=None` 表示只有 L2 副本。

各组件的实际分布：

- **Full KV**：按 token/page 分布在整条缓存路径上，公共前缀可共享；
- **SWA**：只在恢复边界之前的尾部窗口节点上存在，更早节点可为 tombstone；
- **Mamba**：只在部分前缀边界保存完整 recurrent-state checkpoint，不是每个节点一份；活动请求另持一个可写 State Slot，命中 checkpoint 后 Copy-on-Write；
- **Indexer**：通常不是独立 Tree Component，而是 Full KV 的 Sidecar。它有独立 Device/Host Tensor，但通过 `indices_from_pool=KV` 复用 Full 的逻辑 index 和生命周期。

因此：

```text
Full/SWA = 路径上的 token 段数据
Mamba    = 部分前缀边界的状态快照
Indexer  = 与 Full Page 配套的独立物理数据
```

#### 新旧 TreeNode 的核心差异

**重点结论：旧实现并不是在一个 Hybrid 模型中同时维护 Full、SWA、Mamba 三棵 Radix Tree。**

旧实现仍然是一棵逻辑 Radix Tree，但根据模型选择不同的专用节点类：

```text
普通模型       → RadixCache       → 普通 TreeNode
Full + Mamba   → MambaRadixCache  → Mamba 专用 TreeNode
Full + SWA     → SWARadixCache    → SWA 专用 TreeNode
```

旧 Mamba/SWA 节点把混合状态直接写成专用字段：

```text
旧 Mamba TreeNode                 旧 SWA TreeNode
├── value            (Full L1)    ├── value          (Full L1)
├── host_value       (Full L2)    ├── host_value     (Full L2)
├── mamba_value      (Mamba L1)   ├── swa_tombstone
├── mamba_host_value (Mamba L2)   ├── full_lock_ref / swa_lock_ref
└── Full/Mamba 独立锁和 LRU         └── Full/SWA 独立 LRU
```

新实现将这些字段收敛到通用 `UnifiedTreeNode`：

```text
UnifiedTreeNode
├── id / parent / children / key
├── hash_value / event_hash_value
├── hit_count / priority / creation_time / last_access_time
├── component_data[FULL]
├── component_data[SWA]
├── component_data[MAMBA]
├── component_data[C128]
├── 每组件的 Device/Host LRU 指针
├── write_through_pending_id / load_back_pending_id
└── rotation_base
```

每个 `ComponentData` 统一保存：

```python
value          # L1 Device Pool indices
host_value     # L2 Host Pool indices
lock_ref       # Device 引用计数
host_lock_ref  # Host 引用计数
metadata       # 组件私有元数据
session_ref    # Session 引用
session_ids
```

因此新旧变化的本质是：

```text
多套“模型专用的单树实现”
            ↓
一套“多组件的统一单树实现”
```

Radix 树结构仍然只有一份；多的是每个组件各自的 Pool、锁、LRU 和淘汰状态。`UnifiedTreeCore` 另外维护：

```python
_node_arena: dict[NodeId, UnifiedTreeNode]
```

异步传输只需保存 `NodeId`，提交结果时再通过 `node_by_id()` 找回当前树节点。

#### `component_data` 的四种组件类型

当前 `ComponentType` 固定定义了四种树组件：

```python
FULL  = 0
SWA   = 1
MAMBA = 2
C128  = 3
```

每个 `UnifiedTreeNode` 都会创建四个 `ComponentData` 槽位，但未启用的组件以及没有数据的节点，其 `value`、`host_value` 保持为 `None`。这四项是**树上的缓存语义组件**，不等于系统只有四个物理 Pool。

| 组件 | 树上表达的内容 | 数据粒度与特点 |
|---|---|---|
| `FULL` | 主 Full Attention KV，也是基础/Anchor 组件 | 按 token/page 保存整条前缀；节点结构、主命中边界和基础 Hash 主要由它确定 |
| `SWA` | Sliding Window Attention KV | 按 token/page 保存尾部窗口；旧路径允许 tombstone，窗口可能跨多个节点 |
| `MAMBA` | Mamba recurrent-state checkpoint | 通常一个前缀边界对应一个 State Slot；不是逐 token KV，也不是每个节点都有 |
| `C128` | DeepSeek V4 等场景的 128:1 压缩 KV 组件 | 有独立的数据有效性、Host/Device 索引和淘汰需求，因此作为正式 Tree Component 管理 |

其中 `FULL` 是基础组件：

```text
BASE_COMPONENT_TYPE = ComponentType.FULL
```

SWA、Mamba、C128 都依附于 Full 所定义的 token 前缀树，但可以拥有自己的 `value`、`host_value`、锁、LRU 和淘汰状态。

`ComponentType` 与 `PoolName` 不应一一对应：

```text
ComponentType：决定是否需要独立的树状态、命中校验和淘汰语义
PoolName：决定真实数据由哪个 Device/Host Buffer 保存和传输
```

因此还可能存在以下物理 Pool：

```text
INDEXER
INDEXER_SCALE
DRAFT_KV
DRAFT_SWA
DEEPSEEK_V4_C4_INDEXER
DEEPSEEK_V4_C4_INDEXER_STATE
```

它们通常作为 Full/SWA 等组件的 Sidecar 或派生 Pool，通过 `indices_from_pool`、Layer Mapping 和 `PoolTransfer` 跟随主组件，不需要再占用一个 `component_data` 槽位。是否成为独立组件的判断标准不是“是否有独立 Tensor”，而是：

> 是否需要在每个 Radix 节点上独立记录存在性、命中合法性、锁定和淘汰生命周期。

### 14.3 Slot 固定与动态分配

固定的是：

- 每个 Full token slot、SWA token slot、Mamba State Slot 的字节大小；
- 服务启动后预留的总 KV Cache Buffer 通常固定；
- 单个长请求的 SWA 上限约为 `sliding_window`，Mamba 活动状态通常为一个 slot。

动态的是：

- 请求或 Radix 节点当前占用哪些 slot；
- slot 的分配、释放和复用；
- SWA 实际占用为 `min(sequence_length, sliding_window)`，并按 Page 对齐；
- Unified Memory Pool 中 Full、SWA、Mamba 在固定总 Buffer 内的物理容量边界。

```text
总 Buffer 固定
[ Mamba 向上增长 | SWA 浮动 | Full 向下增长 ]
```

三个 allocator 的索引空间彼此独立，同为 `slot 7` 不代表同一地址。Indexer 是例外：它可与 Full 复用编号，但访问的是另一块独立 Tensor。

### 14.4 L1/L2 容量与淘汰关系

L1 逻辑上分别由 Full、SWA、Mamba allocator 管理；新版 Unified Pool 可共享物理 Buffer。L2 使用独立 Host Pool：

```text
HostPoolGroup
├── KV Host Pool
├── SWA Host Pool
├── Mamba Host Pool
└── Indexer Host Pool
```

指定总 `hicache_size` 时，各 Host Pool 通常按对应 Device Pool 字节规模成比例切分；使用 ratio 时，分别根据对应 Device 容量扩展。

淘汰没有固定的“先 Mamba、再 SWA、最后 Full”顺序，而由发生短缺的组件触发：

```text
Full 缺空间  → 驱动 Full 淘汰
SWA 缺空间   → 驱动 SWA 淘汰
Mamba 缺 slot → 驱动 Mamba 淘汰
```

但共享物理 Buffer 和组件依赖会产生联动：Mamba 缺物理字节时，Full 可作为内存 donor；内部节点的保留优先级为 `Full > SWA > Mamba`，因此淘汰 Full 可能级联 SWA/Mamba，淘汰 SWA 可能级联 Mamba。叶节点最终删除时通常原子释放各组件资源。

### 14.5 Hybrid 命中与复用边界

Hybrid 不要求整个请求完全相同，而是寻找所有必要组件共同支持的最深公共前缀边界：

```text
有效恢复边界
= Full 连续命中
∩ SWA 尾部窗口连续
∩ Mamba checkpoint 存在
∩ 必要 Sidecar 有效
```

例如 Full 命中到 10,000，但最近可用 Mamba checkpoint 在 8,192，则可从 8,192 恢复并增量重算 1,808 token，而不是重算全部前缀。

- SWA 只需恢复点之前最后一个窗口，旧窗口缺失不影响正确性；公共前缀的尾部 SWA 可被多个请求共享，分支后窗口内容逐渐私有化；
- Mamba checkpoint 总结此前全部历史，缺失时必须回退到更早 checkpoint。checkpoint 越密，命中后重算越少，但占用的 State Slot 越多；
- Indexer 通常与 Full Page 同生共灭，因此不会单独决定另一套树路径，但缺少必要 Sidecar 时对应 Full Page 不能作为完整可恢复数据发布。

> 最终应区分“每请求运行状态”和“可共享前缀缓存”：活动请求拥有可写的 SWA 窗口/Mamba slot；请求完成后，仍有价值的数据由 Radix 节点持有并可被后续请求共享，直到被淘汰。

### 14.6 `nodes_to_load` 如何与 UnifiedTreeNode 关联

`PoolTransfer.nodes_to_load` 在 Unified Tree 路径中实际保存 `list[NodeId]`，不保存 Python node 对象：

```text
PoolTransfer.nodes_to_load
        │ NodeId
        ▼
UnifiedTreeCore._node_arena
        │ node_by_id(node_id)
        ▼
UnifiedTreeNode
```

它的准确含义是：

> 这次 `PoolTransfer` 中的连续 Buffer 按顺序覆盖了哪些树节点。

例如 Full KV 的 A、B、C 三个节点只有 L2 副本：

```python
PoolTransfer(
    name=PoolName.KV,
    host_indices=torch.cat([
        A.component_data[FULL].host_value,
        B.component_data[FULL].host_value,
        C.component_data[FULL].host_value,
    ]),
    nodes_to_load=[A.id, B.id, C.id],
)
```

H2D 传输完成后，commit 阶段按相同顺序拆分返回的 `device_indices`：

```python
offset = 0
for node_id in transfer.nodes_to_load:
    node = tree_core.node_by_id(node_id)
    size = len(node.component_data[FULL].host_value)
    node.component_data[FULL].value = (
        transfer.device_indices[offset : offset + size].clone()
    )
    offset += size
```

因此必须保证：

```text
host_indices 的拼接顺序
    = nodes_to_load 的 NodeId 顺序
    = device_indices 的拆分顺序
```

不同组件的典型粒度：

| 组件 | `nodes_to_load` 典型内容 | commit 行为 |
|---|---|---|
| Full KV | 根到命中点之间需要 load-back 的多个节点 ID | 按每个节点 `host_value` 长度切分 Device indices |
| SWA | 尾部窗口覆盖的若干节点 ID | 写回 SWA `value`，并重建 Full→SWA mapping |
| Mamba | 通常为一个 checkpoint 节点 ID | 将恢复的 State Slot 写回 Mamba `value` |

`nodes_to_load` 这个名字略窄：它也可用于 D2H backup 完成后，把连续返回的 `host_indices` 分配回多个节点。所以从语义上更接近 `covered_node_ids`。

L2↔L3 的 storage prefetch 不一定使用 `nodes_to_load`：此时 `keys + host_indices` 已足以完成存储 I/O，树节点关联可由更上层的 anchor `node_id` 和 commit 流程保持。

### 14.7 Tree 是否保存 L3 信息

**Tree 不保存 L3 数据的物理位置，只保存定位 L3 对象所需的逻辑信息和生命周期标记。**

```text
UnifiedTreeNode 保存                  Storage Backend 保存
├── hash_value                         ├── 物理 Key 到 Block 的映射
├── event_hash_value                   ├── 文件路径、对象位置或远端地址
├── external_cache_stored              ├── 对象当前是否存在
└── write_through_pending_id            └── 实际数据内容
```

L1/L2 索引直接保存在组件上：

```python
node.component_data[ct].value       # L1 Device Pool indices
node.component_data[ct].host_value  # L2 Host Pool indices
```

L3 使用 Key-Value 语义。Tree 通过 `hash_value` 生成候选 logical keys，backend 再加入模型、TP rank/size、Pool 等信息生成物理 key，具体存储位置对 Tree 不可见。

```text
Tree 候选前缀
    → hash_value / logical keys
    → batch_exists_v2()
    → Backend 查询实际 L3 对象
    → 返回各必要 Pool 共同可恢复的前缀
```

`external_cache_stored=True` 只表示该节点**曾经成功发布到外部缓存**，可用于避免重复备份和管理 write-through；它不能保证对象仍存在。L3 可能发生 TTL、容量淘汰、重启或部分 TP/Sidecar 缺失，因此真正恢复前仍需要 backend lookup。

> 最简化理解：Tree 知道“要查什么 Key”，Storage Backend 知道“数据实际存在哪里、现在是否还存在”。
