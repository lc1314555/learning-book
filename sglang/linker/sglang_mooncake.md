# SGLang 对接 Mooncake 方案分析

> 本文基于 SGLang 当前 `python/sglang/srt/mem_cache/storage/mooncake_store` 及其调用链整理，重点分析 Mooncake 接入 HiCache v1、Hybrid/HiCache v2 和 Unified Cache Direct Linker 的实现方式。

## 1. 核心结论

Mooncake 在 SGLang 中有三条 KV Cache 接入路径：

```text
HiCache v1：       L1 Device ↔ L2 HostKVCache ↔ Mooncake
Hybrid HiCache v2：L1 Device ↔ L2 HostPoolGroup ↔ Mooncake
Direct Linker：    L1 Device ↔ Mooncake
```

三条路径共用底层 `MooncakeStore` 和 `MooncakeDistributedStore`，区别主要在于：

- 谁负责缓存生命周期编排；
- 是否存在 Host Pool；
- 传输是面向单一 KV Pool，还是面向多个 component/pool；
- Mooncake 注册的是 Host buffer 还是 GPU Device buffer。

```text
MooncakeBaseStore
    └── MooncakeStore
          ├── HiCacheController 使用 v1 I/O
          ├── HybridCacheController 混合使用 v1/v2 I/O
          └── MooncakeDirectLinker 使用 v2 对象布局直接访问 Device Pool
```

## 2. 公共 Mooncake 存储层

`MooncakeStore` 继承 `HiCacheStorage`，内部持有一个 `MooncakeDistributedStore` 客户端，统一负责：

- Mooncake配置加载和客户端初始化；
- Host/GPU buffer 注册；
- object key 生成；
- `exists/get/put` 批量操作；
- 单 buffer、multi-buffer 和 range I/O；
- SSD offload、tenant、group semantics 和指标统计。

Hybrid 模型不是分别创建 v1 Store 和 v2 Store。每个 worker 通常只有一个 `MooncakeStore` 实例，同时暴露：

```python
batch_get_v1()
batch_set_v1()

batch_exists_v2()
batch_get_v2()
batch_set_v2()
```

这里的 v1/v2 是存储 I/O 接口和数据表达方式，不代表两个独立的 Mooncake服务。

## 3. HiCache v1 接入

### 3.1 架构

```text
HiRadixCache
    → HiCacheController
        ├── L1 GPU KV Pool
        ├── L2 HostKVCache
        └── L3 MooncakeStore
```

Mooncake只注册 Host Pool：

```text
GPU KV ↔ Host KV：L2TransferEngine
Host KV ↔ Mooncake：batch_get_v1 / batch_set_v1
```

### 3.2 查询与恢复

```text
L1/L2 miss
    → 根据 token page 生成 hash keys
    → Mooncake batch_exists()
    → HostKVCache 分配 pages
    → batch_get_v1(keys, host_indices)
    → Mooncake直接写入 Host Pool
    → Host→GPU 按层加载
```

`batch_get_v1()` 根据 `host_indices` 和已知 KV 布局展开 K/V 指针及大小。

### 3.3 写回

```text
GPU→Host
    → HiCache backup queue
    → batch_set_v1(keys, host_indices)
    → Mooncake
```

Mooncake传输的数据源是 Host Pool，因此写入期间不直接持有 GPU slots。

### 3.4 特点

- 适合传统 Full Attention MHA/MLA；
- Host L2 可以吸收 Mooncake延迟并提供快速命中；
- Mooncake命中需要 `Mooncake→Host→GPU` 两跳；
- 需要较大的 pinned Host DRAM；
- 对 SWA、Mamba、Draft 和 Indexer 等复合状态表达有限。

## 4. Hybrid/HiCache v2 接入

### 4.1 架构

```text
UnifiedRadixCache
    → HybridCacheController
        ├── Device Pool Group
        ├── HostPoolGroup
        │     ├── KV
        │     ├── SWA
        │     ├── MAMBA
        │     ├── DRAFT
        │     └── INDEXER / Sidecars
        └── MooncakeStore
```

数据路径仍是：

```text
Device ↔ HostPoolGroup ↔ Mooncake
```

与 v1 的关键差别是从单一 KV Pool 升级为多个具有独立布局和命中语义的 Pool。

### 4.2 一个 Store 混合使用 v1/v2

HybridCacheController 只持有一个 `MooncakeStore`：

- 主 KV anchor 通常使用 `batch_get/set_v1()`；
- SWA、Mamba、Draft、Indexer 等额外 Pool 使用 `batch_get/set_v2()`；
- 查询使用 `batch_exists_v2()` 汇总所有必要 Pool 的完整性。

对于 DeepSeek-V4 等逻辑 KV anchor，anchor 自身可能没有物理 `kv_buffer`。此时 v1 调用只返回逻辑成功，实际数据完全由 C4、C128、Indexer、State 等 v2 side pools 承载。

### 4.3 `PoolTransfer`

v2 使用 `PoolTransfer` 描述逻辑传输：

```python
PoolTransfer(
    name=PoolName.SWA,
    host_indices=...,
    device_indices=...,
    keys=...,
    hit_policy=...,
    indices_from_pool=...,
)
```

它表达：

- 数据属于哪个 Pool；
- 逻辑 page keys；
- Host/Device slots；
- 命中策略；
- 当前 Pool 是否复用另一个 Pool 的索引范围。

### 4.4 多组件命中

不同 Pool 可使用不同命中策略：

```text
ALL_PAGES：从 prefix 开头到停止点全部存在
TRAILING_PAGES：停止点之前的 sliding window 完整即可
```

因此 `batch_exists_v2()` 返回的不只是最大页数，还包括稀疏的合法恢复边界：

```python
restorable_prefix_pages = [2, 4, 7]
```

最终可恢复边界必须满足：

```text
所有必要 Pool 的边界交集
    ∩
所有 TP/CP/PP shard 的边界交集
```

### 4.5 特点

- 支持 UnifiedRadixCache 和复合模型状态；
- 支持 SWA、Mamba、Draft、Indexer 及 DeepSeek-V4 sidecars；
- 保留 Host L2 的容量和故障缓冲能力；
- 多 Pool 必须在同一合法边界完整；
- 仍需要 Mooncake→Host→GPU 两跳；
- HostPoolGroup 和多组件状态管理较复杂。

## 5. Mooncake Direct Linker：直接接 L1

### 5.1 架构

```text
UnifiedRadixCache
    → UnifiedCacheLinkerWrapper
        → MooncakeDirectLinker
            → MooncakeStore
                → Mooncake Global Store
```

数据路径为：

```text
L1 GPU Device Pool ↔ Mooncake
```

该模式不创建 HostPoolGroup，与 HiCache 配置互斥。

### 5.2 Device Pool 映射

`MooncakeDirectLinker` 初始化时构建 `DevicePoolGroup`：

```text
PoolTransfer
    → DevicePoolGroup.resolve_transfers()
    → DevicePoolEntry
    → GPU page row
    → pointer / stride / size / offset
```

每个 `DevicePoolEntry` 描述一个物理 GPU Pool：

```text
PoolName.KV      → Full KV Device Pool
PoolName.SWA     → SWA Device Pool
PoolName.DRAFT   → Draft Device Pool
PoolName.MAMBA   → Mamba State Pool
PoolName.INDEXER → Indexer Device Pool
```

`indices_from_pool` 允许 sidecar 复用 KV 的逻辑 page 范围，同时从自己的物理 Pool 取数。

### 5.3 GPU buffer 注册

Direct Linker 遍历所有 Device Pool 的底层 allocation，并直接调用：

```python
MooncakeDistributedStore.register_buffer(
    gpu_storage_ptr,
    gpu_storage_size,
)
```

共享同一 allocation 的 tensor 会去重注册。这是 Direct Linker 能绕过 Host staging 的关键。

### 5.4 Lookup

```text
UnifiedRadixCache Device miss
    → Wrapper生成各 component 的 PoolTransfer
    → DevicePoolGroup 展开为物理 Pool
    → MooncakeStore.batch_exists_v2()
    → 返回所有 Pool 共同可恢复的 prefix boundaries
    → 各并行 rank 再求交集
```

### 5.5 Layer-wise load

`load()` 只登记 pending transfer，真正传输由 `start_layer_wise_loading()` 启动：

```text
batch_get_session_start(keys)
    → layer 0：按 Pool 获取 GPU ptr/size/offset并读取
    → layer_done_counter.complete(0)
    → layer 1：读取
    → ...
    → batch_get_session_end(keys)
```

模型在使用某层 KV 前调用 `wait_until(layer)`，因此可以让 Mooncake加载后续层与当前层计算形成流水。

### 5.6 Direct offload

```text
Tree产生 BackupKV
    → Wrapper.offload_nodes()
    → MooncakeDirectLinker.offload()
    → 等待 GPU ready event
    → MooncakeStore.batch_set_v2()
    → Mooncake从 GPU buffer 直接取数
    → completion result
    → Wrapper提交 TreeNode 状态并解锁
```

`offload()` 返回成功只表示任务入队。只有最终 completion 成功，Tree 才设置：

```python
node.external_cache_stored = True
```

### 5.7 不同 Pool 的处理

Direct Linker 不会把所有 Pool 拼成一个 tensor，而是：

```text
Tree Component
    → 逻辑 PoolTransfer
    → DevicePoolGroup 选择物理 GPU Pool
    → DevicePoolEntry 生成 ptr/size/offset
    → MooncakeStore生成该 Pool 的独立 object keys
    → 按 Pool、按层 load/offload
```

一次请求可以同时包含 KV、SWA、Draft 和 Indexer 等 transfer。Lookup 时求共同恢复边界，传输时分别写入对应 GPU buffer。

## 6. Mooncake侧的数据组织

Mooncake是统一的分布式对象存储：

```text
Object Key → bytes及其副本位置
```

不同类型的数据位于同一个 Global Store，但通常使用不同 object key 独立存储，而不是拼成一个大对象。

例如基础 page hash 为 `abc123`：

```text
Full KV：
    abc123_tp0_cp0_pp0_k
    abc123_tp0_cp0_pp0_v

SWA：
    abc123_tp0_cp0_pp0_swa_k
    abc123_tp0_cp0_pp0_swa_v

Indexer：
    abc123_cp0_pp0_indexer

Draft：
    abc123_tp0_cp0_pp0_draft_k
    abc123_tp0_cp0_pp0_draft_v

Mamba：
    abc123_tp0_cp0_pp0_mamba_temporal
    abc123_tp0_cp0_pp0_mamba_conv_0
```

因此：

```text
物理系统层面：共享一个 Mooncake Global Memory Pool
逻辑对象层面：按 Pool、K/V、component 和并行 rank 分开存储
```

### 6.1 对象完整性

一个逻辑 page 可能展开成多个物理对象。只有所有必需对象都存在，该 page 才算命中：

```text
MHA Full page = K object存在 AND V object存在
Hybrid boundary = Full完整 AND SWA窗口完整 AND 必要sidecar完整
```

Mooncake对象可能被独立淘汰，因此每次恢复前都要重新执行完整性查询，不能仅依赖 Tree 中“曾经写入成功”的标记。

### 6.2 Group semantics

可选的 group semantics 可以给同一逻辑 page 的相关对象设置共同 group ID：

```text
group: sglang-hicache:abc123
    ├── Full K
    ├── Full V
    ├── SWA K
    ├── SWA V
    └── Indexer
```

它用于 group-aware metadata routing、lease 和 eviction，但不会把这些对象合并成一个 object。

## 7. 变长 Tensor 与 multi-buffer

Mooncake批量 I/O 接收逐对象的指针和长度：

```python
key_strs
buffer_ptrs
buffer_sizes
```

因此支持：

- 同一个 batch 内不同 object 使用不同 byte size；
- 一个逻辑 page 展开成多个大小不同的对象；
- 一个 object 由多个不连续 buffer segment 组成；
- 不同 layer、Pool 和 component 使用不同 tensor layout；
- Direct Linker 使用 range I/O 读取对象中特定 layer 的区域。

multi-buffer 对应：

```python
batch_put_from_multi_buffers()
batch_get_into_multi_buffers()
```

Direct Linker layer-wise load 使用：

```python
batch_get_into_multi_buffer_ranges()
```

但这种能力是“预定义布局下的异构对象尺寸”，不是任意动态 tensor：

- 普通 KV key 仍对应一个完整、对齐的逻辑 page；
- `host/device_indices` 数量必须符合 page size；
- 不支持随意将 137、211 等不同 token 数的残缺尾页作为普通 KV page；
- 同一个 key 在写入和读取时必须使用一致的 dtype、shape、layout 和大小；
- 读取方必须提前分配并注册目标 buffer。

## 8. 并行与所有权

对象 key 后缀通常编码 TP/CP/PP 维度，防止不同 shard 冲突。

对于 rank-sharded Pool：

```text
每个 rank 写自己的 object
恢复时所有必要 rank 都必须命中
```

对于 MLA 等 rank-replicated Pool：

```text
仅 owner rank（通常 TP0）实际 offload
其他 rank产生逻辑成功 completion
```

这样可以避免将相同数据重复写入 Mooncake。

## 9. 三种方案对比

| 维度 | HiCache v1 | Hybrid/HiCache v2 | Direct Linker |
|---|---|---|---|
| Tree | `HiRadixCache` | `UnifiedRadixCache` | `UnifiedRadixCache` |
| 编排者 | `HiCacheController` | `HybridCacheController` | `UnifiedCacheLinkerWrapper` |
| 数据路径 | GPU↔Host↔Mooncake | GPU↔Host↔Mooncake | GPU↔Mooncake |
| Host Pool | 单一 KV Pool | 多 Pool Group | 无 |
| 传输描述 | keys + host indices | `PoolTransfer` | `PoolTransfer` |
| 注册内存 | Host buffer | 多个 Host buffers | GPU buffers |
| 主 KV I/O | v1 | v1或逻辑 anchor | v2 |
| Sidecar I/O | 有限 | v2 | v2 |
| Host L2 hit | 支持 | 支持 | 不支持 |
| Mooncake hit | 两跳 | 两跳 | 一跳 |
| Host内存成本 | 较高 | 更高 | 无额外 Host Pool |
| 复合模型 | 有限 | 最完整 | 受 Linker component 能力限制 |
| 故障缓冲 | Host可缓冲 | Host可缓冲 | 更依赖 Mooncake可用性 |

## 10. 选型建议

### HiCache v1

适用于传统模型和已有 HiRadixCache 部署。新模型不建议继续围绕 v1 扩展。

### Hybrid/HiCache v2

适用于：

- UnifiedRadixCache；
- SWA、Mamba、Draft、Indexer 等混合状态；
- Host DRAM充足；
- 需要 Host L2 快速命中和故障缓冲。

### Direct Linker

适用于：

- 希望降低 Host DRAM 占用；
- Mooncake网络和全局内存足够快；
- 希望避免 Mooncake→Host→GPU 两跳；
- 相关模型 component 已被 Linker 支持。

需要注意：Direct Linker 没有 Host L2 缓冲，GPU buffer 注册、load 失败处理、请求取消和 Tree 回滚也更复杂。

## 11. 最简记忆

```text
HiCache v1
    = Mooncake作为单 KV Host Cache 的 L3

Hybrid HiCache v2
    = Mooncake作为多组件 HostPoolGroup 的 L3

Mooncake Direct Linker
    = Mooncake绕过 Host Pool，直接连接 UnifiedRadixCache 的 L1 Device Pools
```

```text
不同类型 KV：
    共用一个 Mooncake Global Store
    但按 Pool、component、K/V 和 rank 使用独立 object key

变长能力：
    支持不同 object 的 byte size 和 multi-buffer
    不等于支持任意 token 数的非对齐 KV page
```
