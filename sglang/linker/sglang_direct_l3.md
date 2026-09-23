# SGLang Unified Cache Linker 与 Direct L3

> 基于当前 SGLang `unified_cache/unified_cache_linker.py` 实现整理，重点关注 Direct L3 数据路径、与 HiCache 的差异，以及 TreeNode 的异步状态管理。

## 1. 核心结论

`UnifiedCacheLinker` 用于将 `UnifiedRadixCache` 的 Device Pool 直接连接到外部 L3 KV Store，不经过 L2 Host Pool：

```text
传统 HiCache： L1 Device ↔ L2 Host ↔ L3 Storage
Direct Linker： L1 Device ↔ L3 External Store
```

它不是具体的存储后端，而是 Tree 与 backend 之间的统一协议和编排层：

- `UnifiedCacheLinker`：backend 需要实现的抽象传输接口；
- `UnifiedCacheLinkerWrapper`：负责将 lookup、load、offload 接入 Unified Radix Tree 生命周期。

当前实现包括 `MooncakeDirectLinker` 和 `UMBPDirectLinker`。

## 2. 两个核心类

### 2.1 `UnifiedCacheLinker`

backend 实现以下能力：

```python
lookup()                    # 查询所有完整可恢复的前缀长度
load()                      # 将 L3→Device 任务入队
start_layer_wise_loading()  # 启动按层加载
cancel_queued_load()
num_completed_loads()
pop_completed_load()

offload()                   # 将 Device→L3 任务入队
num_completed_offloads()
pop_completed_offload()     # 返回实际写入的最终成功/失败结果

reset()
close()
```

backend 负责真实 Device↔L3 I/O、异步队列和完成通知，不负责 Radix Tree 结构。

### 2.2 `UnifiedCacheLinkerWrapper`

Wrapper 将 Direct L3 接入三个 Tree 时机：

```text
match_prefix      → match()         查询 L3 命中
init_load_back    → load_back()     分配 Device slot 并加载
BackupKV action   → offload_nodes() 将 Device 数据下沉到 L3
```

Wrapper 负责：

- 根据 token 前缀生成 page hashes；
- 调用各 Tree Component 构造 `PoolTransfer`；
- 在 TP rank 间求可恢复前缀集合的交集；
- 分配、插入和回收 Device indices；
- 在异步 I/O 期间锁定 TreeNode；
- 接收 backend 最终完成结果，更新 node 状态。

### 2.3 Wrapper 接口职责速查

这几个方法属于 `UnifiedCacheLinkerWrapper` 的 Tree 编排接口，不是 backend 直接实现的传输接口：

| 方法 | 核心功能 | 对应 backend 操作 |
|---|---|---|
| `match()` | 查询 Device 未命中部分在 L3 中可恢复的最长公共前缀，只记录命中，不搬运数据 | `lookup()` |
| `load_back()` | 分配 Device slots，将恢复路径插入 Tree，并启动 L3→Device 加载 | `load()` |
| `offload_nodes()` | 将 TreeNode 转成 `PoolTransfer`，锁定节点并启动 Device→L3 下沉 | `offload()` |
| `commit_completed_offloads()` | 消费最终写入结果，更新 `external_cache_stored`/pending 状态并解锁节点 | 消费 `pop_completed_offload()` 结果 |
| `release_request()` | 请求结束或取消时清理 hit marker，并尽可能取消尚未启动的 load | `cancel_queued_load()` |

调用关系可简记为：

```text
请求恢复：match() → load_back() → backend 完成后释放 load lock

节点下沉：offload_nodes() → backend 完成
                         → commit_completed_offloads()

请求取消：release_request()
```

其中 `load()`/`offload()` 返回成功通常只代表任务已入队；TreeNode 状态提交和解锁必须等待 backend 的最终完成通知。`release_request()` 目前只能可靠取消尚未启动的 queued load，已发布 Tree 路径和 Device slots 的完整原子回滚仍有限制。

## 3. L3 命中查询

Tree 先得到当前 L1 Device 已命中的前缀，再为未命中 tail 计算 hash：

```text
请求 token 前缀
├── Device 已命中部分
└── Device 未命中 tail
       ↓
    tail_hashes
       ↓
    component lookup PoolTransfers
       ↓
    backend.lookup()
```

`lookup()` 返回所有完整可恢复的 page 前缀长度，而不只是一个最大值：

```python
[1, 3, 5]
```

这用于表达 SWA 等 `TRAILING_PAGES` 组件的稀疏合法边界。Wrapper 将各 rank 的集合转为 0/1 mask，通过 `MIN all-reduce` 求交集，再选择所有 rank 都可恢复的最长前缀。

## 4. L3→Device 直接加载

命中后的主要流程：

```text
L3 lookup 命中
    → 为 Full/SWA 等组件分配 Device indices
    → 将命中前缀先插入 Unified Radix Tree
    → 根据 adopted_ranges 保留 Tree 实际接纳的 pages
    → 锁定插入端点
    → backend.load() 将 L3 数据直接写入 Device Pool
    → 完成后解锁
```

先插树、后 DMA 是为了处理前缀共享与并发插入：

- 另一请求可能已经插入部分 page；
- Tree split 可能改变 node 结构；
- 新分配的 indices 可能未被 Tree 全部采纳；
- 必须以 Tree 最终的 canonical indices 为准。

Tree 中已登记 Device slot 所有权，不代表 DMA 已完成。`pending_loads`、node lock、backend completion queue 和 layer done counter 共同保证模型不会过早读取该 slot。

## 5. Device→L3 直接下沉

Tree 产生 `BackupKV` action 后：

```text
BackupKV(node_ids)
    → 为各组件构造 offload PoolTransfer
    → 锁定 node
    → backend.offload()
    → 标记 write-through pending
    → backend 完成通知
    → 提交 external_cache_stored
    → 解锁 node
```

Wrapper 启用 direct linker 时会设置：

```python
tree_core.enable_external_cache_linker = True
cache.write_through_threshold = 1
```

对已成功存储或正在下沉的 node，Tree 不会再次提交：

```python
needs_offload = (
    not node.external_cache_stored
    and node.write_through_pending_id is None
)
```

## 6. HiCache 与 Direct Linker 的区别

| 维度 | 传统 HiCache / HiCache v2 | Unified Cache Linker |
|---|---|---|
| 数据路径 | L1↔L2↔L3 | L1↔L3 |
| L2 Host Pool | 核心中间层 | Direct path 可绕过 |
| `host_indices` | L2↔L3 的核心地址 | 通常不需要 |
| `device_indices` | 主要用于 L1↔L2 | 直接交给 external backend |
| Tree 接入 | CacheController / HybridCacheController | `UnifiedCacheLinkerWrapper` |
| backend 接口 | `HiCacheStorage` | `UnifiedCacheLinker` |
| 命中返回 | 连续页数或 `PoolTransferResult` | 所有可恢复前缀长度集合 |
| 异步管理 | Controller threads/queues | Linker backend 队列与 Wrapper pending 状态 |
| 典型 backend | UCM Posix、NIXL、Mooncake Storage | Mooncake Direct、UMBP Direct |

Direct Linker 不是 HiCache v2 的完全替代：两者都可以使用 `PoolTransfer` 表达多 Pool 数据，但传输路径和 Tree 编排模型不同。

## 7. Direct L3 下的 TreeNode 管理

### 7.1 Node 不保存 L3 物理地址

TreeNode 主要保存：

```python
node.component_data[ct].value   # L1 Device Pool indices
node.hash_value                 # L3 logical keys 的生成依据
node.external_cache_stored      # 曾确认成功写入 L3
node.write_through_pending_id   # 当前正在进行的 offload
```

Node 不保存：

```text
L3 文件路径
远端对象地址
Posix/Mooncake/UMBP block offset
L3 数据 pointer
```

这些信息归 backend 管理。Tree 知道“查什么 key”，backend 知道“数据存在哪里、现在是否存在”。

### 7.2 Node 状态机

```text
L1 only
value != None
stored = False
pending = None
    │ offload() 成功入队
    ▼
L1 + offload pending
value != None
stored = False
pending = ack_id
    │ backend 实际写入成功
    ▼
L1 + L3 confirmed
value != None
stored = True
pending = None
    │ L1 淘汰
    ▼
L3 only
```

L3-only 状态不要求本地 TreeNode 永久保留。节点被删除后，新请求仍可以根据 token 重新生成 hashes，查询 L3 并重建 Tree 路径。

### 7.3 任务入队成功 ≠ 实际写入成功

```python
queued = cache_linker.offload(transfers)
```

`queued=True` 只表示 backend 接受并排队任务。此时数据可能仍在 GPU、正在 DMA/网络传输或尚未落到远端：

```python
node.write_through_pending_id = ack_id
node.external_cache_stored = False
```

只有 backend 最终完成且：

```python
cache_linker.pop_completed_offload() is True
```

Tree 才会通过 `finish_external_linker_offload()` 提交：

```python
node.write_through_pending_id = None
node.external_cache_stored = True
```

实际写入失败时：

```python
node.write_through_pending_id = None
node.external_cache_stored = False  # 对新节点通常仍为 False
```

无论成功还是失败，只有消费最终完成结果后才能释放 node lock。

backend 所谓“写入成功”的持久性强度取决于具体实现：可能表示已写入远端内存、文件系统 page cache 或持久介质，不一定等价于磁盘 `fsync`。Tree 只依赖 backend 定义的完成语义。

### 7.4 异步 I/O 期间的内存安全

Load 和 offload 都会在发起前增加 node lock，完成后释放，保证：

- Device slot 不会在 DMA 中途被 allocator 复用；
- TreeNode 不会在传输中被非法淘汰；
- 模型不会在 L3→Device 完成前读取未就绪数据；
- 失败时可清理 pending 状态并安全回收资源。

### 7.5 Node split 与 pending offload

异步 offload 期间可能因新前缀插入发生 node split。新 parent 继承：

```python
external_cache_stored
write_through_pending_id
```

Tree 同时生成 `ReplaceWriteThroughOnNodeSplit` action，Wrapper 将 pending offload 的发布列表从：

```text
[old_node_id]
```

替换为：

```text
[new_parent_id, child_id]
```

这保证 I/O ack 到达时，状态会发布到 split 后的正确节点。

### 7.6 Node 长度、Page 对齐与分裂

HiCache 与 Direct Linker 共用 Tree 的分页约束：

- 请求 KV 长度可以任意，但进入 Tree 前会向下截断到完整 page；不足一个 page 的尾部仅供当前请求使用，不进入 L2/L3。
- 非根 node 的长度可以包含多个 page，但必须是 `page_size` 的整数倍；node split 也只能发生在 page 边界。
- HiCache/Linker 可在多 page node 内按最大合法命中边界 split，只恢复命中的前缀，剩余部分重新计算。
- 混合模型必须取 FULL、SWA、Mamba 以及所有 TP/CP rank 的**共同可恢复边界**，不能只取 FULL KV 的最大命中长度，也不能简单对各方最大值取 `min()`。
- Linker 按 node 管理 offload 任务、锁和完成状态；node 内部仍按 page 生成逻辑 key，并进一步展开为各 Pool/component 的物理对象。
- Mamba checkpoint 是 node 边界上的 state slot，不是普通 token page；其合法边界还可能受 `mamba_cache_chunk_size` 限制。

例如 `page_size=64`、请求长度为 `64+64+50` 时，Tree 最多接纳前 128 tokens，不会生成长度为 50 的 node；这 50 tokens 要等积累成完整 page 后才能进入共享缓存。

```text
请求 KV（任意长度）
  → 向下对齐 page_size
  → page-aligned Tree node / split
  → 选择所有 Pool、所有 rank 的最大共同恢复边界
  → node 管任务状态，page/component 管物理存储
```

### 7.7 `external_cache_stored` 不是 L3 强一致目录

```python
node.external_cache_stored = True
```

只表示当前进程曾收到该 node 成功写入 L3 的确认，不能保证对象永久存在。L3 可能由于 TTL、容量淘汰、服务重启或部分 TP/Pool 对象缺失而失效。

因此真正恢复时仍要执行：

```python
backend.lookup()
```

Tree 不会仅凭 `external_cache_stored` 就把请求判定为 L3 命中。

### 7.8 Lookup miss 不会清除 `external_cache_stored`

**当前实现中，`backend.lookup()` 返回 miss 时，只会缩短本次请求的可恢复前缀，不会将本地 node 的 `external_cache_stored` 改为 `False`。**

```text
lookup miss
    → 本次请求回退到更短前缀或重算
    → 不修改 TreeNode.external_cache_stored
```

主要原因是 lookup 查询的是 Device 未命中 tail 的 page keys，而查询结果不一定能精确映射到某一个本地 node：

- 远端 page 对应的 node 可能已删除或尚未创建；
- 一次 lookup 可能跨越多个 node；
- 一个 node 可能包含多个 page、Pool 和 TP shard；
- node 级的单一 `bool` 无法表达“部分 page/Pool/rank 缺失”。

还有一种情况根本不会查询 L3：如果 node 的 Device KV 仍存在，`match()` 会将它作为 L1 hit，lookup 只查询其后的 tail。因此对应 L3 对象即使已被 GC，Tree 也不会立即发现。

这会形成一个不影响正确性、但可能影响复用率的窗口：

```text
node 曾成功写入 L3
    → external_cache_stored=True
L3 后续 GC 该对象
    → Tree 不知道，stored 仍为 True
node 的 L1 数据仍存在
    → needs_offload=False，不会重新下沉
L1 之后也被淘汰
    → 下次 lookup miss，回退重算
```

所以当前语义应理解为：

```text
external_cache_stored=True
    = 该 node 在当前 Tree 生命周期内曾获得过成功写入/恢复确认
    ≠ 对应的所有 L3 对象此刻一定存在
```

**正确性由恢复时的 `backend.lookup()` 保证；陈旧的 `external_cache_stored=True` 最多导致错过重新 offload，不会把已 GC 数据当成有效命中直接加载。**

若要让 Tree 对 L3 GC 及时响应，需要额外的失效协议，可选方案包括：

1. backend GC 主动发送 key eviction callback；
2. 使用 backend epoch/TTL，让旧 stored 标记过期；
3. 在淘汰 L1 node 前重新验证 L3，miss 时先重新 offload；
4. 将 node 级 `bool` 升级为 page/Pool/rank 粒度的存储状态。

> 结论：当前 lookup miss 不会清除 `external_cache_stored`。它是“曾经存储成功”的去重提示，不是 L3 实时存在性目录。

## 8. 当前限制

1. **仅适用于 `UnifiedRadixCache`**：依赖 Unified Tree 的 component、node action、锁和 split 状态管理，不能直接用于旧版 `RadixCache`/`HiRadixCache`。
2. **Direct Linker 目前仅支持 FULL 和 SWA component**：Mamba 本身是 Unified Tree 的独立 `MambaComponent`，并已支持 HiCache；但其 `build_external_linker_transfer()` 尚未实现，且不在 Direct Linker 支持白名单中，因此当前标准 Linker 不能用于包含 Mamba component 的 Tree。
3. **混合 Pool 必须统一恢复**：当前受支持的 FULL/SWA 等必要数据只能选择共同可恢复的 page/node 边界。未来接通 Mamba Direct Linker 后，若不改变现有匹配协议，Mamba 也必须参与共同边界判断，不能与 FULL 分别选择复用长度。
4. **所有 TP rank 必须同时命中**：各 rank 的可恢复边界取交集；任一 rank 缺少 shard，整体就要回退到更短前缀。
5. **恢复边界可能稀疏**：SWA 等 trailing-window 状态通常只在下沉的 node boundary 上完整存在，并非命中最大页数以内的每一页都可恢复；Mamba 接入 Direct Linker 后也需要定义类似的状态快照边界。
6. **L3 状态不是强一致的**：`external_cache_stored=True` 只表示曾成功写入/恢复。L3 GC 后 Tree 不会主动获知，lookup miss 也不会清除此标记，可能错过再次 offload，但实际恢复仍由 lookup 校验，不影响正确性。
7. **取消和回滚尚不完整**：只能可靠取消尚未启动的 queued load；Tree 路径或 device slot 已发布后的原子回滚仍有 TODO。
8. **异步接口存在关联约束**：offload 完成结果按提交顺序消费，backend 需要保证或整理为 FIFO；同一 `rid` 同时只能维护一条 lookup/load 流程。
9. **不提供 Host L2 能力**：链路是 Device Pool 与 L3 直连，不包含 HiCache 的 Host 命中、容量管理及 GPU↔Host 快速换入换出。

### 8.1 Mamba component 的支持状态

Mamba 在 UnifiedRadixTree 中是独立的 `MambaComponent`，与 FULL、SWA 共用同一棵 Tree 和 node 边界，并非附属于 FULL 的普通物理 Pool。当前差异是：

```text
Unified Tree：支持 Mamba component 的状态、插入、淘汰和锁管理
HiCache：已实现 Mamba 的 Device↔Host↔L3 transfers
Direct Linker：尚未实现 Mamba transfer hook，当前禁止接入
```

虽然 `unified_cache_linker.py` 已有识别 `PoolName.MAMBA` 的加载框架，但标准 `MambaComponent.build_external_linker_transfer()` 仍会抛出“will support soon”，属于预留代码，不能视为完整支持。

### 8.2 SWA request-ring 的特殊限制

当 SWA allocator 使用 request ring 时，物理 slot 会随请求循环覆盖，无法直接按稳定的 radix page 构造 L3 传输，因此 Linker 设置：

```python
self._skip_swa = True
```

此时：

```text
FULL 等其他 Pool：仍可 lookup、load 和 offload
SWA ring：不参与 L3 查询、加载和下沉
缺失的 SWA 范围：标记为不可用/已淘汰，避免读取未初始化 slot
```

这不表示整个请求或全部前缀都立即重算，而是 **Linker 只完成非 SWA Pool 的部分 L3 恢复；需要补齐多少 SWA 状态，由后续 SWA 调度和模型计算路径决定**。因此，当前的“SWA 支持”特指可构造稳定 `PoolTransfer` 的 page-based SWA，不包含 request-ring SWA 的完整 L3 复用。

## 9. 上线时间

根据当前本地 Git 历史：

| 时间 | 提交 | 内容 |
|---|---|---|
| 2026-08-30 | `c9eb475a` | 定义 external linker cache contract |
| 2026-08-30 | `6a9366f0` | 增加 direct linker Device Pool assembly |
| 2026-08-31 | `5d92e607` | 首次加入 `unified_cache_linker.py` 核心实现 |
| 2026-09-01 | `b21000ae` | 增加 Mooncake direct backend |
| 2026-09-04 | `abed6803` | 完成端到端接入 |
| 2026-09-05 | `f1f2380d` | 增加 UMBP external linker |

首个包含该功能的版本标签为 `v0.5.19`。如果以“用户可端到端启用”为上线标准，则对应 2026-09-04 的端到端接入。

## 10. 最简记忆

```text
HiCache
    = Tree + L1 Device + L2 Host + L3 Storage

Unified Cache Linker
    = Tree + L1 Device + Direct L3 Backend
```

```text
TreeNode 不保存 L3 地址：
hash_value                → 生成 L3 keys
external_cache_stored     → 曾确认成功写入
write_through_pending_id  → 正在进行异步 offload
node lock                 → 保护 I/O 期间的 Device slot
```

> `offload() == True` 只代表任务已入队；只有 backend 返回最终完成成功，Tree 才能设置 `external_cache_stored=True`。

## 11. LMCache 接入 SGLang L1 层方案

### 11.1 当前 PR 的核心方案

当前 LMCache PR **没有使用 `UnifiedCacheLinker`**，而是新增专用子类：

```python
class LMCacheUnifiedRadixCache(UnifiedRadixCache):
    ...
```

该类直接复用 Unified Radix Tree、GPU KV Pool 和节点生命周期，并在 Tree 的查询、恢复与请求完成路径中调用 LMCache MP Connector：

```text
SGLang Scheduler
        │
        ▼
LMCacheUnifiedRadixCache
  ├── UnifiedRadixCache：管理 L1 GPU KV 与 Radix Tree
  └── UnifiedLMCacheMPConnector：执行 lookup / retrieve / store
                         │
                         ▼
                  LMCache MP Server
                    ├── CPU L1
                    └── FS 等 L2 backend
```

这里所说的“SGLang L1”是 UnifiedRadixCache 管理的 GPU Device KV。LMCache 位于它的外部，为 GPU miss 提供可恢复 KV，并保存从 GPU 下沉的数据。

相关 PR 分工：

- `sgl-project/sglang#38652`：`LMCacheUnifiedRadixCache`、缓存注册和调度接入；
- `LMCache/LMCache#4828`：LMCache 侧 `UnifiedLMCacheMPConnector`；
- `LMCache/LMCache#5041`：设计与使用文档。

### 11.2 查询与恢复

```text
请求到达
    → UnifiedRadixCache 先匹配 L1 GPU 前缀
    → GPU 未完整命中时，Connector 查询 LMCache
    → lookup 只返回可恢复长度，不立即搬运 KV
    → Scheduler 调度请求时分配 GPU slots
    → Connector 将 KV 从 LMCache 恢复到指定 slots
    → 恢复结果插入 UnifiedRadixCache
    → 已恢复 token 跳过重复 prefill
```

因此一次请求可以拆成：

```text
GPU 已命中前缀 + LMCache 可恢复前缀 + 必须重新计算的尾部
```

### 11.3 写回与节点安全

请求产生可复用 KV 后，`LMCacheUnifiedRadixCache` 将其提交给 MP Connector：

```text
锁定 TreeNode
    → GPU KV 写入 LMCache
    → LMCache Server 保存到 CPU/L2
    → 写入完成
    → 释放 TreeNode lock
```

传输期间锁定节点，避免 GPU KV slot 在 LMCache 尚未复制完成时被 UnifiedRadixCache 淘汰或复用。写入失败不影响当前推理结果，只会失去后续外部复用机会。

### 11.4 Flush 与缓存层级

当前预期语义：

```text
flush SGLang radix cache
    → 清理 L1 GPU Tree
    → 不清理 LMCache

后续相同请求
    → L1 GPU miss
    → LMCache hit
    → KV 恢复到 GPU 并重建 Tree 路径
```

因此 LMCache 可跨 GPU cache flush 和 SGLang 实例生命周期保留 KV；实际持久性取决于 LMCache Server 及其后端配置。

### 11.5 与 Unified Cache Linker 的区别

| 维度 | LMCache 当前 PR | Unified Cache Linker |
|---|---|---|
| 接入方式 | 继承 `UnifiedRadixCache` | 标准 Tree 组合 linker |
| 专用 Tree 类 | `LMCacheUnifiedRadixCache` | 不需要 |
| 外部接口 | `UnifiedLMCacheMPConnector` | `UnifiedCacheLinker` |
| 编排位置 | LMCache 子类覆盖 Tree 生命周期方法 | `UnifiedCacheLinkerWrapper` 统一编排 |
| 当前状态 | PR 正在采用 | LMCache 尚未采用 |

当前方案不是维护另一棵完全独立的 radix tree，而是继承并扩展 Unified Radix Tree；代价是 LMCache 逻辑与 Tree 生命周期耦合较深。PR 作者认为现有 Linker 接口暂不足以覆盖 LMCache 需求，因此先采用继承方案，未来再考虑迁移到 external linker。

> 最简记忆：当前 LMCache PR 是 `LMCacheUnifiedRadixCache + UnifiedLMCacheMPConnector`，不是 `UnifiedRadixCache + UnifiedCacheLinker`。
