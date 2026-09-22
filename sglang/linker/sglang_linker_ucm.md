# SGLang UnifiedCacheLinker 对接 UCM 方案

## 1. 背景与目标

当前 UCM 在 `ucm/integration/sglang` 中提供的是 SGLang HiCache L3 适配：

```text
SGLang Device Pool → HostPool（L2）→ UCM（L3）
```

现有实现依赖 `HostKVCache`、`host_indices` 和 `page_first` 布局。目标是增加 SGLang `UnifiedCacheLinker` 适配，使 UCM 可以直接连接 Device Pool：

```text
UnifiedRadixCache
    → UnifiedCacheLinkerWrapper
    → UcmSglangDirectLinker
    → UcmKVStoreBaseV1
    → UCM backend

数据路径：SGLang Device Pool ⇄ UCM/L3
```

该路径不经过 SGLang HostPool。

> 本方案基于 UCM `develop` 分支和当前 SGLang `UnifiedCacheLinker` 接口。UCM 当前 `develop` 中的 SGLang 集成仍为 HiCache v1。

## 2. 设计原则

1. 保留现有 `UnifiedCacheStore(HiCacheStorage)`，继续服务 HostPool/HiCache 路径。
2. 新增独立的 `UcmSglangDirectLinker`，不要让 HostPool connector 同时承担 Device Pool 语义。
3. SGLang 负责 Tree、component、Device Pool 布局、node lock 和状态发布。
4. UCM 负责物理 key、精确查询、Device pointer I/O、异步任务和存储后端。
5. 使用 SGLang `PoolTransfer` 作为两侧的数据描述边界。
6. 第一阶段仅支持 FULL；随后支持 SWA；Mamba 需先补齐 SGLang Linker hook。

## 3. 推荐代码结构

```text
ucm/integration/sglang/
├── unifiedcache_store.py       # 保留：HiCache HostPool 后端
├── ucm_connector.py            # 保留：HiCache connector
├── ucm_direct_linker.py        # 新增：UnifiedCacheLinker 实现
├── ucm_linker_config.py        # 新增：Direct Linker 配置和 Pool layout
├── ucm_key_builder.py          # 可选：物理 key 生成
└── layerwise_counter.py        # 新增或复用 SGLang 公共实现
```

核心类建议：

```python
class UcmSglangDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components: set[ComponentType],
        config: Optional[dict] = None,
        store_factory: Optional[Callable] = None,
    ) -> None:
        ...
```

## 4. 当前可以复用的 UCM 能力

### 4.1 Store 创建与配置

可复用：

```python
UcmConnectorFactoryV1.create_connector(name, config, module_path)
```

以及：

```text
UNIFIEDCACHE_CONFIG_FILE
kv_connector_extra_config
ucm_connector_config
```

当前 `UnifiedCacheStoreConfig.load_from_config()` 依赖 `HostKVCache.get_size_per_token()`，Direct Linker 应改为根据 Device Pool layout 显式生成配置。

### 4.2 精确批量查询

```python
store.lookup(block_ids: list[bytes]) -> list[bool]
```

该接口可以查询每个 Pool/component 的对象存在性，并计算稀疏可恢复边界。

`lookup_on_prefix()` 只能表达连续前缀，不足以支持 SWA/状态 Pool 的 `TRAILING_PAGES` 语义。

### 4.3 Device pointer I/O

```python
store.load_data(
    block_ids: list[bytes],
    shard_index: list[int],
    dst_addr: list[list[int]],
) -> Task
```

```python
store.dump_data(
    block_ids: list[bytes],
    shard_index: list[int],
    src_addr: list[list[int]],
    prerequisite_handle: int = 0,
) -> Task
```

`dst_addr/src_addr` 可以是 Device buffer 指针，因此可以直接实现 UCM/L3 与 SGLang Device Pool 之间的传输。

### 4.4 异步 Task

```python
store.check(task) -> bool
store.wait(task) -> None
```

可用于：

- layerwise load；
- 后台 offload；
- completed queue；
- 最终成功/失败状态；
- reset/close 的任务清理。

当前 HiCache connector 提交任务后立即 `wait()`，表现为同步接口。Direct Linker 应保留 Task，在后台 worker 中等待。

### 4.5 Shard 与 stream 同步

`shard_index` 可以映射 logical layer：

```text
同一个 page block_id
layer 0 → shard_index=0
layer 1 → shard_index=1
...
```

`dump_data(..., prerequisite_handle)` 可以保证模型写 Device KV 完成后再读取并下沉。

### 4.6 可借鉴的已有实现

UCM 的 `UCMLayerWiseConnector` 已实现：

- 每层提取 pointer；
- `shard_index=layer_id`；
- load/dump Task；
- 当前层等待与下一层提交；
- rank consistency；
- dump 生命周期。

它与 vLLM 强耦合，不应直接继承，但任务组织方式可以复用。

## 5. SGLang Device Pool 映射

优先复用 SGLang：

```python
resolve_hybrid_device_pool_group(
    kvcache=...,
    page_size=...,
    params=...,
    components=...,
)
```

返回 `DevicePoolGroup`，其中包含：

```text
entry_map: PoolName → DevicePoolEntry
num_layers
page_size
rank_replicated
```

`DevicePoolEntry` 已提供：

```python
get_hybrid_pool_buffer()
translate_indices()
get_page_buffer_meta(indices)
prepare_locations(indices)
get_prepared_layer_range_meta(locations, layer)
```

因此 UCM 不需要自行推断不同模型的 Device KV 布局。

## 6. Pool 与 Store Layout

当前 UCM SGLang 配置只有一组固定尺寸：

```text
tensor_size
shard_size
block_size
cache_nums
```

混合模型可能包含 KV、SWA、Mamba temporal/SSM、Mamba Conv、Draft 和 Indexer 等不同大小的对象，不能全部使用同一组尺寸。

建议新增：

```python
@dataclass
class UcmPoolLayout:
    pool_name: PoolName
    object_family: str
    component_index: int
    num_layers: int
    shard_size: int
    object_size: int
    packed: bool
    hit_policy: PoolHitPolicy
    store: UcmKVStoreBaseV1
```

规则：

- 相同对象尺寸和 layout 的 component 可以共享 store；
- 不同尺寸或不同物理格式的 component 使用独立 store；
- 每个 store 使用独立 key namespace；
- 每个 logical page 可以对应多个物理对象。

示例配置：

```yaml
sglang_linker:
  pools:
    kv:
      connector_name: Posix
      object_mode: sharded_by_layer
    swa:
      connector_name: Posix
      object_mode: sharded_by_layer
    mamba_temporal:
      connector_name: Posix
    mamba_conv:
      connector_name: Posix
```

## 7. 物理 Key 设计

物理 key 至少需要包含：

```text
model/config namespace
logical page hash
PoolName
component/object family
TP rank
CP rank
PP rank
layout/version
```

推荐形式：

```text
{namespace}/{logical_page_hash}/{pool}/{component}/
tp{tp}_cp{cp}_pp{pp}/v{layout_version}
```

建议接口：

```python
def build_object_key(
    *,
    logical_page_key: str,
    pool_name: PoolName,
    component: str,
    tp_rank: int,
    cp_rank: int,
    pp_rank: int,
    layout_version: str,
) -> bytes:
    ...
```

不同 rank 或 Pool 的数据不能只使用 logical page key，否则可能互相覆盖。

## 8. UnifiedCacheLinker 与 UCM 能力映射

| Linker 接口 | 参数/返回值 | UCM 映射 | 适配逻辑 |
|---|---|---|---|
| `lookup` | `rid, transfers → list[int]` | `store.lookup(keys)` | 展开 Pool/component keys，计算共同可恢复边界 |
| `load` | `rid, transfers → bool` | 暂不提交 Store | 解析 transfer 并保存到 `pending_loads` |
| `start_layer_wise_loading` | `→ counter_index` | `load_data + Task` | 启动当前 batch 的逐层加载 |
| 每层 load | keys、layer、ptrs | `load_data` | `shard_index=logical_layer`，目标为 Device pointer |
| `cancel_queued_load` | `rid → bool` | 暂无安全回滚 | 第一版返回 `False` |
| `num_completed_loads` | `→ int` | Queue size | 返回已完成 load batch 数 |
| `pop_completed_load` | `→ list[rid]` | Queue pop | Wrapper 根据 rid 释放 node lock |
| `offload` | `transfers → bool` | `dump_data` | 异步入队，使用 prerequisite event |
| `num_completed_offloads` | `→ int` | result queue size | 返回已完成 offload 数 |
| `pop_completed_offload` | `→ bool` | Queue pop | 所有必要 Task 成功才返回 True |
| `reset` | `→ None` | wait/join/reset | 清任务、结果和 counter |
| `close` | `→ None` | worker/store close | 停止 worker 并关闭所有 stores |

## 9. 推荐接口定义

```python
class UcmSglangDirectLinker(UnifiedCacheLinker):
    layer_done_counter: UcmLayerWiseLoadCounter

    def lookup(
        self,
        rid: str,
        transfers: list[PoolTransfer],
    ) -> list[int]: ...

    def load(
        self,
        rid: str,
        transfers: list[PoolTransfer],
    ) -> bool: ...

    def start_layer_wise_loading(self) -> int: ...

    def cancel_queued_load(self, rid: str) -> bool: ...

    def num_completed_loads(self) -> int: ...

    def pop_completed_load(self) -> list[str]: ...

    def offload(self, transfers: list[PoolTransfer]) -> bool: ...

    def num_completed_offloads(self) -> int: ...

    def pop_completed_offload(self) -> bool: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
```

内部辅助接口：

```python
def _resolve_transfers(
    self,
    transfers: list[PoolTransfer],
    *,
    allow_partial: bool,
    allow_missing_kv: bool,
) -> list[PoolTransfer]: ...
```

```python
def _build_keys(
    self,
    transfer: PoolTransfer,
) -> list[list[bytes]]:
    """logical page → physical component keys"""
```

```python
def _build_layer_io(
    self,
    transfer: PoolTransfer,
    layer: int,
) -> tuple[list[bytes], list[int], list[list[int]]]:
    """返回 block_ids、shard_indices、Device pointers。"""
```

```python
def _apply_hit_policy(
    self,
    page_exists: list[bool],
    transfer: PoolTransfer,
    candidate_pages: list[int],
) -> list[int]: ...
```

## 10. Lookup 与恢复边界

对于每个 Pool 执行 exact lookup：

```text
logical page keys
    → physical component keys
    → store.lookup()
    → 合并同一 page 的全部 component
    → 应用 PoolHitPolicy
```

`ALL_PAGES`：

```text
从第 1 页到候选边界必须连续存在
```

`TRAILING_PAGES`：

```text
对每个候选边界 end，
[end-window, end) 必须完整存在
```

最后：

```python
restorable = intersection(all_pool_candidates)
```

必须返回全部可恢复边界，而不是仅返回最大值。跨 TP/CP rank 的交集由 SGLang Wrapper 负责。

## 11. Layerwise Load 时序

```text
Wrapper.load_back(req)
    │
    ├── TreeComponent 构造 LOAD PoolTransfer
    ├── 分配 Device slots
    ├── 插入 Tree
    └── UcmDirectLinker.load(rid, transfers)
            │
            └── pending_loads[rid] = resolved_transfers

调度 batch 准备完成
    │
    └── start_layer_wise_loading()
            │
            ├── counter_index = update_producer()
            ├── 取走当前全部 pending_loads
            └── 放入 load_queue

单个 load worker
    │
    ├── logical layer 0
    │     ├── 生成 keys/shard_index/Device ptrs
    │     ├── store.load_data(...)
    │     ├── store.wait(task)
    │     └── counter.complete(index, 0)
    │
    ├── logical layer 1
    │     └── ...
    │
    └── completed_loads.put(rids)
```

第一版建议采用单 load worker，保证 batch FIFO 和 logical layer 顺序。后续再考虑多 stream 或多 worker。

`LayerWiseLoadCounter` 需要实现：

```python
update_producer() -> int
set_consumer(index: int) -> None
complete(index: int, layer: int) -> None
fail(index: int, error: BaseException) -> None
wait_until(layer: int) -> None
reset() -> None
```

## 12. Offload 时序

```text
Tree BackupKV action
    │
    └── Wrapper.offload_nodes(node_ids)
            │
            ├── 构造 OFFLOAD PoolTransfers
            ├── 加 node lock
            └── UcmDirectLinker.offload(transfers)
                    │
                    ├── resolve transfers
                    ├── 记录 compute-ready event
                    └── offload_queue.put(...)

offload worker
    │
    ├── 遍历 Pool/component/layer
    ├── 生成 physical keys
    ├── 生成 shard_index
    ├── 获取 Device pointers
    ├── store.dump_data(..., prerequisite_handle)
    ├── 等待全部 Task
    └── completed_offloads.put(all_success)
```

语义必须区分：

```text
offload() == True
    = UCM 接受并排队任务

pop_completed_offload() == True
    = 所有必要 Pool/component/layer 已实际写入成功
```

## 13. 内部任务结构

```python
@dataclass
class UcmLoadBatch:
    counter_index: int
    requests: dict[str, list[PoolTransfer]]
    ready_event: object
```

```python
@dataclass
class UcmOffloadBatch:
    transfers: list[PoolTransfer]
    prerequisite_handle: int
```

```python
@dataclass
class UcmTaskGroup:
    tasks: list[Task]
    request_ids: list[str]

    def wait(self) -> bool:
        ...
```

一个逻辑 operation 可能产生：

```text
多个 Pool × 多个 component × 多个 layer
```

任一必要 Task 失败，整个 operation 应返回失败。

## 14. 现有接口的复用边界

### 可以直接复用

- `UcmConnectorFactoryV1`；
- 配置文件读取；
- `store.lookup()`；
- `store.load_data()`；
- `store.dump_data()`；
- `Task`、`check()`、`wait()`；
- `shard_index`；
- dump prerequisite event；
- backend `close()`。

### 需要抽象后复用

- key 编码和 namespace；
- model/rank suffix；
- config construction；
- 多 component pointer grouping；
- 混合 Pool 的 v2 key expansion。

### 不能直接复用

- `register_mem_pool_host()`；
- `HostKVCache` 类型约束；
- `get_size_per_token()`；
- `layout == "page_first"` 限制；
- `_generate_task(host_indices)`；
- 同步 `batch_get_v1/batch_set_v1`；
- 只返回连续长度的 `batch_exists()`。

## 15. SGLang 注册方式

当前 SGLang registry 只识别：

```text
mooncake
mori
```

需要增加：

```python
elif backend == "ucm":
    from ucm.integration.sglang.ucm_direct_linker import (
        UcmSglangDirectLinker,
    )
    linker_cls = UcmSglangDirectLinker
```

然后沿用：

```python
cache.init_cache_linker(
    linker_cls(server_args, params, components=set(cache.components))
)
```

更通用的长期方案是支持配置 dotted class path，避免 SGLang registry 为每个外部 backend 增加硬编码分支。

## 16. Mamba/MiniMax M3 限制

当前 SGLang Direct Linker 白名单只包含 FULL 和 SWA，且 `MambaComponent.build_external_linker_transfer()` 尚未实现。

因此 UCM 第一阶段只能支持：

```text
FULL
FULL + SWA（非 request-ring）
```

要支持 Mamba Direct Linker，还需要在 SGLang 中：

1. 将 `ComponentType.MAMBA` 加入 Linker 白名单；
2. 实现 Mamba LOOKUP/LOAD/OFFLOAD transfer；
3. 定义 temporal/SSM 与 Conv 的物理 key；
4. 定义 Mamba state 的 `TRAILING_PAGES` 边界；
5. 分配并提交 Mamba slot；
6. 保证 Conv 和 SSM 完整恢复；
7. 定义与 FULL/SWA 的共同可恢复边界。

UCM 侧需要为 temporal/SSM 和 Conv 提供独立 Pool layout/store，不能假设所有 component 尺寸一致。

## 17. 分阶段实施建议

### 阶段一：FULL KV

- 新增 `UcmSglangDirectLinker`；
- 复用 `DevicePoolGroup`；
- exact lookup；
- 按 logical layer 使用 `shard_index`；
- 单 load worker；
- 单 offload worker；
- 完成队列和生命周期管理。

### 阶段二：SWA

- 多 Pool key namespace；
- `TRAILING_PAGES`；
- 稀疏 restorable boundary；
- request-ring SWA 明确跳过。

### 阶段三：多 rank 和稳定性

- TP/CP/PP key namespace；
- rank-replicated offload owner；
- prerequisite event；
- reset/close；
- task failure 与 backend GC；
- metrics 和 tracing。

### 阶段四：Mamba/MiniMax M3

- 先补齐 SGLang Mamba Linker hooks；
- temporal/Conv 独立 layout；
- FULL/SWA/Mamba 联合 lookup；
- Conv+SSM 完整性和失败语义。

## 18. 测试清单

1. FULL KV lookup/load/offload round trip；
2. `load()` 只入队，不立即传输；
3. `start_layer_wise_loading()` 每个调度 batch 调用一次；
4. 每层 `wait(task)` 后才 `counter.complete()`；
5. 多请求合并到一个 load batch；
6. offload 入队成功与实际完成成功分离；
7. 任意 Pool/component/layer 失败导致整体 offload 失败；
8. TP/CP/PP rank key 不冲突；
9. `ALL_PAGES` 连续前缀；
10. `TRAILING_PAGES` 稀疏边界；
11. node split 后完成结果发布到正确 node；
12. reset/close 不泄漏 Task、线程和 node lock；
13. L3 GC 后 lookup miss；
14. SWA request-ring 跳过；
15. 后续 Mamba Conv/SSM 缺一时不可恢复。

## 19. 最终结论

推荐新增独立的 `UcmSglangDirectLinker`，复用 UCM Store 已有的异步 Device pointer 和 shard I/O，而不是改造现有 HostPool `UnifiedCacheStore`。

职责边界为：

```text
SGLang DevicePoolGroup
    负责模型物理布局、logical layer 和 Device pointer

UcmSglangDirectLinker
    负责 PoolTransfer 展开、命中策略、异步队列和完成语义

UcmKVStoreBaseV1
    负责 key 查询、Device⇄L3 传输和存储后端
```

该方案可以最大程度复用 UCM 当前能力，同时避免将 HostPool 语义带入 Direct Linker，并为后续 SWA、Mamba 和混合模型扩展保留清晰边界。
