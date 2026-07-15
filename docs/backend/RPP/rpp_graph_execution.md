# RPP graph execution

This document explains how the RPP backend executes a GGML graph after operator mapping has selected the current
`ggml_rpp_node` instances. For node selection and reuse rules, see [RPP operator mapping](rpp_operator_mapping.md).

- [Scope](#scope)
- [Execution entry point](#execution-entry-point)
- [Graph wrapper cache](#graph-wrapper-cache)
- [Graph update decision](#graph-update-decision)
- [Rebuild and capture path](#rebuild-and-capture-path)
- [Replay path](#replay-path)
- [Shared and exclusive kernel graphs](#shared-and-exclusive-kernel-graphs)
- [Child graph update](#child-graph-update)
- [Direct dispatch mode](#direct-dispatch-mode)
- [Synchronization and diagnostics](#synchronization-and-diagnostics)

## Scope

`rpp_operator_mapping.md` answers: which RPP node implements a GGML node?

This document answers: how are the selected RPP nodes captured, updated, launched, and synchronized?

```mermaid
flowchart LR
    mapping["operator mapping"]
    selectedNodes["rpp_in_use_nodes"]
    graphExecution["graph execution"]
    capturedGraphs["rpp_kernel_cgraph"]
    launch["RPP graph launch"]

    mapping --> selectedNodes
    selectedNodes --> graphExecution
    graphExecution --> capturedGraphs
    capturedGraphs --> launch
```

## Execution entry point

Graph execution starts from `ggml_backend_rpp_graph_compute()`.

```mermaid
flowchart TB
    startNode["ggml_backend_rpp_graph_compute"]
    setDevice["ggml_rpp_set_device"]
    getWrapper["get or create ggml_rpp_cgraph"]
    updateState["update_ggml_rpp_infer_states"]
    updateCheck["is_rpp_graph_update_required"]
    execute["evaluate_and_capture_rpp_graph"]
    success["GGML_STATUS_SUCCESS"]

    startNode --> setDevice
    setDevice --> getWrapper
    getWrapper --> updateState
    updateState --> updateCheck
    updateCheck --> execute
    execute --> success
```

The entry point does not directly launch kernels. It prepares the backend state, decides whether the RPP-side graph
must be rebuilt, then delegates to `evaluate_and_capture_rpp_graph()`.

## Graph wrapper cache

The backend context caches one `ggml_rpp_cgraph` wrapper per `ggml_cgraph *`.

```mermaid
flowchart TB
    backendCtx["ggml_backend_rpp_context"]
    incomingGraph["incoming ggml_cgraph"]
    graphMap["rpp_graphs"]
    hit["reuse existing wrapper"]
    miss["create ggml_rpp_cgraph"]
    current["cur_rpp_graph"]
    graphList["gglm_graphs"]

    backendCtx --> graphMap
    incomingGraph --> graphMap
    graphMap -->|"found"| hit
    graphMap -->|"missing"| miss
    hit --> current
    miss --> current
    miss --> graphList
```

The wrapper persists across launches. It owns:

- historical RPP nodes in `rpp_nodes`
- current launch mapping in `cur_rpp_nodes`
- captured RPP kernel graphs in `rpp_kernel_graphs`
- current graph properties in `ggml_graph_properties`

## Graph update decision

`is_rpp_graph_update_required()` compares the current GGML graph with the previous property snapshot.

```mermaid
flowchart TB
    propsSize["property vector size"]
    nodeLoop["for each cgraph node"]
    compare["compare node address op shape stride sources params"]
    changed["mark update required"]
    snapshot["store latest properties"]
    result["return update flag"]

    propsSize --> nodeLoop
    nodeLoop --> compare
    compare -->|"changed"| changed
    compare -->|"same"| snapshot
    changed --> snapshot
    snapshot --> result
```

The graph is rebuilt when:

- the number of GGML nodes changes
- a node data pointer changes
- op type changes
- shape or stride changes
- source tensor data pointers change
- selected op params change, for example `GGML_OP_SCALE`

`update_ggml_rpp_infer_states()` also clears `cur_rpp_nodes` across cached graphs when the first node changes. This is
how prefill/decode-like transitions force node re-selection.

## Rebuild and capture path

When the graph must be rebuilt, `evaluate_and_capture_rpp_graph()` clears current graph state, classifies tensors, maps
or fuses nodes, then captures or directly launches the selected RPP nodes.

```mermaid
flowchart TB
    rebuild["rpp_graph_update_required"]
    reset["ggml_rpp_reset_graph"]
    classify["classify inputs outputs weights KV"]
    walk["walk cgraph nodes"]
    skip["skip view reshape transpose permute empty none"]
    fuse["try fusion"]
    dispatch["ggml_rpp_compute_forward"]
    selected["rpp_in_use_nodes ready"]
    captureCheck["graph capture disabled"]
    direct["direct dispatch"]
    capture["capture or update kernel graphs"]
    launch["launch selected kernel graphs"]
    sync["stream synchronize"]

    rebuild --> reset
    reset --> classify
    classify --> walk
    walk --> skip
    skip --> fuse
    fuse -->|"matched"| selected
    fuse -->|"not matched"| dispatch
    dispatch --> selected
    selected --> captureCheck
    captureCheck -->|"yes"| direct
    captureCheck -->|"no"| capture
    capture --> launch
    direct --> sync
    launch --> sync
```

Tensor classification fills:

- `nodes_i`: inputs whose sources are outside the cgraph
- `nodes_o`: graph outputs not consumed later
- `nodes_matmul_weight`: external matmul weights
- `nodes_mul_weight`: external multiply weights
- `nodes_cache_kv`: KV-cache tensors

Operator mapping fills `cur_rpp_nodes` and appends selected nodes to `rpp_in_use_nodes`. The graph execution layer then
turns `rpp_in_use_nodes` into launchable `rpp_kernel_cgraph` instances.

## Replay path

When no graph update is required, `evaluate_and_capture_rpp_graph()` skips node rebuilding. It uses the previously
selected `rpp_in_use_kernel_graphs` and only reruns deferred launch functions before graph launch.

```mermaid
flowchart TB
    noUpdate["no graph update"]
    captureEnabled["graph capture enabled"]
    launchFuncs["run launch_funcs"]
    launchCached["launch cached rpp_kernel_cgraph"]
    directNodes["direct dispatch current nodes"]
    sync["stream synchronize"]

    noUpdate --> captureEnabled
    captureEnabled -->|"yes"| launchFuncs
    launchFuncs --> launchCached
    launchCached --> sync
    captureEnabled -->|"no"| directNodes
    directNodes --> sync
```

This path is the reason captured graph state must be updated carefully when the selected RPP node changes.

## Shared and exclusive kernel graphs

RPP graph capture groups per-op kernel graphs into `rpp_kernel_cgraph`.

```mermaid
flowchart TB
    inUse["rpp_in_use_nodes"]
    graphKey["graph_index equals node count plus n_ubatch"]
    graphCache["rpp_kernel_graphs"]
    cacheMiss["cache miss"]
    exclusiveCheck["ggml_rpp_is_exclusive_graph_op"]
    shared["shared parent graph"]
    exclusive["exclusive wrapper graph"]
    instantiate["graph_instantiate"]
    active["rpp_in_use_kernel_graphs"]

    inUse --> graphKey
    graphKey --> graphCache
    graphCache --> cacheMiss
    cacheMiss --> exclusiveCheck
    exclusiveCheck -->|"MUL_MAT_ID"| exclusive
    exclusiveCheck -->|"normal op"| shared
    shared --> instantiate
    exclusive --> instantiate
    instantiate --> active
```

Shared graph:

- owns a parent `RPPgraph`
- calls `rpp_node_kernel::add_to_parent_node()`
- launches with `RPP_LAUNCH_KERNEL(graphexec, stream)`

Exclusive graph:

- wraps one already-instantiated per-op graph
- currently used for `GGML_OP_MUL_MAT_ID`
- launches by calling the node's `rpp_dispatch_func()` from `graph_launch()`

## Child graph update

On a graph-cache hit, the backend does not rebuild the parent graph. It calls
`rpp_kernel_cgraph::update_child_graph(cur_rpp_nodes)`.

```mermaid
flowchart TB
    cached["cached rpp_kernel_cgraph"]
    oldChild["old child rpp_node"]
    tensor["child cur_ggml_tensor"]
    currentMap["cur_rpp_nodes"]
    selected["current selected rpp_node"]
    same["same node no-op"]
    replace["rppGraphExecUpdateChildGraphExec"]
    remap["update rpp_to_graph_nodes"]

    cached --> oldChild
    oldChild --> tensor
    tensor --> currentMap
    currentMap --> selected
    selected -->|"same pointer"| same
    selected -->|"different pointer"| replace
    replace --> remap
```

For shared parent graphs, the update path replaces child graph exec bindings through
`rppGraphExecUpdateChildGraphExec()`. The parent graph exec remains alive, while the child binding points at the newly
selected kernel node.

For exclusive graphs, the wrapper updates its `graph`, `graphexec`, node pointer, and node-to-graph map to match the
new selected node.

## Direct dispatch mode

Set `GGML_RPP_DISABLE_GRAPH_CAPTURE=1` to bypass captured graph execution.

```mermaid
flowchart TB
    env["GGML_RPP_DISABLE_GRAPH_CAPTURE"]
    nodeList["rpp_in_use_nodes"]
    dispatch["rpp_dispatch_func"]
    instantial["is_instantial equals 1"]
    launch["is_launch equals 1"]

    env --> nodeList
    nodeList --> dispatch
    dispatch --> instantial
    dispatch --> launch
```

In direct mode, every selected node is called as:

```text
rpp_dispatch_func(ctx, cur_ggml_tensor, 1, 1)
```

This mode is useful when debugging graph capture or child graph replacement, because it keeps operator mapping intact
but removes parent graph replay from the execution path.

## Synchronization and diagnostics

After rebuild, replay, or direct dispatch, the backend synchronizes all streams created for the active device.

```mermaid
flowchart TB
    launchDone["launch path complete"]
    streamArray["streams current device"]
    syncEach["rtStreamSynchronize"]
    doneNode["graph_evaluated_or_captured true"]

    launchDone --> streamArray
    streamArray --> syncEach
    syncEach --> doneNode
```

Useful switches:

| Switch | Effect |
| --- | --- |
| `GGML_RPP_DISABLE_GRAPH_CAPTURE=1` | Disable graph capture/replay and use direct dispatch. |
| `GGML_RPP_DISABLE_FUSION=1` | Disable fusion before operator mapping. |
| `GGML_RPP_DUMP_OPS=ON` | Dump the GGML graph op list and dot graph, then exit graph compute. |
| `GGML_RPP_PERF_TRACE=ON` | Enable Perfetto trace windows around selected graph execution phases. |

When debugging graph execution:

1. Confirm whether `is_rpp_graph_update_required()` rebuilds or replays.
2. Check whether fusion changed the node sequence before `ggml_rpp_compute_forward()`.
3. Inspect `rpp_in_use_nodes` before graph capture.
4. Check the `graph_index` derived from node count and `n_ubatch`.
5. For stale outputs, inspect `update_child_graph()` and `rpp_to_graph_nodes`.
6. Compare captured mode against `GGML_RPP_DISABLE_GRAPH_CAPTURE=1`.
