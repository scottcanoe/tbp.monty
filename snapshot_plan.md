# Per-Step Learning Module Snapshot Telemetry — Findings & Plan

## Goal

Give **learning modules (LMs)** a per-step "snapshot" mechanism analogous to what
**sensor modules (SMs)** already have, so that at (and at the end of) every step we
can grab each LM's current state/metrics for live logging/emitting.

Decided requirements:
- **Consumer:** live/streaming per step (grab-latest at each step, like the live plotter does for SMs).
- **Payload:** full detailed state, but with **configurable filters** to opt-in/opt-out of individual items.

---

## Findings

### 1. How SMs do it today (the pattern to mirror)

`SnapshotTelemetry` — `src/tbp/monty/frameworks/models/sensor_modules.py:60-107`

- Plain class (not a dataclass) holding **parallel lists** (array-of-structs):
  `raw_observations`, `poses` (`sm_rotation` + `sm_location`), `info`.
- Each SM owns `self._snapshot_telemetry = SnapshotTelemetry()`.
- Appended once per step inside `SM.step()`, gated by `self.save_raw_obs and not self.is_exploring`:
  - `CameraSM.step` → `sensor_modules.py:656-659`
  - `Probe` → `sensor_modules.py:416, 436-438`
  - `TwoDSensorModule` → `two_d_sensor_module.py:127, 187-190`
  - `SalienceSM` → `salience/sensor_module.py:56-58, 155-160` (records a rich `info` dict:
    salience_map / segmentation_mask / goals / voxel_grid; telemetry is constructor-injectable)
- `update_state(agent)` refreshes `self.state` (pose) immediately before `step()` records.
- Exposed via `state_dict() -> Memento` (keys `raw_observations`, `sm_properties`, `info`,
  plus merged `processed_observations`).
- **Reset** per episode via `SnapshotTelemetry.reset()`, called from each `SM.reset()`,
  driven by `Monty.reset()` → `monty_base.py:393-394`.

**Read-out seams:**
- Model-level: `Monty.state_dict()` builds `sm_dict = {i: module.state_dict() ...}`
  — `monty_base.py:425-434`.
- Episode-end loggers iterate `model.sensor_modules` and store `sm.state_dict()`:
  `graph_matching_loggers.py` `DetailedGraphMatchingLogger.update_episode_data` (~574-576),
  `SelectiveEvidenceLogger` (~656-659).
- **Live per-step** (the model for our use case): `utils/live_plotter.py:64-67` reaches
  straight into `first_sensor_module._snapshot_telemetry.raw_observations` and takes `[-1]`.

`Memento = Mapping[str, Any]` and the `Snapshotable` Protocol (`state_dict()`/`load_state_dict()`)
live in `src/tbp/monty/memento.py:12-37`.

### 2. What LMs have today (ingredients, but no live-grab interface)

**Class hierarchy**
- Abstract: `LearningModule` — `abstract_monty_classes.py:356` (implements `Snapshotable`).
- Common base with the buffer/stats machinery: `GraphLM` — `graph_matching.py:528`.
- Production LM: `EvidenceGraphLM` — `evidence_matching/learning_module.py:125`.
  (Others: `FeatureGraphLM`, `DisplacementGraphLM`, `NoResetEvidenceGraphLM`.)

**The dict-of-growing-lists** ("keys = metric, values = growing lists")
- `FeatureAtLocationBuffer` — `src/tbp/monty/frameworks/models/buffer.py:36`.
  Each `GraphLM` owns one: created at `graph_matching.py:544`, reset per episode in
  `reset_stm()` at `graph_matching.py:587`.
- `self.buffer.stats` — `buffer.py:55-81`. Grown by `update_stats()` — `buffer.py:144-154`
  (appends `copy.deepcopy(value)` per step when `append=True`; always appends `"time"`).
  Sibling per-step data (also grows, keyed by input channel): `buffer.locations`,
  `buffer.features` (NaN-padded to stay index-aligned by step), `buffer.displacements`,
  `buffer.on_object`, `buffer.input_percepts`.

**The per-step snapshot producer already exists**
- `collect_stats_to_save()`:
  - Base `GraphLM` — `graph_matching.py:918` → `{"possible_matches": ...}` (+ detailed if enabled).
  - `EvidenceGraphLM` override — `evidence_matching/learning_module.py:788` → adds
    `"current_mlh"` and appends `mlh_prediction_error`.
  - `EvidenceGraphLM._add_detailed_stats()` — `learning_module.py:1268` → per-object
    `evidences` / `possible_locations` / `possible_rotations` (keyed by `graph_id`) + `symmetry_evidence`.
- Called each matching step in `matching_step()` and fed to the buffer:
  `graph_matching.py:643` → `self.buffer.update_stats(stats, append=self.has_detailed_logger)`.
- Live MLH is on the LM as `self.current_mlh` (`learning_module.py:301`, updated ~836),
  read via `get_current_mlh()` (`learning_module.py:650`).

**Two gaps vs. the SM pattern**
1. **Transposed layout.** LM state is struct-of-arrays (`stats` = `{metric: [t0,t1,...]}`) with
   sibling data (`locations`/`features`/`displacements`) stored separately — there is no single
   "the LM at step *t*" record to grab.
2. **Gated.** Growing per-step lists only happen when `has_detailed_logger=True`
   (default `False`, `graph_matching.py:567`; set per-LM in `monty_experiment.py:383/398/416`).
   With `append=False`, only the latest value is kept. So live consumers can't rely on it.

Note: `LearningModule.state_dict()` (`Snapshotable`) is a **full save/load memento**
(`graph_memory`, `target_to_graph_id`, ... — `graph_matching.py:945`), **not** per-step metrics.
Different concept from the snapshot telemetry we want.

### 3. The emit seam already exists

`StepHook` — `src/tbp/monty/frameworks/experiments/hooks.py:26-62`.
- Protocol firing **once per step**, receives `ctx, monty, supervised_lm_ids, step, observations,
  actions, experiment`, returns actions. Documented for "visualization of ... any internal state
  of the Monty model."
- `NoOpStepHook` (`hooks.py:69`) is the default no-op.
- This is the natural place for a live emitter to grab each LM's latest snapshot — no core
  step-loop changes needed.

---

## Plan

### Component A — `LMSnapshotTelemetry` (mirror `SnapshotTelemetry`)

New lightweight accumulator, placed on `GraphLM` so all graph LMs inherit it.

```python
class LMSnapshotTelemetry:
    def __init__(self, snapshot_filter: SnapshotFilter | None = None,
                 retain_all: bool = False):
        self._filter = snapshot_filter or SnapshotFilter.default()
        self._retain_all = retain_all
        self.snapshots: list[dict] = []          # array-of-structs, one per step

    def record(self, snapshot: dict) -> None:
        filtered = self._filter.apply(snapshot)  # filter BEFORE deepcopy
        if self._retain_all:
            self.snapshots.append(filtered)
        else:
            self.snapshots = [filtered]          # latest-only (default)

    def get_latest(self) -> dict | None:
        return self.snapshots[-1] if self.snapshots else None

    def reset(self) -> None:
        self.snapshots.clear()

    def state_dict(self) -> Memento:
        return {"snapshots": self.snapshots}
```

Design notes:
- **Array-of-structs**, matching `SnapshotTelemetry`'s shape and the "one record per step" model
  (not the transposed `buffer.stats` layout).
- **Always-on** (not gated on `has_detailed_logger`) — live streaming can't depend on that flag.
  The *filter* is the throttle instead.
- **Retention default = latest-only.** Full-detailed LM state is much heavier than raw obs; an
  episode-length list of per-graph evidence arrays is expensive. Live consumers only need latest.
  `retain_all=True` stays available for anyone who also wants an episode dump.

### Component B — `SnapshotFilter` (the opt-in/opt-out control)

- Configurable include/exclude over snapshot keys; supports the heavy per-graph keys
  (`evidences`, `possible_locations`, `possible_rotations`, `symmetry_evidence`) as opt-in.
- Applied at `record()` time **before** the `deepcopy`, so opted-out heavy items are never copied
  — this is what makes always-on full-detail affordable for a live stream.
- **Config shape (to confirm):** simple include/exclude key-sets vs. glob patterns vs. predicate
  callables. Recommend starting with include/exclude key-sets (+ optional glob).
- **Default filter (to confirm):** include `current_mlh` (graph_id/location/rotation/scale/evidence),
  `on_object`, processed-this-step flag, latest `location`/`feature`; **exclude** the per-graph
  arrays. Cheap always-on stream; opt the heavy keys back in per run.

### Component C — Capture at the existing seam

In `matching_step()` (near `graph_matching.py:643`, where `collect_stats_to_save()` already runs):
- Reuse `collect_stats_to_save()` output (call the detailed path for the snapshot regardless of
  `has_detailed_logger`) — **do not** build a second collector.
- Merge in sibling data that lives outside `stats` so each snapshot is self-contained for step *t*:
  latest `buffer.locations[ch][-1]`, relevant `buffer.features`, `current_mlh`.
- Hand the assembled dict to `self._snapshot_telemetry.record(...)`.

### Component D — Lifecycle wiring

- Construct `self._snapshot_telemetry = LMSnapshotTelemetry(...)` in `GraphLM.__init__` /
  `reset_stm`, taking filter + retention from LM config.
- Call `self._snapshot_telemetry.reset()` from the LM reset path alongside `buffer.reset()`
  (`reset_stm`, `graph_matching.py:587`) so episode-boundary semantics match SMs.
- Optionally surface it in a model-level read-out parallel to `Monty.state_dict()`'s `sm_dict`
  (iterate `self.learning_modules`) if an episode-end dump is also wanted.

### Component E — Live emitter (`StepHook`)

- New `LMSnapshotEmitHook(StepHook)`:
  - `__call__` iterates `monty.learning_modules`, calls `lm._snapshot_telemetry.get_latest()`,
    streams/emits it, returns `actions` unchanged.
  - `close()` flushes/closes the sink.
- Registered like other step hooks (see how `NoOpStepHook` is wired into the experiment).

### Open decisions to confirm before implementation

1. **Filter config shape:** include/exclude key-sets (recommended) vs. glob vs. predicate callables.
2. **Default filter contents:** confirm the cheap MLH-level default above.
3. **Retention default:** latest-only (recommended) vs. bounded ring buffer vs. retain-all.
4. **Emit sink:** where the `StepHook` streams to (callback, queue, file, socket, viewer).
5. **Naming/placement:** `_snapshot_telemetry` on the LM to match SM naming; module location for
   `LMSnapshotTelemetry` / `SnapshotFilter` (likely `buffer.py` or a new `lm_snapshot.py`).

### Key file references

| Concern | Location |
| --- | --- |
| SM `SnapshotTelemetry` (pattern) | `models/sensor_modules.py:60-107` |
| SM live grab-latest | `utils/live_plotter.py:64-67` |
| `Memento` / `Snapshotable` | `memento.py:12-37` |
| `GraphLM` base | `models/graph_matching.py:528` |
| `EvidenceGraphLM` | `models/evidence_matching/learning_module.py:125` |
| Buffer + `stats` | `models/buffer.py:36, 55-81, 144-154` |
| `collect_stats_to_save` | `graph_matching.py:918`; `learning_module.py:788`; `_add_detailed_stats` `learning_module.py:1268` |
| Capture seam (`matching_step`) | `graph_matching.py:643` |
| Reset seam (`reset_stm`) | `graph_matching.py:587` |
| `has_detailed_logger` default/set | `graph_matching.py:567`; `monty_experiment.py:383/398/416` |
| `StepHook` (emit seam) | `frameworks/experiments/hooks.py:26-62` |
| Model-level read-out (parallel) | `monty_base.py:425-434` |
