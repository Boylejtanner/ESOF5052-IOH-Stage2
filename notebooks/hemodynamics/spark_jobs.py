"""Spark workloads for the project, written for the desktop + laptop cluster (see ``cluster.py``).

  smoke_aggregate  B0  counts per (split, dataset, event) and the sum of ``sampling_weight`` over the dev subset, computed by Spark tasks and compared with pandas
  run_features     B1  distributed window-feature extraction (``mapInPandas`` over a DataFrame of file names): decode compact windows, 34 features per window
  stream_replay    B2  Structured Streaming replay: windows are replayed per patient in chronological order at a fixed rate, features and a pluggable score are computed in the
                       stream, end-to-end latency p50/p95/p99 and throughput are recorded

Data are read by the TASKS from the worker's own project root (``BSHI_ROOT``) with pyarrow, so a worker on another machine needs its own copy of the files (the laptop shard), not the
desktop's paths. Nothing here writes through Spark (no winutils): results come back to the driver and are written from Python.
"""
from __future__ import annotations
# Compare partitioned processing and streaming replay with local reference results.

import os
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from . import bshi as S

DEV_REL = Path("data") / "derived" / "bshi" / "dev_subset"
PLANES = ("ecg", "art", "ppg")
FEATURE_NAMES = [f"ctx_{n}" for n in S.CONTEXT_NAMES] + [f"{p}_{s}" for p in PLANES for s in ("mean", "std")] + [f"n_{p}" for p in PLANES]
META_COLUMNS = ["window_id", "split", "dataset", "y_event", "sampling_weight"]
RAW_COLUMNS = ["window_id", "dataset_code", "waveforms_f16", "scalars_f32", "beat_mask_u8", "context_f32", "variant_mask_f32"] + S.LABEL_OUT


def dev_files(root):
    base = Path(root) / DEV_REL
    return sorted(str(p.relative_to(base)).replace("\\", "/") for p in base.glob("split=*/part-*.parquet"))


def _base():
    return Path(os.environ.get("BSHI_ROOT", ".")) / DEV_REL


# B0 smoke aggregation
def _agg_partition(index, rels):
    import socket
    import pyarrow.parquet as pq
    t0 = time.time()
    rows, n_rows = [], 0
    for rel in rels:
        t = pq.read_table(_base() / rel, columns=["split", "dataset", "y_event", "sampling_weight"]).to_pandas()
        n_rows += len(t)
        for (s, d, y), g in t.groupby(["split", "dataset", "y_event"]):
            rows.append(((s, d, bool(y)), (int(len(g)), float(g.sampling_weight.astype("float64").sum()))))
    yield ("task", index, os.environ.get("BSHI_WORKER_NAME", "?"), socket.gethostname(), n_rows, round(time.time() - t0, 3))
    for kv in rows:
        yield ("agg",) + kv


def executor_task_counts(spark):
    """Per executor: host:port and finished tasks, from the driver's REST API (the Spark UI's data)."""
    import json
    ui = spark._bshi["driver_ui"] if hasattr(spark, "_bshi") else "http://localhost:4040"
    app = spark.sparkContext.applicationId
    try:
        with urlopen(f"{ui}/api/v1/applications/{app}/executors", timeout=10) as r:
            ex = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        return [{"error": repr(exc)[:120]}]
    return [{"id": e["id"], "hostPort": e["hostPort"], "cores": e["totalCores"], "completedTasks": e["completedTasks"], "failedTasks": e["failedTasks"]} for e in ex if e["id"] != "driver"]


def smoke_aggregate(spark, root, native=False):
    """Counts per (split, dataset, y_event) and weight sums over the dev subset from Spark tasks, with the pandas reference and the exact comparison. ``native=True`` also tries Spark's own
    parquet reader; on Windows that needs hadoop.dll / winutils for the directory listing (NativeIO), so it is off by default and the tasks read with pyarrow."""
    files = dev_files(root)
    t0 = time.perf_counter()
    out = spark.sparkContext.parallelize(files, len(files)).mapPartitionsWithIndex(lambda i, it: _agg_partition(i, list(it))).collect()
    seconds = time.perf_counter() - t0
    agg = {}
    for rec in out:
        if rec[0] == "agg":
            n, w = agg.get(rec[1], (0, 0.0))
            agg[rec[1]] = (n + rec[2][0], w + rec[2][1])
    # Summarize the records separately within each requested group.
    tasks = pd.DataFrame([r[1:] for r in out if r[0] == "task"], columns=["partition", "worker", "host", "rows", "seconds"])
    ref = pd.concat([pd.read_parquet(Path(root) / DEV_REL / rel, columns=["split", "dataset", "y_event", "sampling_weight"]) for rel in files])
    ref["y_event"] = ref.y_event.astype(bool)
    ref = ref.groupby(["split", "dataset", "y_event"]).agg(n=("sampling_weight", "size"), w=("sampling_weight", lambda s: float(s.astype("float64").sum())))
    # Summarize the records separately within each requested group.
    table = pd.DataFrame([{"split": k[0], "dataset": k[1], "y_event": k[2], "windows_spark": v[0], "windows_pandas": int(ref.loc[k, "n"]), "weight_spark": v[1], "weight_pandas": float(ref.loc[k, "w"])}
                          for k, v in sorted(agg.items())])
    res = {"seconds": round(seconds, 2), "windows": int(table.windows_spark.sum()), "counts_equal": bool((table.windows_spark == table.windows_pandas).all()),
           "weights_equal": bool(np.allclose(table.weight_spark, table.weight_pandas, rtol=1e-9)), "max_rel_weight_diff": float(((table.weight_spark - table.weight_pandas).abs() / table.weight_pandas).max()),
           "tasks_by_worker": tasks.groupby("worker").agg(tasks=("partition", "size"), rows=("rows", "sum"), host=("host", "first")).reset_index().to_dict("records"), "table": table,
           "executors": executor_task_counts(spark)}
    if native:
        try:                                                              # Spark's own parquet reader (needs the driver-side file system patch on Windows, see cluster.ensure_fs_patch)
            p = (Path(root) / DEV_REL).resolve().as_posix()          # a plain path: Hadoop's Path handles the spaces in this project's folder, a file: URI would be %20-encoded
            df = spark.read.parquet(p).groupBy("split", "dataset", "y_event").agg({"sampling_weight": "sum", "window_id": "count"}).toPandas()
            res["native_read"] = {"ok": True, "windows": int(df["count(window_id)"].sum()), "equal_to_pandas": int(df["count(window_id)"].sum()) == res["windows"]}
        except Exception as exc:
            res["native_read"] = {"ok": False, "error": str(exc)[:200]}
    return res


# B1 features
def features_from_batch(b):
    """34 per-window features of a decoded batch: the 25 context values, mean/std of the ECG, ART and PPG planes over the valid beats, and the valid-beat counts."""
    x, bm, ctx = b["x"], b["beat_mask"], b["context"]
    cols = {f"ctx_{n}": ctx[:, i].astype(np.float64) for i, n in enumerate(S.CONTEXT_NAMES)}
    for p, name in enumerate(PLANES):
        # Calculate valid-sample flags and the mean level for the following step.
        valid = bm[:, p, :].astype(np.float64)
        plane = x[:, p].astype(np.float64)
        cnt = valid.sum(axis=1) * x.shape[3]
        s1 = (plane * valid[:, :, None]).sum(axis=(1, 2))
        s2 = ((plane ** 2) * valid[:, :, None]).sum(axis=(1, 2))
        mean = np.divide(s1, cnt, out=np.zeros_like(s1), where=cnt > 0)
        var = np.divide(s2, cnt, out=np.zeros_like(s2), where=cnt > 0) - mean ** 2
        cols[f"{name}_mean"], cols[f"{name}_std"], cols[f"n_{name}"] = mean, np.sqrt(np.maximum(var, 0.0)), valid.sum(axis=1)
    return pd.DataFrame(cols)[FEATURE_NAMES]


def features_for_file(base, rel):
    """Read one dev-subset part, decode it and return meta columns + features (pandas)."""
    # Read the stored table required for this calculation.
    import pyarrow.parquet as pq
    t = pq.read_table(Path(base) / rel, columns=RAW_COLUMNS + ["split", "dataset", "sampling_weight"]).to_pandas()
    feats = features_from_batch(S.decode_compact(t[RAW_COLUMNS].to_dict("list")))
    meta = t[["window_id", "split", "dataset"]].assign(y_event=t.y_event.astype(bool), sampling_weight=t.sampling_weight.astype("float64"))
    return pd.concat([meta.reset_index(drop=True), feats], axis=1)


FEATURE_SCHEMA = "window_id long, split string, dataset string, y_event boolean, sampling_weight double, worker string, " + ", ".join(f"{n} double" for n in FEATURE_NAMES)


def _features_map(batches):
    worker = os.environ.get("BSHI_WORKER_NAME", "?")
    base = _base()
    for pdf in batches:
        for rel in pdf["rel"]:
            out = features_for_file(base, rel)
            out.insert(5, "worker", worker)
            yield out


def run_features(spark, root, files=None):
    """B1: the features of every dev-subset window, one Spark partition per file. Returns (pandas result, timing dict)."""
    # Summarize the records separately within each requested group.
    files = files or dev_files(root)
    df = spark.createDataFrame(pd.DataFrame({"rel": files})).repartition(len(files))
    t0 = time.perf_counter()
    pdf = df.mapInPandas(_features_map, schema=FEATURE_SCHEMA).toPandas()
    seconds = time.perf_counter() - t0
    by_worker = pdf.groupby("worker").size().to_dict()
    return pdf, {"windows": int(len(pdf)), "seconds": round(seconds, 2), "windows_per_s": round(len(pdf) / seconds, 1), "windows_by_worker": by_worker, "partitions": len(files),
                 "executors": len(spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()) - 1}


def verify_features(root, pdf, n=100, seed=5052):
    """The Spark features of ``n`` random windows against features computed from a FRESH ``bshi.assemble`` of the same windows (context exact; planes differ by float16 storage only)."""
    # Build the requested beat images, numerical context, and beat masks.
    ids = pdf.sample(n, random_state=seed).window_id.to_numpy()
    idx = S.load_window_index(root)
    rows = idx[idx.window_id.isin(ids)].sort_values(["dataset", "case_id", "beat_lo"]).reset_index(drop=True)
    # Build the requested beat images, numerical context, and beat masks.
    fresh = features_from_batch(S.assemble(root, rows, "eap", cache=S.CaseCache())).assign(window_id=rows.window_id.to_numpy()).set_index("window_id")
    got = pdf.set_index("window_id").loc[rows.window_id.to_numpy(), FEATURE_NAMES]
    ctx = [c for c in FEATURE_NAMES if c.startswith("ctx_") or c.startswith("n_")]
    wave = [c for c in FEATURE_NAMES if c not in ctx]
    d_ctx = float(np.abs(got[ctx].to_numpy() - fresh.loc[rows.window_id.to_numpy(), ctx].to_numpy()).max())
    d_wave = float(np.abs(got[wave].to_numpy() - fresh.loc[rows.window_id.to_numpy(), wave].to_numpy()).max())
    return {"windows": int(len(rows)), "max_abs_diff_context_and_counts": d_ctx, "max_abs_diff_plane_stats": d_wave, "context_exact": d_ctx < 1e-6, "plane_stats_within_float16": d_wave < 5e-3}


# B2 streaming replay
def default_score(feats):
    """Placeholder scorer (the trained model replaces it next queue): risk rises as the current MAP falls below 75 mmHg."""
    return 1.0 / (1.0 + np.exp((feats["ctx_map_current"].to_numpy() - 70.0) / 6.0))


def replay_plan(root, n_windows=2000, concurrent_patients=4, split="test"):
    """Windows in replay order: ``concurrent_patients`` patients are monitored at the same time, each in chronological order (round robin), patients one group after the other.
    Rows carry the file and the row inside it, so the stream needs only a sequence number and the tasks read the window from the worker's own data."""
    import pyarrow.parquet as pq
    frames = []
    for rel in dev_files(root):
        if f"split={split}" not in rel:
            continue
        t = pq.read_table(Path(root) / DEV_REL / rel, columns=["window_id", "patient_key", "t_end_sec"]).to_pandas()
        frames.append(t.assign(rel=rel, row=np.arange(len(t))))
    p = pd.concat(frames, ignore_index=True).sort_values(["patient_key", "t_end_sec"])
    patients = list(dict.fromkeys(p.patient_key))
    order = []
    for g in range(0, len(patients), concurrent_patients):
        grp = [p[p.patient_key == k].reset_index(drop=True) for k in patients[g:g + concurrent_patients]]
        for i in range(max(len(x) for x in grp)):
            order += [x.iloc[i] for x in grp if i < len(x)]
        if len(order) >= n_windows:
            break
    plan = pd.DataFrame(order).head(n_windows).reset_index(drop=True)
    return plan[["rel", "row", "window_id", "patient_key", "t_end_sec"]]


STREAM_SCHEMA = "seq long, window_id long, patient_key string, t_end_sec double, emit_ms long, done_ms long, score double, worker string, " + ", ".join(f"{n} double" for n in FEATURE_NAMES)


def _stream_map_factory(plan_records, score_path):
    plan = pd.DataFrame(plan_records, columns=["rel", "row", "window_id", "patient_key", "t_end_sec"])

    def fn(batches):
        import importlib
        import pyarrow.parquet as pq
        mod, name = score_path.split(":")
        score = getattr(importlib.import_module(mod), name)
        base, worker = _base(), os.environ.get("BSHI_WORKER_NAME", "?")
        # Process the selected records in order and retain their individual results.
        for pdf in batches:
            pdf = pdf[pdf["value"] < len(plan)].sort_values("value")
            parts = []
            for rel, g in plan.iloc[pdf["value"].to_numpy()].groupby("rel", sort=False):
                # Read the stored table required for this calculation.
                t = pq.read_table(base / rel, columns=RAW_COLUMNS).to_pandas().iloc[g["row"].to_numpy()]
                feats = features_from_batch(S.decode_compact(t.to_dict("list")))
                parts.append(pd.concat([g.reset_index(drop=True)[["window_id", "patient_key", "t_end_sec"]], feats], axis=1))
            if not parts:
                continue
            # Collect the per-record values into a table for comparison.
            out = pd.concat(parts, ignore_index=True)
            seq_of = dict(zip(plan.window_id, range(len(plan))))
            out.insert(0, "seq", out.window_id.map(seq_of).astype("int64"))
            emit = pdf.set_index(pdf["value"].map(lambda v: int(plan.window_id.iloc[v])))["timestamp"]
            emit_ms = (pd.to_datetime(emit.loc[out.window_id], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
            out.insert(4, "emit_ms", emit_ms.to_numpy().astype("int64"))
            out.insert(5, "done_ms", int(time.time() * 1000))
            out.insert(6, "score", score(out))
            out.insert(7, "worker", worker)
            yield out
    return fn


def stream_replay(spark, root, n_windows=1500, rate=100, concurrent_patients=4, score="hemodynamics.spark_jobs:default_score", timeout=180):
    """B2: replay ``n_windows`` dev-subset windows at ``rate`` windows/s through Structured Streaming and record the end-to-end latency (scheduled emission time of a window until its
    result is back on the driver), p50/p95/p99, throughput and the micro-batch sizes. The session must be created with ``streaming=True`` (Windows checkpoint support)."""
    # Calculate the observation count and the saved run state for the following step.
    plan = replay_plan(root, n_windows, concurrent_patients)
    n = len(plan)
    fn = _stream_map_factory(list(plan.itertuples(index=False, name=None)), score)
    src = spark.readStream.format("rate").option("rowsPerSecond", int(rate)).option("numPartitions", 2).load().filter(f"value < {n}")
    stream = src.mapInPandas(fn, schema=STREAM_SCHEMA)
    got, batches, state = [], [], {"stopping": False}

    def on_batch(df, batch_id):
        recv = int(time.time() * 1000)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pdf = df.toPandas()
        except Exception:
            if state["stopping"]:                                      # the last (empty) micro-batch is cancelled by q.stop()
                return
            raise
        if len(pdf):
            pdf["recv_ms"] = recv
            got.append(pdf)
            batches.append({"batch": int(batch_id), "windows": int(len(pdf))})

    q = stream.writeStream.foreachBatch(on_batch).trigger(processingTime="1 second").start()
    t0 = time.time()
    while time.time() - t0 < timeout and sum(len(g) for g in got) < n:
        time.sleep(0.5)
    state["stopping"] = True
    q.stop()
    res = pd.concat(got, ignore_index=True) if got else pd.DataFrame()
    if res.empty:
        return {"ok": False, "windows": 0}, res
    # Summarize the records separately within each requested group.
    lat = (res.recv_ms - res.emit_ms).to_numpy(dtype=float)
    span = (res.recv_ms.max() - res.emit_ms.min()) / 1000
    stats = {"ok": True, "windows": int(len(res)), "requested": n, "rate_windows_per_s": rate, "latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p95": float(np.percentile(lat, 95)),
             "latency_ms_p99": float(np.percentile(lat, 99)), "latency_ms_max": float(lat.max()), "throughput_windows_per_s": round(len(res) / span, 1), "micro_batches": len(batches),
             "batch_size_median": float(np.median([b["windows"] for b in batches])), "windows_by_worker": res.groupby("worker").size().to_dict(),
             "patients": int(res.patient_key.nunique()), "chronological_within_patient": bool(res.sort_values("seq").groupby("patient_key").t_end_sec.apply(lambda s: s.is_monotonic_increasing).all()),
             "scorer": score}
    return stats, res
