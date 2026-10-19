"""Spark session factory, cluster report and a trial harness for the desktop (master + worker) + laptop (worker) design. Standalone mode, Windows native, Java 17.

``get_spark(profile, master)``
  "etl"  every worker contributes all its cores; executor memory = the smallest worker's memory minus 1 GB (the executor size is one number for the whole application).
  "gpu"  ONE task slot per machine: ``spark.executor.cores == spark.task.cpus == <cores of the smallest worker>`` and ``spark.cores.max = that * workers``, so Spark's spread-out scheduling
         puts exactly one executor on each machine and a task owns the machine (and its GPU). Tasks call torch with the machine's own CPU threads / DataLoader workers.
  master "auto" (the cluster if its master answers, else local mode), "local", or an explicit ``spark://host:port``.
The driver (this process) must have the package on sys.path; executors get it from PYTHONPATH (set by scripts/spark/env.ps1 for the worker process), NOT from Spark: shipping files with
``spark.submit.pyFiles`` / ``addFile`` makes Spark ``chmod`` the copy, which needs winutils on Windows. ``cluster_report`` therefore compares a code hash per worker.
Executors use ``python`` from the worker's own PATH (env.ps1 puts that machine's venv first) and read data from the worker's own project root (``BSHI_ROOT``): nothing here
depends on the desktop's absolute paths, so the laptop works with its own copy of the data shard. Fixed driver / block-manager ports come from configs/hemodynamics.json (``cluster``)
so a firewall script can open exactly them. Spark WRITES to local Windows paths need winutils (not used): results travel back through Spark, files are written from Python.
"""
from __future__ import annotations
# Configure Spark workers and run independent processing or training tasks.

import hashlib
import json
import os
import socket
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from urllib.request import urlopen

PKG_DIR = Path(__file__).resolve().parent
JDK_PATTERNS = ["C:/Program Files/Eclipse Adoptium/jdk-17*", "C:/Program Files/Eclipse Adoptium/jdk-21*", "C:/Program Files/Java/jdk-17*", "C:/Program Files/Java/jdk-21*",
                "C:/Program Files/Microsoft/jdk-17*", "C:/Program Files/Microsoft/jdk-21*", "C:/Program Files/Amazon Corretto/jdk17*", "C:/Program Files/Zulu/zulu-17*"]


# environment and addresses
def load_config(root="."):
    return json.loads((Path(root) / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))["cluster"]


def lan_ipv4():
    """IPv4 of the interface the default route uses (a UDP connect sends no packet). Hyper-V (192.168.64.x), link-local and loopback addresses are refused."""
    if os.environ.get("BSHI_BIND_HOST"):
        return os.environ["BSHI_BIND_HOST"]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("203.0.113.1", 9))
        ip = s.getsockname()[0]
    if ip.startswith(("169.254.", "192.168.64.", "127.")):
        raise RuntimeError(f"default route resolves to {ip}, which Spark must not use; set BSHI_BIND_HOST to the Wi-Fi/Ethernet IPv4")
    return ip


def vpn_ipv4():
    """IPv4 of a private-network adapter (Tailscale: 100.64.0.0/10; ZeroTier / WireGuard: by adapter name), or ``None`` when no such adapter is up."""
    import ipaddress
    import re

    import psutil
    cgnat = ipaddress.ip_network("100.64.0.0/10")
    for name, addrs in psutil.net_if_addrs().items():
        for a in addrs:
            if a.family == socket.AF_INET and (ipaddress.ip_address(a.address) in cgnat or re.search("tailscale|zerotier|wireguard", name, re.I)):
                return a.address
    return None


def network_choice(cfg):
    return os.environ.get("BSHI_NETWORK") or cfg.get("master_host", "auto")


def master_host(cfg):
    """Address of the master. ``auto`` (default, desktop only or same LAN): the default-route IPv4. ``vpn`` (laptop on another network through Tailscale / WireGuard): the VPN address, and the
    LAN address when no VPN adapter is up, so the desktop alone keeps working. Anything else is taken as a literal address."""
    if os.environ.get("BSHI_MASTER_HOST"):
        return os.environ["BSHI_MASTER_HOST"]
    choice = network_choice(cfg)
    if choice == "vpn":
        ip = vpn_ipv4()
        if ip:
            return ip
        import warnings
        warnings.warn("network 'vpn' requested but no VPN adapter has an address: using the LAN address (desktop only)")
        return lan_ipv4()
    return lan_ipv4() if choice in ("auto", "lan") else choice


def _java_major(home):
    try:
        for line in (Path(home) / "release").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("JAVA_VERSION="):
                return int(line.split("=")[1].strip('"').split(".")[0])
    except OSError:
        pass
    return 0


def find_jdk(cfg=None):
    """JAVA_HOME of a JDK 17 or 21 (Spark 4 needs one of them); the default `java` on this machine may be 1.8."""
    import glob
    cands = [os.environ.get("BSHI_JAVA_HOME"), os.environ.get("JAVA_HOME"), (cfg or {}).get("java_home_desktop")]
    for pat in JDK_PATTERNS:
        cands += sorted(glob.glob(pat), reverse=True)
    for c in cands:
        if c and _java_major(c) in (17, 21):
            return c
    raise RuntimeError("Java 17 or 21 not found (Spark 4). Install Temurin 17 or set BSHI_JAVA_HOME.")


def ensure_env(root="."):
    # Calculate the run settings for the following step.
    cfg = load_config(root)
    os.environ["JAVA_HOME"] = find_jdk(cfg)
    os.environ["PATH"] = os.path.join(os.environ["JAVA_HOME"], "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["SPARK_LOCAL_IP"] = lan_ipv4()
    os.environ.setdefault("BSHI_ROOT", str(Path(root).resolve()))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import pyspark
    os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)           # a user-level SPARK_HOME (another Spark install) would otherwise start a different JVM version than the cluster's
    pysite = str(Path(root).resolve() / "scripts" / "spark" / "pysite")      # flushes the Python worker socket at exit (PySpark 4.0.1 Windows bug, see pysite/sitecustomize.py)
    if pysite not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        os.environ["PYTHONPATH"] = pysite + os.pathsep + os.environ.get("PYTHONPATH", "")
    hh = os.environ.get("HADOOP_HOME")
    if not hh or not os.path.isdir(hh):
        # Hadoop's Shell class refuses to load on Windows unless HADOOP_HOME is an existing folder (a stale machine-level C:\Hadoop does not exist). An empty project-local folder satisfies it
        # for this process only; no winutils is downloaded, so Spark never WRITES local files (results come back through Spark, files are written from Python).
        home = Path(root).resolve() / "logs" / "spark" / "hadoop_home"
        (home / "bin").mkdir(parents=True, exist_ok=True)
        os.environ["HADOOP_HOME"] = str(home)
    return cfg


def code_hash():
    """Short hash of this package's source files (no tests). Every worker reports its own: a different hash on the laptop means a stale copy of the code."""
    h = hashlib.sha1()
    for p in sorted(p for p in PKG_DIR.glob("*.py") if not p.name.startswith("test_")):
        h.update(p.name.encode() + p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:10]


# the master's own view
def master_view(host, ui_port, timeout=5):
    try:
        with urlopen(f"http://{host}:{ui_port}/json", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def alive_workers(view):
    return [w for w in (view or {}).get("workers", []) if w.get("state") == "ALIVE"]


def derive_conf(profile, workers, cfg, host, app_name="bshi", python_exe=None):
    """Spark configuration (a dict) for a profile from the master's worker table. Pure function: unit-tested without Spark."""
    if not workers:
        raise RuntimeError("no ALIVE worker registered with the master")
    # Calculate the window boundary for the following step.
    min_cores = min(int(w["cores"]) for w in workers)
    min_mem_mb = min(int(w["memory"]) for w in workers)
    lo, hi = cfg["block_manager_port_range"]
    conf = {
        "spark.app.name": app_name, "spark.driver.host": host, "spark.driver.bindAddress": host, "spark.driver.port": str(cfg["driver_port"]),
        "spark.driver.blockManager.port": str(lo), "spark.blockManager.port": str(lo + 1), "spark.port.maxRetries": str(hi - lo - 1), "spark.ui.port": str(cfg["driver_ui"]),
        "spark.pyspark.python": "python", "spark.pyspark.driver.python": python_exe or sys.executable,
        "spark.network.timeout": "300s", "spark.executor.heartbeatInterval": "30s", "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.session.timeZone": "UTC", "spark.sql.shuffle.partitions": str(max(8, 2 * sum(int(w["cores"]) for w in workers))),
    }
    if profile == "etl":
        mem = max(1024, min(min_mem_mb - 1024, int(cfg.get("etl_executor_memory_max_gb", 16)) * 1024))
        conf.update({"spark.executor.memory": f"{mem}m", "spark.cores.max": str(sum(int(w["cores"]) for w in workers)), "spark.task.maxFailures": "2"})
    elif profile == "gpu":
        conf.update({"spark.executor.cores": str(min_cores), "spark.task.cpus": str(min_cores), "spark.cores.max": str(min_cores * len(workers)),
                     "spark.executor.memory": f"{min(2048, max(512, min_mem_mb - 1024))}m", "spark.task.maxFailures": "1"})
    else:
        raise ValueError(f"unknown profile {profile!r} (use 'etl' or 'gpu')")
    return conf


def ensure_fs_patch(root="."):
    """Compile scripts/spark/java/NoPermLocalFileSystem.java with the local JDK (no download) into a space-free folder and return it. Structured Streaming checkpoints need file
    permissions to be settable, which on Windows means winutils.exe; this class makes that a no-op (driver only)."""
    import subprocess
    src = Path(root).resolve() / "scripts" / "spark" / "java" / "NoPermLocalFileSystem.java"
    out = Path(tempfile.gettempdir()) / "bshi_spark_fs"
    cls = out / "bshi" / "NoPermLocalFileSystem.class"
    if not cls.exists() or cls.stat().st_mtime < src.stat().st_mtime:
        # Create the destination directory before writing any outputs.
        import glob
        api = glob.glob(os.path.join(os.environ["SPARK_HOME"], "jars", "hadoop-client-api-*.jar"))[0]
        out.mkdir(parents=True, exist_ok=True)
        javac = os.path.join(os.environ["JAVA_HOME"], "bin", "javac.exe")
        p = subprocess.run([javac, "-nowarn", "-cp", api, "-d", str(out), str(src)], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("javac failed: " + p.stderr[-500:])
    return str(out)


_LAST_KEYS: set = set()


def _clear_leaked_properties(new_keys):
    """PySpark launches ONE JVM per Python process and the first session's ``--conf`` values stay behind as JVM system properties (e.g. ``spark.executor.cores`` of a 1-worker session
    would silently apply to the next session). Clear the keys the previous session set and this one does not."""
    from pyspark import SparkContext
    gw = SparkContext._gateway
    if gw is not None:
        for k in _LAST_KEYS - set(new_keys):
            gw.jvm.System.clearProperty(k)


def get_spark(profile="etl", master="auto", app_name=None, root=".", extra=None, wait_executors=45, workers=None, streaming=False):
    """SparkSession on the cluster (or local mode). See the module docstring for the profiles. ``workers=1`` uses only the largest worker (for 1-worker vs all-workers comparisons);
    ``streaming=True`` enables checkpoint writes on Windows (see ``ensure_fs_patch``)."""
    cfg = ensure_env(root)
    from pyspark.sql import SparkSession
    host = master_host(cfg)
    app = app_name or f"bshi-{profile}"
    view = None
    if master in ("auto", None):
        view = master_view(host, cfg["master_ui"])
    elif str(master).startswith("spark://"):
        view = master_view(str(master)[8:].split(":")[0], cfg["master_ui"])
    standalone = view is not None and bool(alive_workers(view))
    if master == "local" or (master in ("auto", None) and not standalone):
        if master != "local":
            print("[cluster] no running master with workers: using local mode (start scripts/spark/start_master.ps1 and start_worker.ps1)")
        conf = {"spark.app.name": app, "spark.driver.host": host, "spark.driver.bindAddress": host, "spark.pyspark.python": sys.executable, "spark.pyspark.driver.python": sys.executable,
                "spark.ui.port": str(cfg["driver_ui"]), "spark.sql.execution.arrow.pyspark.enabled": "true", "spark.sql.session.timeZone": "UTC", "spark.driver.memory": "4g",
                "spark.sql.shuffle.partitions": "16"}
        url = f"local[{max(2, (os.cpu_count() or 4) // 2)}]"
    else:
        if not standalone:
            raise RuntimeError(f"master {master} is not reachable or has no ALIVE workers")
        sel = alive_workers(view)
        if workers == 1:                                              # only the largest worker: one executor that no smaller worker can host
            sel = [max(sel, key=lambda w: (int(w["cores"]), int(w["memory"])))]
        conf = derive_conf(profile, sel, cfg, host, app)
        if workers == 1:
            conf["spark.executor.cores"] = str(int(sel[0]["cores"]))
            conf["spark.cores.max"] = str(int(sel[0]["cores"]) if profile == "etl" else conf["spark.cores.max"])
        url = view["url"] if master in ("auto", None) else master
    if streaming:
        conf["spark.hadoop.fs.file.impl"] = "bshi.NoPermLocalFileSystem"
        conf["spark.hadoop.fs.file.impl.disable.cache"] = "true"
        conf["spark.sql.streaming.checkpointFileManagerClass"] = "org.apache.spark.sql.execution.streaming.FileSystemBasedCheckpointFileManager"      # the FileContext one bypasses fs.file.impl
        conf["spark.driver.extraClassPath"] = ensure_fs_patch(root)
        conf["spark.driver.extraJavaOptions"] = "-Xss16m"            # the stream thread redacts the (large, pickled) plan with a recursive regex: the default stack overflows
    conf.update(extra or {})
    # PySpark's driver side reads the worker interpreter from the PYSPARK_PYTHON ENVIRONMENT variable (not from spark.pyspark.python): "python" is resolved on each executor through that
    # machine's own PATH (the laptop's venv), a full desktop path would not exist there
    os.environ["PYSPARK_PYTHON"] = sys.executable if url.startswith("local") else "python"
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()
    global _LAST_KEYS
    _clear_leaked_properties(conf)
    _LAST_KEYS = set(conf)
    b = SparkSession.builder.master(url)
    for k, v in conf.items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    if url.startswith("spark://"):
        want = len(alive_workers(view))
        t0 = time.time()
        while time.time() - t0 < wait_executors:
            n = len(spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()) - 1      # minus the driver
            if n >= want:
                break
            time.sleep(0.5)
    spark._bshi = {"profile": profile, "master": url, "conf": conf, "master_ui": f"http://{host}:{cfg['master_ui']}", "driver_ui": f"http://{host}:{cfg['driver_ui']}"}
    return spark


# start / stop (PowerShell scripts)
SIM_WORKERS = ({"Name": "desktop", "Port": 7078, "WebUi": 8081, "Cores": 8, "Memory": "24g"},
               {"Name": "sim2", "Port": 7079, "WebUi": 8082, "Cores": 4, "Memory": "6g"})


def run_script(name, *args, root=".", timeout=240):
    """Run scripts/spark/<name> and return its text. Output goes to a file (not a pipe): the hidden Spark processes inherit the handle and would otherwise keep a pipe open."""
    import subprocess
    log = Path(tempfile.gettempdir()) / f"bshi_{Path(name).stem}.log"
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(Path(root).resolve() / "scripts" / "spark" / name), *[str(a) for a in args]]
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return log.read_text(encoding="utf-8", errors="replace")


def cluster_up(root=".", workers=SIM_WORKERS):
    """Start the master and the given workers on THIS machine unless a master with workers already answers (idempotent). Returns the master's worker table."""
    cfg = load_config(root)
    host = master_host(cfg)
    if not alive_workers(master_view(host, cfg["master_ui"])):
        print(run_script("start_master.ps1", root=root))
    have = {w["id"] for w in alive_workers(master_view(host, cfg["master_ui"]))}
    for w in workers:
        if not any(f"-{w['Port']}" in i for i in have):
            print(run_script("start_worker.ps1", "-Name", w["Name"], "-Port", w["Port"], "-WebUi", w["WebUi"], "-Cores", w["Cores"], "-Memory", w["Memory"], root=root))
    for _ in range(30):
        if len(alive_workers(master_view(host, cfg["master_ui"]))) >= len(workers):
            break
        time.sleep(1)
    return alive_workers(master_view(host, cfg["master_ui"]))


def cluster_down(root="."):
    return run_script("stop_all.ps1", root=root)


# cluster report
def _probe(_=None):
    """Runs inside a Spark task: which machine, which Python / torch / pyspark / CUDA."""
    import platform
    out = {"worker": os.environ.get("BSHI_WORKER_NAME", "?"), "host": socket.gethostname(), "pid": os.getpid(), "python": platform.python_version(), "executable": sys.executable,
           "bshi_root": os.environ.get("BSHI_ROOT"), "java_home": os.environ.get("JAVA_HOME")}
    try:
        import pyspark
        out["pyspark"] = pyspark.__version__
    except Exception as exc:
        out["pyspark"] = f"missing: {exc!r}"[:80]
    try:
        import numpy
        import pandas
        import pyarrow
        out.update(numpy=numpy.__version__, pandas=pandas.__version__, pyarrow=pyarrow.__version__)
    except Exception as exc:
        out["libs"] = f"missing: {exc!r}"[:80]
    try:
        import torch
        out.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=bool(torch.cuda.is_available()),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        if torch.cuda.is_available():
            out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception as exc:
        out["torch"] = f"missing: {exc!r}"[:80]
    try:
        import hemodynamics
        out["hemodynamics_from"] = os.path.dirname(str(getattr(hemodynamics, "__path__", ["?"])[0]))
        from hemodynamics import cluster as _c
        out["code_hash"] = _c.code_hash()
    except Exception as exc:
        out["hemodynamics_from"] = f"import failed: {exc!r}"[:80]
    time.sleep(1.0)                                                     # keeps the slot busy so the tasks spread over every executor
    return out


VERSION_KEYS = ("python", "pyspark", "numpy", "pandas", "pyarrow", "torch", "cuda_build", "code_hash")


def cluster_report(spark, tasks=None):
    """Workers (the master's view) and what actually runs on them (a tiny task per slot): one row per distinct worker, plus a list of version MISMATCHES between workers."""
    # Calculate the run settings and the observation count for the following step.
    cfg = load_config(os.environ.get("BSHI_ROOT", "."))
    sc = spark.sparkContext
    view = master_view(master_host(cfg), cfg["master_ui"]) or {}
    workers = [{"id": w["id"], "host": w["host"], "cores": w["cores"], "memory_mb": w["memory"], "cores_used": w["coresused"], "state": w["state"]} for w in view.get("workers", [])]
    n = tasks or max(4, 2 * (len(sc._jsc.sc().statusTracker().getExecutorInfos()) - 1))
    probes = sc.parallelize(range(n), n).map(_probe).collect()
    by_worker = {}
    for p in probes:
        by_worker.setdefault(p["worker"], {**p, "tasks_seen": 0})["tasks_seen"] += 1
    mismatches = []
    for k in VERSION_KEYS:
        vals = {w: p.get(k) for w, p in by_worker.items()}
        if len({str(v) for v in vals.values()}) > 1:
            mismatches.append({"key": k, "values": vals})
    return {"master": view.get("url"), "workers_registered": workers, "workers_running_tasks": by_worker, "mismatches": mismatches, "profile": getattr(spark, "_bshi", {}).get("profile"),
            "executor_cores": sc.getConf().get("spark.executor.cores", "all"), "task_cpus": sc.getConf().get("spark.task.cpus", "1"), "default_parallelism": sc.defaultParallelism}


def print_report(rep):
    print(f"master {rep['master']} | profile {rep['profile']} | executor cores {rep['executor_cores']} | task cpus {rep['task_cpus']} | default parallelism {rep['default_parallelism']}")
    for w in rep["workers_registered"]:
        print(f"  registered: {w['id']}  cores {w['cores']}  memory {w['memory_mb']} MB  state {w['state']}")
    for name, p in rep["workers_running_tasks"].items():
        print(f"  ran tasks : {name:8s} host {p['host']}  python {p['python']}  pyspark {p.get('pyspark')}  torch {p.get('torch')} (cuda {p.get('cuda_build')}, gpu {p.get('gpu')})  tasks {p['tasks_seen']}")
    print("  version mismatches between workers: " + (json.dumps(rep["mismatches"]) if rep["mismatches"] else "none"))


# B3: trial harness
def _run_one(payload):
    """One trial inside one Spark task. Never raises: a failure is returned as data so the driver can retry it once and report it."""
    trial_id, cfg, fn_path, attempt = payload
    base = {"trial_id": trial_id, "attempt": attempt, "worker": os.environ.get("BSHI_WORKER_NAME", "?"), "host": socket.gethostname()}
    t0 = time.perf_counter()
    try:
        import importlib
        mod, name = fn_path.split(":")
        fn = getattr(importlib.import_module(mod), name)
        metrics = fn({**cfg, "_trial_id": trial_id, "_attempt": attempt})
        return {**base, "ok": True, "metrics": metrics, "error": None, "seconds": round(time.perf_counter() - t0, 3)}
    except BaseException as exc:                                          # noqa: BLE001 - report everything, including SystemExit from a library
        return {**base, "ok": False, "metrics": None, "error": f"{type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-700:], "seconds": round(time.perf_counter() - t0, 3)}


def run_trials(spark, configs, fn="hemodynamics.cluster:demo_trial", retries=1):
    """One trial per Spark task (the gpu profile gives every machine one slot, so a trial owns that machine and its GPU). ``fn`` is 'module:function' importable on the workers and
    returns a metrics dict. Failed trials are retried ``retries`` times (a retry may land on the other machine) and every failure stays in the report. No shared mutable files: results
    come back through Spark. Returns a list of dicts ordered like ``configs``: trial_id, ok, attempts, worker, host, seconds, metrics, error, history."""
    sc = spark.sparkContext
    final, history = {}, {i: [] for i in range(len(configs))}
    pending = list(range(len(configs)))
    for attempt in range(retries + 1):
        if not pending:
            break
        payload = [(i, configs[i], fn, attempt) for i in pending]
        for res in sc.parallelize(payload, len(payload)).map(_run_one).collect():
            history[res["trial_id"]].append({k: res[k] for k in ("attempt", "ok", "worker", "host", "seconds", "error")})
            final[res["trial_id"]] = res
        pending = [i for i in pending if not final[i]["ok"]]
    return [{**{k: final[i][k] for k in ("trial_id", "ok", "worker", "host", "seconds", "metrics", "error")}, "attempts": len(history[i]), "history": history[i]} for i in range(len(configs))]


def demo_trial(cfg):
    """A cheap deterministic trial for tests and demos: optional sleep, optional failure on a given attempt."""
    if cfg.get("fail_on_attempt") == cfg["_attempt"]:
        raise RuntimeError(f"planned failure on attempt {cfg['_attempt']}")
    time.sleep(float(cfg.get("sleep", 0)))
    x = float(cfg.get("lr", 1e-3))
    return {"score": round(1.0 - abs(x - 3e-3), 6), "lr": x}


def gpu_trial(cfg):
    """A trial that really uses the machine's GPU: a bf16 matmul loop; reports the device and the throughput (the laptop's RTX 4050 will differ from the desktop's RTX 3080 Ti)."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this worker")
    n, iters = int(cfg.get("n", 4096)), int(cfg.get("iters", 20))
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return {"gpu": torch.cuda.get_device_name(0), "bf16_tflops": round(2 * n ** 3 * iters / (time.perf_counter() - t0) / 1e12, 1)}
