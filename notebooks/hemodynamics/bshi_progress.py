"""Record and display epoch metrics, resource use, and training progress.

A ``Progress`` object is attached to a trainer (``trainer.progress = Progress(dir)``); the trainer calls it every few seconds. It writes, atomically, into ``<dir>``:
  progress.json   latest state (phase, epoch, step, percent, windows/s, loss, learning rate, ETA, last validation, GPU use)
  progress.jsonl  one line per update (at most one every ``log_every_s``) for plots and post-mortems
  progress.png    loss / validation / throughput curves (refreshed every ``plot_every_s`` and at every epoch end)
  progress.html   a self-refreshing page with the table and the picture

Watching it, from anywhere that can read the folder (or reach the desktop through the VPN):
  python -m hemodynamics.bshi_progress status <dir>              one-off text status
  python -m hemodynamics.bshi_progress watch  <dir> [--every 10]  repeating status line
  python -m hemodynamics.bshi_progress serve  <dir> --bind <vpn ip> --port 8899   then open http://<ip>:8899/progress.html in a browser
In a notebook the status line and the picture update in place (``Progress(..., notebook=True)`` is chosen automatically inside Jupyter).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def fmt_seconds(s):
    if s is None or s != s or s < 0:
        return "?"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def gpu_sample():
    try:
        u, m = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0].split(",")
        out = {"gpu_util": float(u), "vram_gb": round(float(m) / 1024, 2)}
        try:
            import psutil
            out["cpu_util"] = psutil.cpu_percent(None)
        except Exception:
            pass
        return out
    except Exception:
        return {}


def stale_seconds(st):
    """Seconds since the run last wrote its state (None when unknown)."""
    try:
        return time.time() - time.mktime(time.strptime(st["updated"], "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return None


def status_line(st, stale_after=180.0):
    if not st:
        return "no progress file yet"
    ph = st.get("phase", "?")
    base = f"[{st.get('label', 'run')}] {ph}"
    age = stale_seconds(st)
    if ph == "paused":
        last = f" | last val AUPRC {st['val']['auprc']:.4f}" if st.get("val") else ""
        return base + f" at epoch {st.get('epoch', 0) + 1}/{st.get('epochs', '?')} | " + st.get("note", "") + last
    if ph not in ("done", "paused") and age is not None and age > stale_after:
        return base + f" | NO UPDATE FOR {fmt_seconds(age)}: the run stopped or is stuck (last state: epoch {st.get('epoch', 0) + 1}/{st.get('epochs', '?')}); run the training cell again to resume"
    if ph in ("train", "validate", "startup", "done"):
        ep = f" | epoch {st.get('epoch', 0) + 1}/{st.get('epochs', '?')}" if ph != "done" else ""
        pct = f" {st.get('pct_epoch', 0):.0f}% of the epoch" if ph == "train" else (f" {st.get('pct_val', 0):.0f}% of validation" if ph == "validate" else "")
        rate = f" | {st.get('windows_per_s', 0):,.0f} windows/s" if st.get("windows_per_s") else ""
        loss = f" | loss {st['loss']:.4f}" if st.get("loss") is not None else ""
        val = f" | last val AUPRC {st['val']['auprc']:.4f} AUROC {st['val']['auroc']:.4f}" if st.get("val") else ""
        eta = f" | run done in ~{fmt_seconds(st.get('eta_run_s'))}" if st.get("eta_run_s") is not None and ph != "done" else ""
        gpu = f" | GPU {st['gpu_util']:.0f}%" if st.get("gpu_util") is not None else ""
        return base + ep + pct + rate + loss + val + gpu + eta + f" | elapsed {fmt_seconds(st.get('elapsed_s'))}"
    return base


class Progress:
    def __init__(self, out_dir, label="run", every_s=10.0, log_every_s=30.0, plot_every_s=60.0, console=True, console_every_s=30.0, notebook=None, html_page=True, gpu=True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.label, self.every_s, self.log_every_s, self.plot_every_s = label, every_s, log_every_s, plot_every_s
        self.console, self.console_every_s, self.html_page, self.gpu = console, console_every_s, html_page, gpu
        self.notebook = self._in_notebook() if notebook is None else notebook
        self.t0 = time.monotonic()
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")
        self._last = -1e9
        self._last_log = self._last_plot = self._last_console = -1e9
        self._mark = None                                   # (time, windows) of the previous update, for the recent rate
        self.rate = None
        self.state = {"label": label, "host": socket.gethostname(), "started": self.started, "phase": "startup", "epoch": 0, "epochs": None, "history": []}
        self.epoch_seconds, self.val_seconds = [], []
        self._epoch_t0 = self._val_t0 = None
        self._handle = None
        self._img_handle = None
        self._img_t = None
        self.write()

    # -- helpers
    @staticmethod
    def _in_notebook():
        try:
            from IPython import get_ipython
            return get_ipython() is not None and "IPKernelApp" in get_ipython().config
        except Exception:
            return False

    def due(self):
        return time.monotonic() - self._last >= self.every_s

    def _atomic(self, name, text):
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        for _ in range(5):
            try:
                tmp.replace(self.dir / name)
                return
            except PermissionError:                         # a viewer has the file open (Windows): try again shortly
                time.sleep(0.2)

    def write(self, force_plot=False):
        st = self.state
        st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st["elapsed_s"] = round(time.monotonic() - self.t0, 1)
        self._atomic("progress.json", json.dumps(st, indent=1, default=float))
        now = time.monotonic()
        if now - self._last_log >= self.log_every_s or force_plot:
            self._last_log = now
            with open(self.dir / "progress.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({k: v for k, v in st.items() if k != "history"}, default=float) + "\n")
        if self.html_page and (force_plot or now - self._last_plot >= self.plot_every_s):
            self._last_plot = now
            try:
                render_png(self.dir)
            except Exception as exc:                        # a plotting problem must never stop the training
                st["plot_error"] = repr(exc)[:120]
            self._atomic("progress.html", render_html(st))
        line = status_line(st)
        if self.console and (now - self._last_console >= self.console_every_s or force_plot):
            self._last_console = now
            if self.notebook:
                self._show_notebook(line)
            else:
                print(line, flush=True)

    def _show_notebook(self, line):
        try:
            from IPython.display import HTML, Image, display
            body = HTML(f"<pre style='margin:0'>{html.escape(line)}</pre>")
            if self._handle is None:
                self._handle = display(body, display_id=True)
            else:
                self._handle.update(body)
            png = self.dir / "progress.png"
            if png.exists() and (self._img_t is None or png.stat().st_mtime > self._img_t):
                self._img_t = png.stat().st_mtime
                img = Image(data=png.read_bytes(), format="png")
                if self._img_handle is None:
                    self._img_handle = display(img, display_id=True)
                else:
                    self._img_handle.update(img)
        except Exception:
            print(line, flush=True)

    # -- the trainer's calls
    def start_run(self, epochs, steps_per_epoch, windows_per_step, start_epoch=0, info=None):
        self.state.update(epochs=int(epochs), steps_per_epoch=int(steps_per_epoch), windows_per_step=int(windows_per_step), epoch=int(start_epoch), phase="startup", info=info or {})
        self.write(force_plot=True)

    def begin_epoch(self, epoch):
        self._epoch_t0 = time.monotonic()
        self._mark = None
        self.rate = None
        self.state.update(epoch=int(epoch), phase="startup", step_in_epoch=0, pct_epoch=0.0, note="waiting for the first batches (data-loader workers starting)")
        self.write()

    def update(self, step, windows, loss=None, lr=None, force=False):
        """Called from the training loop when ``due()``: ``loss`` is a dict of running means for this epoch (or ``None``)."""
        now = time.monotonic()
        if self._mark is not None and now > self._mark[0]:
            inst = (windows - self._mark[1]) / (now - self._mark[0])
            self.rate = inst if self.rate is None else 0.5 * self.rate + 0.5 * inst
        self._mark = (now, windows)
        self._last = now
        st = self.state
        spe = st.get("steps_per_epoch") or 1
        st.update(phase="train", step_in_epoch=int(step), pct_epoch=round(100.0 * step / spe, 1), windows_per_s=round(self.rate, 1) if self.rate else None, lr=lr, note="")
        if loss:
            st["loss"] = float(loss.get("total", float("nan")))
            st["loss_parts"] = {k: round(float(v), 5) for k, v in loss.items()}
        wps = self.rate or (windows / max(1e-9, now - (self._epoch_t0 or now)))
        wps_step = st.get("windows_per_step") or 1
        left_epoch = (spe - step) * wps_step / max(wps, 1e-9)
        st["eta_epoch_s"] = round(left_epoch, 1)
        per_epoch = (sum(self.epoch_seconds) / len(self.epoch_seconds)) if self.epoch_seconds else spe * wps_step / max(wps, 1e-9)
        val_s = (sum(self.val_seconds) / len(self.val_seconds)) if self.val_seconds else 0.15 * per_epoch
        epochs_left = max(0, (st.get("epochs") or 0) - st["epoch"] - 1)
        st["eta_run_s"] = round(left_epoch + val_s + epochs_left * (per_epoch + val_s), 1)
        if self.gpu:
            st.update(gpu_sample())
        self.write()

    def begin_validation(self, batches=None):
        self._val_t0 = time.monotonic()
        self._val_batches = batches
        self.state.update(phase="validate", pct_val=0.0, note="")
        self._last = -1e9
        self.write()

    def validation_batch(self, i):
        if self.due():
            self._last = time.monotonic()
            if self._val_batches:
                self.state["pct_val"] = round(100.0 * (i + 1) / self._val_batches, 1)
            self.write()

    def epoch_done(self, row):
        now = time.monotonic()
        if self._epoch_t0 is not None and self._val_t0 is not None:
            self.epoch_seconds.append(self._val_t0 - self._epoch_t0)
            self.val_seconds.append(now - self._val_t0)
        keep = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items() if isinstance(v, (int, float, bool, str)) or v is None}
        self.state["history"].append(keep)
        self.state["val"] = {"auprc": row.get("val_auprc"), "auroc": row.get("val_auroc"), "brier": row.get("val_brier"), "epoch": row.get("epoch")}
        self.state["best_auprc"], self.state["best_epoch"] = row.get("best_auprc"), self.state.get("best_epoch") if not row.get("improved") else row.get("epoch")
        self.state.update(phase="train", note="epoch finished")
        with open(self.dir / "progress.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "epoch_done", **keep}, default=float) + "\n")
        self.write(force_plot=True)

    def pause(self, note="paused (interrupted); run the training cell again to continue from the last finished epoch"):
        self.state.update(phase="paused", note=note)
        self.write(force_plot=True)

    def finish(self, summary=None):
        self.state.update(phase="done", note="finished", summary=summary or {}, eta_run_s=0.0, pct_epoch=100.0)
        self.write(force_plot=True)


# views
def read_jsonl(path):
    rows = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return rows


def render_png(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import figstyle as F
    out_dir = Path(out_dir)
    rows = read_jsonl(out_dir / "progress.jsonl")
    st = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
    live = [r for r in rows if "event" not in r and r.get("loss") is not None and r.get("steps_per_epoch")]
    spe = st.get("steps_per_epoch") or 1
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    if live:
        x = [r["epoch"] + r.get("step_in_epoch", 0) / spe for r in live]
        ax[0].plot(x, [r["loss"] for r in live], color="#1f77b4", lw=1.2)
    F.finish_axes(ax[0], "training progress (epochs completed)", "loss (running mean, dimensionless)", "training loss")
    hist = [h for h in st.get("history", []) if h.get("val_auprc") is not None]
    if hist:
        e = [h["epoch"] + 1 for h in hist]
        ax[1].plot(e, [h["val_auprc"] for h in hist], marker="o", color="#2ca02c")
        ax[2].plot(e, [h["val_auroc"] for h in hist], marker="s", color="#d62728")
    F.finish_axes(ax[1], "epoch (1 = first)", "validation AUPRC (0-1)", "validation AUPRC (early stopping)")
    F.finish_axes(ax[2], "epoch (1 = first)", "validation AUROC (0-1)", "validation AUROC")
    if live:
        t = [r["elapsed_s"] / 60 for r in live if r.get("windows_per_s")]
        if t:
            ax[3].plot(t, [r["windows_per_s"] for r in live if r.get("windows_per_s")], color="#9467bd", lw=1.2)
    F.finish_axes(ax[3], "wall time since start (min)", "throughput (windows/s)", "training speed")
    fig.suptitle(status_line(st)[:170], fontsize=8)
    fig.tight_layout()
    F.audit_figure(fig, "progress")
    tmp = out_dir / "progress.tmp.png"
    fig.savefig(tmp, dpi=80)
    plt.close(fig)
    for _ in range(5):
        try:
            tmp.replace(out_dir / "progress.png")
            return
        except PermissionError:
            time.sleep(0.2)


def render_html(st):
    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in st.items() if k not in ("history", "info", "summary", "loss_parts"))
    hist = st.get("history", [])[-12:]
    cols = [c for c in ("epoch", "total", "val_auprc", "val_auroc", "train_windows_per_s", "train_seconds") if hist and c in hist[0]]
    table = ("<table border=1 cellpadding=3><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>" +
             "".join("<tr>" + "".join(f"<td>{(h.get(c) if not isinstance(h.get(c), float) else round(h.get(c), 4))}</td>" for c in cols) + "</tr>" for h in hist) + "</table>") if hist else ""
    return (f"<html><head><meta charset='utf-8'><meta http-equiv='refresh' content='15'><title>{html.escape(st.get('label', 'run'))}</title></head><body style='font-family:sans-serif'>"
            f"<h3>{html.escape(status_line(st))}</h3><img src='progress.png?t={int(time.time())}' style='max-width:100%'><h4>last epochs</h4>{table}<h4>state</h4><table border=1 cellpadding=3>{rows}</table></body></html>")


def read_status(d):
    p = Path(d) / "progress.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="watch a BSHI training run")
    ap.add_argument("cmd", choices=("status", "watch", "serve"))
    ap.add_argument("dir")
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--bind", default="127.0.0.1", help="serve: address to listen on (use the VPN address; never 0.0.0.0 on a public network)")
    ap.add_argument("--port", type=int, default=8899)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(status_line(read_status(a.dir)))
    elif a.cmd == "watch":
        try:
            while True:
                st = read_status(a.dir)
                print(time.strftime("%H:%M:%S"), status_line(st), flush=True)
                if st and st.get("phase") == "done":
                    break
                time.sleep(a.every)
        except KeyboardInterrupt:
            pass
    else:
        import functools
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(Path(a.dir).resolve()))
        srv = ThreadingHTTPServer((a.bind, a.port), handler)
        print(f"serving {a.dir} at http://{a.bind}:{a.port}/progress.html  (read-only; Ctrl+C to stop)", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
