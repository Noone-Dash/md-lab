"""A single-node job scheduler — no SLURM needed.

Runs each experiment as its own process inside a systemd *transient scope* so the
kernel enforces CPU (CPUQuota) and memory (MemoryMax) caps via cgroups v2.  A
small queue serialises GPU work and honours a user-set budget so the machine
can't be overwhelmed.  Jobs can be paused (SIGSTOP), resumed (SIGCONT) or killed
— all through the unit's cgroup.  Falls back to plain subprocesses (CPU-affinity
capped) if the systemd user manager isn't available.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import psutil

from .engine import RUNS_DIR
from .recipes import REGISTRY
from .tracks import TRACKS

ROOT = RUNS_DIR.parent.parent
JOBS_DIR = RUNS_DIR.parent / "_jobs"
VENV_PY = sys.executable
GMX_ROOT = os.environ.get("GMX_ROOT", "/home/v_u/Documents/tools/opt/gromacs-2026.2")
TOTAL_CORES = psutil.cpu_count(logical=True) or 4
TOTAL_RAM_GB = round(psutil.virtual_memory().total / 1e9)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _num(s):
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        return None


def _meta(key):
    if key in REGISTRY:
        return REGISTRY[key].meta()
    if key in TRACKS:
        return TRACKS[key].meta()
    return None


def _systemd_ok():
    if not shutil.which("systemd-run"):
        return False
    try:
        r = subprocess.run(["systemd-run", "--user", "--quiet", "--scope", "true"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


HAVE_SYSTEMD = _systemd_ok()


@dataclass
class Budget:
    max_concurrent: int = 2         # total jobs running at once
    max_gpu_jobs: int = 1           # GPU jobs at once (serialise the GPU)
    cores_per_job: int = 6          # CPU cores each job may use
    mem_per_job_gb: int = 24        # hard memory cap per job


@dataclass
class Job:
    id: str
    key: str
    name: str
    track: str
    needs_gpu: bool
    params: dict
    state: str = "queued"           # queued|running|paused|done|error|killed
    unit: str = None
    pid: int = None
    submitted: str = ""
    started: str = None
    finished: str = None
    cores: int = 0
    mem_gb: int = 0
    error: str = None


class Scheduler:
    def __init__(self):
        self.budget = Budget()
        self.jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.RLock()
        self._started = False
        JOBS_DIR.mkdir(parents=True, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------- #
    def start(self):
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def submit(self, key, params):
        m = _meta(key)
        if not m:
            raise KeyError(key)
        if m.get("mode") == "unavailable":
            raise ValueError(f"{key} is not installed")
        with self.lock:
            base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            rid = f"{base}_{key}"
            while rid in self.jobs or (RUNS_DIR / rid).exists():
                base += "x"
                rid = f"{base}_{key}"
        job = Job(id=rid, key=key, name=m["name"], track=m.get("track", "?"),
                  needs_gpu=bool(m.get("needs_gpu", True)), params=params,
                  submitted=_now())
        with self.lock:
            self.jobs[rid] = job
        self._write_stub(job, m)
        self._schedule()
        return rid


    def submit_plan(self, plan_dict):
        """Queue a Plan (dict) — same budget/cgroup/GPU rules as any other job."""
        import json as _json
        from datetime import datetime as _dt
        with self.lock:
            base = _dt.now().strftime("%Y%m%d_%H%M%S_%f")
            rid = f"{base}_plan"
        job = Job(id=rid, key="__plan__", name=plan_dict.get("name", "plan"),
                  track="plan", needs_gpu=True, params={"plan": plan_dict},
                  submitted=_now())
        with self.lock:
            self.jobs[rid] = job
        d = RUNS_DIR / rid; d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(_json.dumps({
            "id": rid, "recipe": "plan", "recipe_name": job.name, "category": "Solvent",
            "track": "plan", "engine": "GROMACS 2026.2", "mode": "live",
            "params": {"stages": len(plan_dict.get("stages", []))}, "status": "queued",
            "created": _now(), "steps": [], "error": None, "outputs": {},
            "energy": None, "analyses": [],
        }, indent=2))
        self._schedule()
        return rid

    # -- scheduling --------------------------------------------------------- #
    def _schedule(self):
        with self.lock:
            running = [j for j in self.jobs.values() if j.state in ("running", "paused")]
            gpu_running = sum(1 for j in running if j.needs_gpu)
            free = self.budget.max_concurrent - len(running)
            for job in sorted((j for j in self.jobs.values() if j.state == "queued"),
                              key=lambda j: j.submitted):
                if free <= 0:
                    break
                if job.needs_gpu and gpu_running >= self.budget.max_gpu_jobs:
                    continue
                avail = psutil.virtual_memory().available / 1e9
                if avail < self.budget.mem_per_job_gb * 0.5:
                    continue                     # not enough RAM headroom — wait
                self._launch(job)
                if job.state == "running":
                    free -= 1
                    if job.needs_gpu:
                        gpu_running += 1

    def _launch(self, job):
        cores = max(1, min(self.budget.cores_per_job, TOTAL_CORES - 2))
        mem = self.budget.mem_per_job_gb
        job.cores, job.mem_gb = cores, mem
        pfile = JOBS_DIR / f"{job.id}.json"
        pfile.write_text(json.dumps(job.params))
        runjob = [VENV_PY, "-m", "labkit.runjob", job.id, job.key, str(pfile)]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GMX_ROOT": GMX_ROOT,
            "CUDA_VISIBLE_DEVICES": "0" if job.needs_gpu else "",
            "OMP_NUM_THREADS": str(cores),
            "GMX_MAXBACKUP": "-1",
        }
        try:
            if HAVE_SYSTEMD:
                unit = f"mdlab-{job.id}"
                job.unit = unit
                cmd = ["systemd-run", "--user", "--quiet", f"--unit={unit}",
                       f"--property=CPUQuota={cores * 100}%",
                       f"--property=MemoryMax={mem}G",
                       "--property=MemorySwapMax=0",
                       f"--working-directory={ROOT}"]
                for k, v in env.items():
                    cmd.append(f"--setenv={k}={v}")
                cmd += ["--"] + runjob
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr.strip()[-300:] or "systemd-run failed")
            else:
                full = dict(os.environ); full.update(env)
                proc = subprocess.Popen(
                    runjob, cwd=str(ROOT), env=full, start_new_session=True,
                    preexec_fn=lambda: os.sched_setaffinity(0, set(range(cores))))
                self._procs[job.id] = proc
                job.pid = proc.pid
            job.state = "running"
            job.started = _now()
        except Exception as e:  # noqa: BLE001
            job.state = "error"
            job.error = f"launch failed: {e}"
            self._mark_manifest(job.id, "error", str(e))

    # -- monitoring --------------------------------------------------------- #
    def _monitor_loop(self):
        while True:
            time.sleep(2)
            try:
                self._poll()
            except Exception:  # noqa: BLE001
                pass

    def _poll(self):
        changed = False
        with self.lock:
            for job in self.jobs.values():
                if job.state not in ("running", "paused"):
                    continue
                if self._alive(job):
                    continue
                st = self._manifest_status(job.id)
                job.state = "error" if st == "error" else "done"
                if job.state == "error":
                    job.error = job.error or self._manifest_error(job.id)
                job.finished = _now()
                self._reset_unit(job)
                changed = True
        if changed:
            self._schedule()

    def _alive(self, job):
        if HAVE_SYSTEMD and job.unit:
            out = subprocess.run(
                ["systemctl", "--user", "show", job.unit + ".service",
                 "-p", "ActiveState", "--value"],
                capture_output=True, text=True).stdout.strip()
            return out in ("active", "activating", "deactivating", "reloading")
        proc = self._procs.get(job.id)
        return proc is not None and proc.poll() is None

    # -- controls ----------------------------------------------------------- #
    def pause(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if job and job.state == "running":
                self._signal(job, "SIGSTOP")
                job.state = "paused"

    def resume(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if job and job.state == "paused":
                self._signal(job, "SIGCONT")
                job.state = "running"

    def kill(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job or job.state not in ("running", "paused", "queued"):
                return
            if job.state == "queued":
                job.state = "killed"; job.finished = _now()
                self._mark_manifest(jid, "killed")
                return
            self._signal(job, "SIGCONT")          # unpause so it can terminate
            if HAVE_SYSTEMD and job.unit:
                subprocess.run(["systemctl", "--user", "stop", job.unit + ".service"],
                               capture_output=True)
                self._reset_unit(job)
            else:
                proc = self._procs.get(jid)
                if proc:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:  # noqa: BLE001
                        pass
            job.state = "killed"; job.finished = _now()
            self._mark_manifest(jid, "killed")
        self._schedule()

    def _signal(self, job, sig):
        try:
            if HAVE_SYSTEMD and job.unit:
                subprocess.run(["systemctl", "--user", "kill", "--kill-whom=all",
                                f"--signal={sig}", job.unit + ".service"],
                               capture_output=True)
            elif job.pid:
                os.killpg(os.getpgid(job.pid), getattr(signal, sig))
        except Exception:  # noqa: BLE001
            pass

    def _reset_unit(self, job):
        if HAVE_SYSTEMD and job.unit:
            subprocess.run(["systemctl", "--user", "reset-failed", job.unit + ".service"],
                           capture_output=True)

    def set_budget(self, **kw):
        with self.lock:
            for k, v in kw.items():
                if hasattr(self.budget, k) and v is not None:
                    setattr(self.budget, k, int(v))
        self._schedule()

    def clear_finished(self):
        with self.lock:
            self.jobs = {k: j for k, j in self.jobs.items()
                         if j.state in ("queued", "running", "paused")}

    # -- manifest helpers --------------------------------------------------- #
    def _write_stub(self, job, m):
        d = RUNS_DIR / job.id
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(json.dumps({
            "id": job.id, "recipe": job.key, "recipe_name": m["name"],
            "category": m.get("category", ""), "track": job.track,
            "engine": m.get("engine", ""), "mode": m.get("mode", "live"),
            "needs_gpu": job.needs_gpu, "params": job.params, "status": "queued",
            "created": _now(), "steps": [], "error": None, "outputs": {},
            "energy": None, "analyses": [],
        }, indent=2))

    def _manifest(self, jid):
        f = RUNS_DIR / jid / "run.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _manifest_status(self, jid):
        return self._manifest(jid).get("status", "done")

    def _manifest_error(self, jid):
        return self._manifest(jid).get("error")

    def _mark_manifest(self, jid, status, error=None):
        f = RUNS_DIR / jid / "run.json"
        if not f.exists():
            return
        try:
            m = json.loads(f.read_text())
            m["status"] = status
            if error:
                m["error"] = error
            f.write_text(json.dumps(m, indent=2))
        except Exception:  # noqa: BLE001
            pass

    # -- readout for the UI ------------------------------------------------- #
    def list_jobs(self):
        with self.lock:
            return [asdict(j) for j in sorted(self.jobs.values(),
                                              key=lambda j: j.submitted, reverse=True)]

    def telemetry(self):
        vm = psutil.virtual_memory()
        with self.lock:
            running = [j for j in self.jobs.values() if j.state == "running"]
            queued = sum(1 for j in self.jobs.values() if j.state == "queued")
            paused = sum(1 for j in self.jobs.values() if j.state == "paused")
        try:
            load = os.getloadavg()
        except Exception:  # noqa: BLE001
            load = (0, 0, 0)
        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "cores": TOTAL_CORES,
            "ram_used_gb": round(vm.used / 1e9, 1),
            "ram_total_gb": round(vm.total / 1e9, 1),
            "ram_pct": vm.percent,
            "load1": round(load[0], 2),
            "gpu": self._gpu(),
            "budget": asdict(self.budget),
            "backend": "systemd cgroups v2" if HAVE_SYSTEMD else "subprocess (affinity)",
            "counts": {"running": len(running), "queued": queued, "paused": paused},
        }

    def _gpu(self):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            name, util, temp, power = [x.strip() for x in out.split(",")]
            return {"name": name, "util": _num(util), "temp": _num(temp),
                    "power": _num(power), "unified_mem": True}
        except Exception:  # noqa: BLE001
            return None


SCHED = Scheduler()
