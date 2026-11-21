from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO = Path("/raid/dm236/synpath/repo")
RUN_ROOT = os.environ.get("SYNPATH_DGX_ROOT", "/raid/dm236/synpath")
DATA_MOUNT = os.environ.get("SYNPATH_DGX_DATA_MOUNT", "/raid/dm236/data")
ENV_FILE = os.environ.get("SYNPATH_DGX_ENV_FILE", f"{RUN_ROOT}/dgx.env")
IMAGE = os.environ.get("SYNPATH_DOCKER_IMAGE", "synpath:latest")
CONFIG = "configs/experiments/main_pcam_budget_aucfix.yaml"
LOG_ROOT = Path(RUN_ROOT) / "logs" / "pcam_aucfix_parallel"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

seeds = [0, 1, 2]
generators = ["stylegan2ada", "sd_lora", "dcgan"]
sources = [
    "synthetic_only_true_label",
    "real_plus_synthetic_true_label",
    "synthetic_only_label_global_cluster",
    "real_plus_synthetic_label_global_cluster",
    "synthetic_only_label_local_cluster",
    "real_plus_synthetic_label_local_cluster",
]

jobs: list[tuple[str, list[str]]] = []
for seed in seeds:
    jobs.append((f"real_only_seed{seed}", ["uv", "run", "--extra", "gpu", "synpath", "train-utility", "pcam", "real_only", "--seed", str(seed), "--config", CONFIG]))
for source in sources:
    for gen in generators:
        for seed in seeds:
            jobs.append((f"{gen}_{source}_seed{seed}", ["uv", "run", "--extra", "gpu", "synpath", "train-utility", "pcam", source, "--seed", str(seed), "--synthetic-model", gen, "--config", CONFIG]))

gpus = list(range(8))
running: dict[int, tuple[subprocess.Popen, str, object]] = {}
failures: list[tuple[str, int]] = []
queue = list(enumerate(jobs))

def docker_cmd(job_index: int, gpu: int, args: list[str]) -> list[str]:
    return [
        "docker", "run", "--rm",
        "--name", f"synpath-pcam-aucfix-u{job_index:02d}",
        "--gpus", f"device={gpu}",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "--env-file", ENV_FILE,
        "-v", f"{RUN_ROOT}:{RUN_ROOT}",
        "-v", f"{DATA_MOUNT}:{DATA_MOUNT}:ro",
        "-v", f"{REPO}:/workspace/synpath",
        "-w", "/workspace/synpath",
        IMAGE,
        "bash", "scripts/docker/dgx_entrypoint.sh",
        *args,
    ]

while queue or running:
    for gpu in gpus:
        if gpu in running or not queue:
            continue
        job_index, (name, args) = queue.pop(0)
        log_path = LOG_ROOT / f"{job_index:02d}_{name}.log"
        log = open(log_path, "w")
        print(f"START gpu={gpu} job={job_index:02d} {name} log={log_path}", flush=True)
        proc = subprocess.Popen(docker_cmd(job_index, gpu, args), cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        running[gpu] = (proc, name, log)
    time.sleep(15)
    for gpu, (proc, name, log) in list(running.items()):
        rc = proc.poll()
        if rc is None:
            continue
        log.close()
        print(f"DONE gpu={gpu} {name} rc={rc}", flush=True)
        if rc != 0:
            failures.append((name, rc))
        del running[gpu]

if failures:
    print("FAILURES", failures, flush=True)
    raise SystemExit(1)

report_log = LOG_ROOT / "report.log"
with open(report_log, "w") as log:
    cmd = [
        "docker", "run", "--rm", "--name", "synpath-pcam-aucfix-report",
        "--gpus", "device=0", "--ipc=host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "--env-file", ENV_FILE,
        "-v", f"{RUN_ROOT}:{RUN_ROOT}",
        "-v", f"{DATA_MOUNT}:{DATA_MOUNT}:ro",
        "-v", f"{REPO}:/workspace/synpath",
        "-w", "/workspace/synpath", IMAGE,
        "bash", "scripts/docker/dgx_entrypoint.sh",
        "uv", "run", "--extra", "gpu", "synpath", "report", "--config", CONFIG,
    ]
    print(f"START report log={report_log}", flush=True)
    rc = subprocess.call(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    print(f"DONE report rc={rc}", flush=True)
    raise SystemExit(rc)
