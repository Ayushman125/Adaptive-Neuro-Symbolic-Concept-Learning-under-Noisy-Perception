import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone

try:
    from evaluation.reporting import ensure_dir
except Exception:
    from reporting import ensure_dir  # type: ignore


def prepare_managed_run(base_out_dir, benchmarks_path, seeds, run_ablations, config_path, argv, run_name=""):
    ensure_dir(base_out_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = (run_name or "managed").strip().replace(" ", "_")
    run_id = f"{timestamp}_{safe_name}"

    run_dir = os.path.join(base_out_dir, run_id)
    ensure_dir(run_dir)

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": run_dir,
        "benchmarks": benchmarks_path,
        "seeds": int(seeds),
        "run_ablations": bool(run_ablations),
        "config": config_path or "",
        "argv": list(argv),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
    }

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    latest_path = os.path.join(base_out_dir, "latest_managed_run.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "run_dir": run_dir}, f, indent=2)

    return run_dir, manifest_path