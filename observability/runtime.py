import json
import os
import time
from dataclasses import dataclass, field


SLO_TARGETS = {
    "perception_success_rate": 0.995,
    "perception_p95_latency_ms": 2500.0,
}

ERROR_BUDGET_TARGETS = {
    "perception_failure_ratio": 0.005,
}


@dataclass
class RuntimeObservability:
    structured_logs: bool = field(default_factory=lambda: os.getenv("TM_STRUCTURED_LOGS", "1") == "1")
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    latencies_ms: list = field(default_factory=list)

    def _emit(self, payload):
        if self.structured_logs:
            print(json.dumps(payload, sort_keys=True))

    def record_backend_call(self, backend, success, latency_ms, status_code=None, error_type=None):
        self.call_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.latencies_ms.append(float(latency_ms))

        self._emit(
            {
                "event": "perception_backend_call",
                "ts": time.time(),
                "backend": backend,
                "success": bool(success),
                "latency_ms": round(float(latency_ms), 3),
                "status_code": status_code,
                "error_type": error_type,
            }
        )

    def summary(self):
        total = self.call_count
        success_rate = (self.success_count / total) if total else 1.0
        sorted_latencies = sorted(self.latencies_ms)
        if sorted_latencies:
            p95_index = min(len(sorted_latencies) - 1, int(round(0.95 * (len(sorted_latencies) - 1))))
            p95_latency = sorted_latencies[p95_index]
        else:
            p95_latency = 0.0

        failure_ratio = (self.failure_count / total) if total else 0.0
        return {
            "calls_total": total,
            "calls_success": self.success_count,
            "calls_failure": self.failure_count,
            "success_rate": success_rate,
            "p95_latency_ms": p95_latency,
            "failure_ratio": failure_ratio,
            "slo_targets": dict(SLO_TARGETS),
            "error_budget_targets": dict(ERROR_BUDGET_TARGETS),
            "slo_pass": {
                "perception_success_rate": success_rate >= SLO_TARGETS["perception_success_rate"],
                "perception_p95_latency_ms": p95_latency <= SLO_TARGETS["perception_p95_latency_ms"],
                "perception_failure_ratio": failure_ratio <= ERROR_BUDGET_TARGETS["perception_failure_ratio"],
            },
        }
