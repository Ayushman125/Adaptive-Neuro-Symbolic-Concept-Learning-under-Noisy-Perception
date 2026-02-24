# Service Level Objectives (SLO) and Error Budget

## Scope

These objectives apply to the System 1 perception backend call path used by `ThinkingMachine.perceive()`.

## SLO Targets

- **Perception success rate**: `>= 99.5%`
- **Perception p95 latency**: `<= 2500 ms`

## Error Budget

- **Perception failure ratio**: `<= 0.5%`

This error budget is calculated as:

`failure_ratio = failed_calls / total_calls`

## Runtime Metrics Source

Runtime metrics are emitted and summarized by `observability/runtime.py`:

- `calls_total`
- `calls_success`
- `calls_failure`
- `success_rate`
- `p95_latency_ms`
- `failure_ratio`
- `slo_pass` verdict map

## Structured Logging

Set `TM_STRUCTURED_LOGS=1` to emit JSON lines for backend calls. Sensitive values are redacted from debug logs.
