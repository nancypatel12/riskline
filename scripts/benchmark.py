"""Measure end-to-end HTTP latency against a real uvicorn process."""

import json
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

N_SINGLE, N_BATCH, BATCH, WARMUP = 2000, 200, 100, 100

SAMPLE = {
    "limit_bal": 50000,
    "education": 2,
    **{f"pay_{m}": 0 for m in range(1, 7)},
    **{f"bill_amt{m}": 20000 for m in range(1, 7)},
    **{f"pay_amt{m}": 2000 for m in range(1, 7)},
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pct(xs: list[float], q: float) -> float:
    return round(statistics.quantiles(xs, n=100)[q - 1], 2)


def timed(client: httpx.Client, path: str, body: dict, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t = time.perf_counter()
        client.post(path, json=body).raise_for_status()
        out.append((time.perf_counter() - t) * 1e3)
    return out


def main() -> None:
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "riskline.serve:app", "--port", str(port), "--log-level", "warning"]
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            for _ in range(100):
                try:
                    client.get("/health").raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(0.1)
            timed(client, "/predict", SAMPLE, WARMUP)
            single = timed(client, "/predict", SAMPLE, N_SINGLE)
            batch = timed(client, "/predict/batch", {"applicants": [SAMPLE] * BATCH}, N_BATCH)
    finally:
        server.terminate()
        server.wait()

    result = {
        "predict": {"n": N_SINGLE, "p50_ms": pct(single, 50), "p95_ms": pct(single, 95), "p99_ms": pct(single, 99)},
        f"batch_{BATCH}": {
            "n": N_BATCH,
            "p50_ms": pct(batch, 50),
            "p95_ms": pct(batch, 95),
            "p99_ms": pct(batch, 99),
            "rows_per_s": round(BATCH * 1e3 / statistics.median(batch)),
        },
        "setup": "1 uvicorn worker, sequential keep-alive client, same host",
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latency.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
