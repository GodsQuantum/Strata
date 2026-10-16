#!/usr/bin/env python3
"""Read-only Cloud9 Strata launch preflight. Never starts a model or changes services.

A positive result is permission to begin a controlled QA attempt, NOT a
guarantee that a 125B model will fit or complete generation.
The external memory guard remains the authoritative safety control.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from typing import Any


def review(snapshot: dict[str, Any], min_host_available_gib: float = 48.0,
           request_gtt_gib: float = 8.0, max_psi_full_avg10: float = 0.45) -> dict[str, Any]:
    """Independently verify the guard's answer and conservative model headroom.

    48 GiB default = approximately 24 GiB Coder expert working set plus
    24 GiB host reserve; mapped experts do not guarantee page-cache residency.
    """
    reasons: list[str] = []
    if snapshot.get("allow") is not True:
        reasons += [str(x) for x in snapshot.get("reasons", [])]
        if not reasons:
            reasons.append("host GPU memory guard refused admission")
    for key in ("mem_available_gib", "safe_total_gtt_gib", "psi_memory_full_avg10"):
        value = snapshot.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            reasons.append("missing or invalid guard metric: " + key)
    available = snapshot.get("mem_available_gib")
    safe_gtt = snapshot.get("safe_total_gtt_gib")
    psi = snapshot.get("psi_memory_full_avg10")
    if isinstance(available, (int, float)) and math.isfinite(available):
        if available < min_host_available_gib:
            reasons.append(f"available host memory {available:.2f} GiB below Strata 125B floor "
                           f"{min_host_available_gib:.2f} GiB")
    if isinstance(safe_gtt, (int, float)) and math.isfinite(safe_gtt):
        if safe_gtt < request_gtt_gib:
            reasons.append(f"safe total GTT {safe_gtt:.2f} GiB below requested "
                           f"{request_gtt_gib:.2f} GiB")
    if isinstance(psi, (int, float)) and math.isfinite(psi):
        if psi >= max_psi_full_avg10:
            reasons.append(f"host memory PSI full avg10 {psi:.3f} not safely below "
                           f"{max_psi_full_avg10:.3f}")
    return {"safe_to_attempt": len(reasons) == 0,
            "note": "admission preflight only; inference, quality, and speed not certified",
            "reasons": reasons,
            "observed": {k: snapshot.get(k) for k in (
                "allow", "mem_available_gib", "safe_total_gtt_gib",
                "swap_used_percent", "psi_memory_full_avg10")},
            "request_gtt_gib": request_gtt_gib,
            "required_host_available_gib": min_host_available_gib}


def query_guard(host: str, port: int, mode: str, request_gtt_gib: float) -> dict[str, Any]:
    if mode not in {"coexist-light", "exclusive"}:
        raise ValueError("unexpected guard mode")
    wire = f"CHECK {mode} {request_gtt_gib:.2f}\n".encode("ascii")
    with socket.create_connection((host, port), timeout=3) as sock:
        sock.settimeout(3)
        sock.sendall(wire)
        chunks = []
        size = 0
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > 65536:
                raise ValueError("guard returned an oversized response")
            if b"\n" in chunk:
                break
    data = json.loads(b"".join(chunks).split(b"\n", 1)[0])
    if not isinstance(data, dict):
        raise ValueError("guard response is not a JSON object")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guard-host", required=True, help="host of an already-running GPU guard")
    ap.add_argument("--guard-port", type=int, default=19755)
    ap.add_argument("--gtt-request-gib", type=float, default=8.0)
    ap.add_argument("--min-host-available-gib", type=float, default=48.0)
    ap.add_argument("--max-psi-full-avg10", type=float, default=0.45)
    ap.add_argument("--mode", choices=("coexist-light", "exclusive"), default="coexist-light")
    args = ap.parse_args()
    if not (0 < args.guard_port < 65536 and
            0 < args.gtt_request_gib < 128 and 0 < args.min_host_available_gib < 1024 and
            0 < args.max_psi_full_avg10 <= 100):
        ap.error("invalid preflight bounds")
    try:
        guard = query_guard(args.guard_host, args.guard_port, args.mode, args.gtt_request_gib)
        result = review(guard, args.min_host_available_gib,
                        args.gtt_request_gib, args.max_psi_full_avg10)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"safe_to_attempt": False, "reasons": [f"guard unavailable or malformed: {exc}"],
                  "note": "FAIL CLOSED: do not launch Strata 125B"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["safe_to_attempt"] else 2


if __name__ == "__main__":
    sys.exit(main())
