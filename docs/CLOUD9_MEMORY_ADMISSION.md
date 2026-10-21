# Cloud9 Strata 125B: fail-closed GPU memory admission

The Cloud9 fork must **not** launch a heavy 125B model solely because HIP
initializes successfully or the model shards exist. On integrated Radeon GPUs,
host RAM, GTT and page cache compete for the same physical memory.

`tools/cloud9_strata_admission.py` is a **read-only** client of an
already-running GPU safety guard. It sends `CHECK coexist-light N` and
requires both guard approval and conservative Strata-specific headroom.
It never changes limits, mounts, guard configuration, models or services.

Run from an **authorized GPU container**, with a guard address supplied
at runtime (do not commit private addresses):

```sh
python3 -B tools/cloud9_strata_admission.py --guard-host "$EXISTING_GUARD_HOST" \
  --guard-port 19755 --gtt-request-gib 8 --min-host-available-gib 48
```

Exit codes: **0** = currently eligible for a *controlled attempt*, **2** =
denied or guard unavailable. Exit 0 does **not** certify a full-model load,
correctness, speed, agent tool calls, or persistent service safety. A live
Strata process remains governed by the external guard; do not bypass it.

The 48 GiB default is a conservative starting headroom: roughly 24 GiB of
Coder expert working set plus 24 GiB of host reserve. It is **not** a model
RAM estimate nor a hard upstream requirement. Mapped GGUF experts can page
fault and the GPU's unified-memory allocations can still trip the guard.
Re-evaluate this threshold from real telemetry before enabling unattended
launches. The default requested 8 GiB GTT is also only a preflight budget.

The executable fails closed on a negative guard response, malformed or
missing metrics, insufficient memory, insufficient safe GTT or elevated PSI.
Its address is never embedded in the source, CI or release binary.

Offline contract validation:

```sh
python3 -B tools/test_cloud9_strata_admission.py
```

**Promotion gate:** obtain actual model output, recover from model faults,
run identical quantized workloads against the current llama.cpp production
backend, measure RSS/GTT/pre-fill/decode, validate real read-only agent tool
calls, and pass the GitHub CI. Until then, keep Strata's PR in draft and
the production LLM router unchanged.
