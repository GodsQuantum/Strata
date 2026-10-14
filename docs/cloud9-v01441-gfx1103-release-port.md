# Cloud9 Radeon 780M port — Strata 0.1.44.1 (14 October 2026)

**Development candidate only. No production promotion, no speed claim.**

Upstream release: `v0.1.44.1` / `9791ea335ba2001c87b5f783146d6eb08fdd9e5e`.
Based on the previous Cloud9 v0.1.43.1 custom branch (`ecd00164`), preserving the existing certified-source checks, the native gfx1103 signed dot4 selection and Linux HIP UMA device-memory budgeting (`STRATA_DEVICE_BUDGET_MIB`).

The upstream Strata v0.1.44 release adds native gfx1103 WMMA code paths and reports **x1.53 faster decode on another Radeon 780M** with identical answers. This number belongs to the *upstream author*, not to the Cloud9 Hall of Fame. Upstream `docs/AMD_HIP.md` still describes the integrated 780M as unsupported for official end-to-end runs; this fork needs full hardware/model verification.

## Merge specifics

Two real conflicts in `v0.1.43.1 -> v0.1.44.1`:

1. `include/strata/hip_compat/intrinsics.hpp`: upstream now also recognizes `gfx1103` and keeps the same signed 4-byte dot4 path. Use upstream's new conditional; do not apply the old gfx1103 patch again.
2. `tools/iq_pack.py`: the 0.1.44 source adds NVFP4 mixed expert formats with a per-expert **16-byte scale tail** and rejects non-finite/non-positive scales. Its default packer concatenates whole layers into temporary numpy buffers, which undoes the low-RAM improvement required for Cloud9's shared-memory APU. This fork keeps the upstream tail validation but writes each expert's gate/up/down rows plus the 16-byte tail sequentially. Existing Q2/IQ encodings still produce byte-identical expert-major output.

Offline regression tests:
- `python3 -B tests/cloud9_iq_pack_streaming_test.py` (old Q2/IQ layout; 3 tests);
- `python3 -B tests/cloud9_nvfp4_tail_streaming_test.py` (new NVFP4 packed scales, mixed expert and no-concatenation; 4 tests);
- `python3 -B tests/cloud9_iq_pack_numpy_smoke.py` (NumPy required for integration; must actually pass before merge, not silently skip).
- CI `.github/workflows/cloud9-port-contract.yml` executes no-GPU layout contracts on pull requests.

## Unfinished promotion gates

- Native HIP Release build for `gfx1103`; `strata-device --selftest`; relevant HIP kernel tests and full model `ctest` without unexpected failures.
- Same original model weights/quant and context across v0.1.43.1 and v0.1.44.1, at least three actual repetitions. Record decode/prefill tok/s, time, GTT/RAM, CPU pressure and text correctness, as *Strata* (not llama.cpp).
- If using Qwen3.6-35B support newly shipped in 0.1.44, verify format compatibility and keep `STRATA_MMQ_KQ`, adaptive overlap, and async expert tiers off until independently benchmarked.
- API compatibility, tool calls and AutoPublisher's true V22.3 extraction+copy prompts, followed by Archie read-only tool calls. **Never trigger any publication or email** during benchmark.
- Record validated results solely in `/srv/storage/cloud/arezki/Projets/HomeLab/Moteurs IA/HALL_OF_FAME.json`, via its source-backed writer. Do not invent numerical results or overwrite historical records.

**Operational limits:** DEV CT700 rootfs 96% full at port time; do not download large weights or leave unreferenced build trees. Host GPU preflight may refuse transient I/O PSI (>5%) even when GTT is only ~200 MiB. Do not circumvent it, restart Proxmox or stop other sessions' Syncthing/rsync/Filebrowser tasks.
