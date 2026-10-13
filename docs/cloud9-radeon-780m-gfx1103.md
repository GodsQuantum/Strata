# Cloud9 / AMD Ryzen 7 8845HS Radeon 780M (gfx1103) port

**Developer candidate, not a universally certified release.** This port is maintained in the GodsQuantum fork of upstream [Niko1221/Strata](https://github.com/Niko1221/Strata) (original MIT license retained). Upstream v0.1.43.1 is the base, commit `a0116b2782a5b82622358c04a06496485d12b2d6`.

## Minimal Cloud9 delta

Branch `cloud9/gfx1103-v0.1.43.1`, original port commit `6fd013389436eef0a59ae8180660460cd2157764`. Changes versus upstream are deliberately restricted to:

1. Native gfx1103 handling for sudot4 HIP intrinsics (`include/strata/hip_compat/intrinsics.hpp`).
2. Linux HIP `cudaMemGetInfo` wrapper (`include/strata/hip_compat/cuda_runtime.h`) and UMA planning cap (`src/core/device.cu`).
3. Streaming expert packer (`tools/iq_pack.py`) to avoid allocating concatenated multi-GB intermediate NumPy matrices; preserves row layout by iterating each expert then its gate/up/down tensor views.

The cap is **opt-in** using `STRATA_DEVICE_BUDGET_MIB=8192`. It reports a **planning budget**, not actual dedicated VRAM, and never replaces the AMDGPU kernel's resource accounting. It is possible to benchmark other caps later when the host is healthy. Do **not** change LXC mounts, Proxmox IOMMU settings, `ttm.pages_limit`, or the GPU driver to make this run.

## Previously observed hardware checks (13 Oct 2026)

- Release HIP/gfx1103 build: **PASS**.
- `strata-device --selftest`: Radeon 780M, gfx1103, wave32, 8GiB planning cap, excessive allocation rejected: **PASS**.
- Eight targeted HIP gates: intrinsics, mapped_alias, handoff, expert_cache_staging, router_fast, native_qsa_score, prefill_mmq_parity, prefill_native_batch: **PASS**.
- Measured expert routing 512 experts 31.75 → 11.66 µs and 256 experts 29.02 → 9.83 µs, with no parity divergence observed in the targeted test.
- Full test suite **NOT** green: 5 tests failed/not validated out of 109 (8 skipped). `hip_prompt_attn_wmma`, `ple_parity`, `platform_memory_test`, and `expert_parity`/`pool_test` (the latter two depend on missing `pack/full/experts.bin` test fixture). Do not silently mark these as passing.
- Full Coder IQ1_M *model* A/B and Strata 0.1.43 → 0.1.43.1 speed comparisons on gfx1103 **not completed**. The recorded 5.96 tok/s is historical and not proof of gain from this port. It cannot be compared directly to llama.cpp Flash-Next 17.878 tok/s (different workloads).

## Safe next benchmark protocol (only after GPU recovery)

**First run the Proxmox HOST preflight** in the Cloud9 Engine source, `scripts/host-gpu-preflight.py`. If it reports `REFUSE`, **do not** attempt any Strata HIP kernel, shader compile, or model load. Never assume that CT410 being healthy means the global GPU is healthy.

Host has had two uninterruptible kernel tasks since 13 October: PID 2625774 (`llama-server` / `drm_suballoc_new`) and PID 2683546 (debugfs GTT test / `ttm_bo_evict_first`). At 11:42 CEST GTT = 11,427,979,264 bytes and CT700 `pct exec` still fails. An attempted GPU reset was blocked by authorized remote access policy. Do not bypass that protection, reset driver, or reboot host for this benchmark.

Once GPU is healthy, compare v0.1.43 and v0.1.43.1 on the **same** model weights/quantization, exact prompts, context, expert residency, device budget and cache settings, while verifying the eight HIP gates, *all* suite failures and restore safety. Baseline `STRATA_ADAPT_OVERLAP=0`; only enable by explicit A/B, because upstream 0.1.43.1 disabled default adaptive overlap after regressions on some GPUs. Keep stage pin off by default; test lookahead on/off only with equal resident-expert conditions. Collect warm/cold prefill, decode tok/s, wall time, peak GTT, host memory/swap, answer quality, and repeated runs across French, English, CJK and real agentic tool-use. Never promote based on router microbench alone.

## Provenance and restoration

- Local canonical project folder: `/srv/storage/cloud/arezki/Projets/HomeLab/Moteurs IA/Strata/`.
- Verified Git bundle: `00-handoff/strata-cloud9-gfx1103-v01431-20261013.bundle`, SHA256 `ff6b400f7e2f1d3c98e2ba53737e38172f8eab402dbbad747b1402232b274868`.
- Verified patch: `00-handoff/strata-v01431-gfx1103-20261013.patch`, SHA256 `1aa85eb3e2c783381533eb3a2a551ac0e4f36c2a60350a4cfae4a135e2f70d9e`.
- Candidate binary formerly built in CT700: `strata-runtime/candidates/v01431-6fd01338/strata`, SHA256 `484c69b9bcb64e0ab6868126b0556f5edec1cfdb4030d23de99f7f13c52b9c61`.
- The branch preserved on `GodsQuantum/Strata` does **not** represent a published/benchmarked container image; do not update production runtime or reconfigure other LXCs by merging it.

The fork retains upstream license and attribution; Cloud9-specific changes are isolated in the named branch for upstream rebase and review.
