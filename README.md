# llama.cpp — qwen4exp + MTP + GPU-resident LRU expert cache

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that adds native MTP
(multi-token prediction) speculative decoding for **Qwen3.8-Flash-Next** (Alibaba's
`qwen4exp` MoE architecture), plus an asynchronous, device-side GPU-resident LRU cache for
CPU-offloaded MoE expert weights.

Base: upstream commit `88ddbf0a1` (the commit that merged `qwen4exp` architecture support,
[PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)).

This repo is published as a squashed snapshot (one commit, no incremental history) rather
than the full commit-by-commit history against that base — `git log` here won't show
upstream's history or the intermediate steps that produced this fork. The code is the real,
built-and-measured artifact either way; only the trail of how it was written is missing.

## What's different from upstream

1. **MTP draft-head support** — native speculative decoding for `qwen4exp` (`nextn`/
   `hc_head` tensors, draft-head-only GGUF loading via `-md`).
2. **A GPU-resident LRU expert cache** (`--moe-expert-cache-experts`) — see below.

## The expert cache

`--n-cpu-moe N` keeps the first `N` MoE layers' expert weights in host RAM to fit large MoE
models in limited VRAM, computing those layers' `mul_mat_id` on the CPU every decode step.
Decode on those layers is bound by host RAM bandwidth: every token streams its routed
experts' weights out of system RAM.

This adds a fixed-size pool of frequently-used experts kept resident in VRAM in front of
those CPU-offloaded tensors:

```
--moe-expert-cache-experts N      # slots per cached layer (0 = disabled, default)
--moe-expert-cache-inserts N      # max uploads per cached layer per decode step (default: 2)
```

How it works:

- A cache miss this step just runs the **normal CPU path** for that expert — exactly what
  would happen with the cache off, so the cache only ever removes work, never adds a stall.
  Decode never blocks waiting on a cache miss.
- A background worker thread fills the cache **off the critical path**. The new mapping is
  only published (table swapped) once the upload actually completes, at a *later*
  `llama_moe_cache_step()` call — a running graph can never observe a torn slot.
- Uploads are throttled (`--moe-expert-cache-inserts` per layer per step) so a cold cache
  can't saturate the host↔GPU link.
- Every `--n-cpu-moe`-offloaded expert weight tensor gets host-memory pinned (page-locked in
  place, no copy) at load time — this happens whenever offloaded experts exist, independent
  of whether the cache above is even enabled. It speeds up both the cache's own uploads and
  `ggml-backend-sched`'s separate op-offload path (which streams these same weights to a GPU
  for large batches like prefill, cache or no cache) — see Benchmarks.

## Correctness

Verified via greedy (temp=0) decoding on low-entropy prompts (e.g. "list the first 20 prime
numbers") — far more sensitive to a data bug than stochastic sampling or open-ended
creative/technical prompts — against the plain `--n-cpu-moe` (no cache) build:
**bit-identical** output, standalone and with MTP.

If you're validating this on your own box: run the same prompt at `temp=0` (greedy) with the
cache on and off and diff the output byte-for-byte. Greedy decoding surfaces a caching bug
immediately — any wrong expert weight changes the argmax token somewhere in the sequence,
whereas stochastic sampling can mask a bug behind sampling noise for a long time before it's
visible.

## Benchmarks

Dual-GPU box: AMD Radeon AI PRO R9700 32GB (gfx1201, `main-gpu`) + AMD Radeon RX 9070 16GB
(gfx1201, on a 4-lane PCIe 3.0 link), Qwen3.8-Flash-Next UD-Q4_K_XL, `--ctx-size 262144`,
`--tensor-split 87,13`, `--n-cpu-moe 40`. Figures are pooled averages across multiple trials
of three varied prompts each; run-to-run noise on this box is real (~2-2.5 t/s band) — treat
any single number as ± that, not exact.

| Config | Speed |
|---|---|
| Cache disabled | 10.9 t/s |
| Cache enabled | ~20.8 t/s |

That's **+57%** at matched settings otherwise. A few other things measured along the way,
all reflected in the flags below:

- `HSA_ENABLE_SDMA=1` (not `=0`) — re-enables ROCm's dedicated copy engines for cross-device
  traffic. Measured **+8.9%** on this dual-GPU box; the *opposite* direction on a single-GPU
  box (no cross-device traffic to unblock), so don't carry this over blindly.
- `--spec-type ngram-mod,draft-mtp` instead of just `draft-mtp` — a priority-ordered chain
  (cheap exact n-gram matches get first shot each step, falling back to MTP's learned draft).
  +2.5% measured on novel/non-repetitive content, much larger on repetitive or templated
  content, no measured downside.
- GPU performance level: leave it at **`auto`**, don't lock it to `high`. Counterintuitive,
  but measured **~10% slower** locked to `high` in a repeated A/B on this workload — this
  offload-heavy decode pattern has enough idle gaps between GPU bursts that a locked-high
  clock loses to the driver's own dynamic boosting.
- `--threads`: more isn't better past your physical core count. On this 6-core/12-thread
  CPU, 6 threads beat 4, 8, and 10; **12 threads (the full logical count) hung the server
  outright** rather than just running slower. Don't default to `nproc`.
- `--moe-expert-cache-experts 96` / `--n-cpu-moe 40` / `--moe-expert-cache-inserts 4` are
  this box's measured sweep optimum, not defaults to copy blindly — re-tune for your own
  VRAM budget and GPU count.
- **Prefill**: every `--n-cpu-moe`-offloaded expert weight tensor gets host-memory pinned
  (page-locked in place, no copy) at load time, not just the ones the cache tracks —
  `ggml-backend-sched`'s own op-offload path streams these same weights to a GPU for large
  batches (prefill) regardless of whether the cache is enabled, and that copy is much faster
  from pinned memory (direct async DMA) than from plain mmap'd memory (routed through the
  driver's bounce buffer at roughly half bandwidth). Measured **+41%** prefill throughput
  (253 → 357 tok/s pooled, 35 offloaded layers, ~3600-token prompt) for a fixed one-time load
  cost of **+3.9s** (pinning ~53GB of host memory). This applies with the cache on or off.

Actual speedup depends heavily on model, quant, hardware, and `--n-cpu-moe`/context-size
tuning. The numbers above are one box's measurements, not a general guarantee.

## Building

Standard llama.cpp build process — see [docs/build.md](docs/build.md) for the full matrix.
ROCm/HIP example (used for the numbers above):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1201 -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/lib/cmake" \
    -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=ON -S .
cmake --build build --config Release -j "$(nproc)"
```

Swap `-DAMDGPU_TARGETS=gfx1201` for your GPU's target. CUDA builds should work the same way
with `-DGGML_CUDA=ON` in place of `-DGGML_HIP=ON` — the cache's tensor-set/synchronize calls
go through the standard `ggml-backend` async API, nothing HIP-specific — but this hasn't
been tested on real NVIDIA hardware.

## Running

```bash
./build/bin/llama-server \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf \
  --spec-type ngram-mod,draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 --spec-draft-ngl 999 \
  --moe-expert-cache-experts 96 --moe-expert-cache-inserts 4 \
  --ctx-size 262144 --n-gpu-layers 999 --split-mode layer --tensor-split 87,13 --main-gpu 0 \
  --n-cpu-moe 40 --override-tensor "per_layer_token_embd.*=CPU" \
  --flash-attn on --threads 6 --jinja
```

`HSA_ENABLE_SDMA=1` should already be your ROCm default; only worth setting explicitly if
something else in your environment disables it.

## License

MIT, same as upstream llama.cpp — see [LICENSE](LICENSE).

---

For everything else (supported models, general server/CLI docs, the broader llama.cpp
project), see [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
