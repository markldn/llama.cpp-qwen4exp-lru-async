# llama.cpp — qwen4exp + MTP + RDNA-boosts + GPU-resident LRU expert cache

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that adds native MTP
(multi-token prediction) speculative decoding for **Qwen3.8-Flash-Next** (Alibaba's
`qwen4exp` MoE architecture), an asynchronous, device-side GPU-resident LRU cache for
CPU-offloaded MoE expert weights, and a set of RDNA4 (AMD gfx1201) kernel-level speed
boosts ported on top of both.

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
3. **RDNA-boosts kernel port** — a set of AMD RDNA4 kernel optimizations ([stew675/
   llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)) ported onto
   this fork's `qwen4exp` code path — see below.
4. **Experimental per-batch expert-count override** (`--n-expert-used-*`) — see below.
   Not validated for output quality; off by default.

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

## RDNA-boosts port

[stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) is a
separate line of AMD RDNA4-targeted kernel work against upstream llama.cpp. It was ported
onto this fork's `qwen4exp` code in two stages, cherry-picked commit-by-commit (not squashed
patches) and verified against the authoritative source rather than trusting merge
auto-resolution:

- **Stage 1** (12 commits, generic `ggml-cuda`/CPU kernels, no overlap with the cache code):
  k-quant `mmvq` VDR boosts (Q4_K/Q5_K/Q6_K/Q8_0), fused MoE gate+up+GLU MMQ, `mmvq` decode
  item-split, RDNA4 WMMA flash-attention, BF16 KV cache, and related prefill/decode kernel
  work.
- **Stage 2** (14 commits, `qwen4exp`-specific, touches the same `qwen4exp.cpp` the cache and
  MTP code live in): QSA sparse flash-attention (with Q8_0 KV support), and the
  `GGML_OP_HC_MIX`/`GGML_OP_HC_COMBINE` decode-path fusion campaign (collapses the model's
  hybrid-conv mixing from 6 separate kernel launches down to 1-2 fused ones per call), plus a
  fused indexer top-k op (`GGML_OP_INDEXER_TOPK`) and `SWIGLU_CLAMP` (a minimal subset of
  upstream [PR #27930](https://github.com/ggml-org/llama.cpp/pull/27930), ROCm/HIP + CPU
  only).

Real bugs found and independently verified during the port, since a kernel-level port this
size can silently regress correctness or performance without producing a visible error:

- A silent function-signature mismatch after a merge (a dropped `fusion` parameter) — caught
  by inspection before the first build.
- A new prefill-MMQ fusion arm with no batch-size lower bound was stealing MTP's small
  (2-4 token) speculative-decode verify batches away from the fast `mul_mat_vec_q_moe`
  kernel into a batched/prefill-oriented MMQ kernel instead — found via `rocprofv3`
  kernel-trace comparison against the pre-port baseline, fixed by gating the arm to
  `src1->ne[2] > MMVQ_MAX_BATCH_SIZE`.
- A cascading merge conflict had deleted an entire kernel function (`mul_mat_vec_q_moe`)
  from `mmvq.cu` — caught by a genuine compile error, fixed by restoring the function body
  from the source commit.
- `ggml_hc_mix()` hard-asserts Q8_0 weights but the fused call site never checked weight
  type first, so it crashed on context init the instant MTP was enabled (the MTP draft head
  ships as Q4_K_M) — fixed by adding the missing type check; the unfused fallback path was
  already there, just unreachable.
- An out-of-bounds write in the MMQ scatter-quantize path for duplicate ids, and a
  multi-seq decode corruption from a Q8_1 quantize layout mismatch (flat vs. padded
  per-row layout) that only surfaced at `n_seqs > 1` — both caught and fixed independently
  of the RDNA-boosts source.

**Results** (5 trials each, same 7156-token real-code prompt used throughout this fork's
tuning history, full production config: `--ctx-size 262144`, `--moe-expert-cache-experts
110`, MTP + `ngram-mod` chain enabled): stage-1 alone was roughly tied with the pre-port
baseline on decode (within noise) with a +3.2% prefill win. Stage-1+stage-2 together held a
real gain on both: **decode 21.24 t/s avg vs 19.96 t/s baseline (+6.4%)**, **prefill 299.1
t/s avg vs 253.4 t/s baseline (+18.0%)**, with VRAM headroom unchanged or slightly better
(QSA's sparse attention appears to use somewhat less KV working memory). Verified
end-to-end at real production settings (temp 1.0, `--reasoning on`, `--jinja`, full
`ngram-mod,draft-mtp` chain) — coherent output, correct usage accounting, no crash.

## Experimental: per-batch expert-count override

A separate, less-tested addition: flags to override `n_expert_used` (how many MoE experts
are active per token) independently for prefill vs. decode, or derive it adaptively from
observed router confidence:

```
--n-expert-used-prefill N          # override for prefill ubatches only (compute-bound;
                                    # cheaper to trade router precision for speed here)
--n-expert-used-decode N           # override for decode + speculative-verify ubatches
                                    # (memory-bandwidth-bound; changes what every generated
                                    # token actually computes)
--n-expert-used-adaptive           # derive the prefill value from live router confidence
                                    # instead of a fixed number
--n-expert-used-adaptive-log       # log what --n-expert-used-adaptive would pick, without
                                    # applying it — use this to calibrate first
--n-expert-used-adaptive-layer N, --n-expert-used-adaptive-k-min N,
--n-expert-used-adaptive-conf-low F, --n-expert-used-adaptive-conf-high F
                                    # tuning knobs for the adaptive heuristic above
```

The model was not trained at a reduced `n_expert_used`, so any of these can change output
quality, not just speed. Status: `--n-expert-used-decode 4` has one single-trial data point
on the RDNA-boosts build (27.5 t/s vs. ~21.2 t/s at the default K=10 — a promising direction,
not a confirmed average) and only spot-checked coherence, no systematic quality diff against
default output. The adaptive variant is un-benchmarked. Treat all of this as
try-and-compare, not a drop-in default — all disabled (0 / off) unless set.

## Correctness

The cache and MTP integration were verified via greedy (temp=0) decoding on low-entropy
prompts (e.g. "list the first 20 prime numbers") — far more sensitive to a data bug than
stochastic sampling or open-ended creative/technical prompts — against the plain
`--n-cpu-moe` (no cache) build: **bit-identical** output, standalone and with MTP.

If you're validating this on your own box: run the same prompt at `temp=0` (greedy) with the
cache on and off and diff the output byte-for-byte. Greedy decoding surfaces a caching bug
immediately — any wrong expert weight changes the argmax token somewhere in the sequence,
whereas stochastic sampling can mask a bug behind sampling noise for a long time before it's
visible. The RDNA-boosts port was checked the same way (see Results above) plus a
kernel-trace diff against the pre-port baseline to catch a fusion arm silently routing work
through the wrong kernel.

## Benchmarks

Dual-GPU box: AMD Radeon AI PRO R9700 32GB (gfx1201, `main-gpu`) + AMD Radeon RX 9070 16GB
(gfx1201, on a 4-lane PCIe 3.0 link), Qwen3.8-Flash-Next UD-Q4_K_XL, `--tensor-split 87,13`,
`--n-cpu-moe 40`.

The cache-only rows below isolate the cache's contribution with everything else fixed
(no MTP, no RDNA-boosts): single trial each, one real ~8000-token llama.cpp source-code
prompt, `--ctx-size 32768`, greedy (`temp 0`), 200 tokens generated, cache row measured
after a warm-up pass (a cold cache understates its steady-state benefit). Treat single-trial
numbers as noisier than the multi-trial production figure below it, which uses a different,
larger `--ctx-size 262144` config and isn't directly comparable to these two rows:

| Config | Decode | Prefill |
|---|---|---|
| Cache + MTP disabled | 12.2 t/s | 207.0 t/s |
| Cache enabled (MTP still disabled) | 15.1 t/s | 254.1 t/s |
| Cache + MTP + RDNA-boosts (production config, `--ctx-size 262144`, 5-trial pooled avg) | 21.24 t/s | 299.1 t/s |

The cache alone is good for **+24% decode / +23% prefill** here, consistent with the +23%
decode figure quoted elsewhere in this fork's tuning notes for the same isolated
no-cache/no-MTP comparison. The third row is a separate, larger-scale measurement with MTP
and RDNA-boosts also enabled — its own +6.4% decode / +18.0% prefill delta (see RDNA-boosts
port) is against a cache+MTP baseline, not against row 1 or 2 above. A few other things
measured along the way, all reflected in the flags below:

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
been tested on real NVIDIA hardware. Same build steps produce the RDNA-boosts binary; there
is no separate build flag, the boosts are compiled in.

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
something else in your environment disables it. Add `--n-expert-used-decode N` (or the other
`--n-expert-used-*` flags above) only if you've read the Experimental section and want to
try-and-compare — they're off by default.

## License

MIT, same as upstream llama.cpp — see [LICENSE](LICENSE).

---

For everything else (supported models, general server/CLI docs, the broader llama.cpp
project), see [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
