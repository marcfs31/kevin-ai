# Plan: gpt-oss-20b in C99 on an 8 GB CPU box

## Decision

**Model: `openai/gpt-oss-20b`.** Target: Dell OptiPlex 3050, i5-6500T (4 cores, AVX2),
8 GB RAM, 238 GB NVMe SSD, 931 GB 5400 rpm HDD, Ubuntu Server 26.04.

| candidate | checkpoint | why / why not |
|---|---:|---|
| Kimi K3 | 1,561 GB | larger than either disk; ~135 GB read per token |
| GLM-5.2 | 1,507 GB | larger than either disk |
| DeepSeek-V4-Flash | 160 GB | SSD-only fit, tens of GB read per token, FP8 |
| Qwen3.5-35B-A3B | 72 GB | bf16 only: 4x the bytes per weight, or a lossy quantiser we would own |
| Gemma 4 26B-A4B | 52 GB | same |
| gpt-oss-120b | 65 GB | same engine runs it (config-driven); ~2 GB read per token, a later option |
| **gpt-oss-20b** | **13.76 GB** | native MXFP4 experts, 1.7x RAM, Apache-2.0 |

gpt-oss-20b is the only candidate whose experts ship in MXFP4, the format the reference
engine multiplies without dequantising, and the only one where streaming is a modest
stretch (1.7x RAM) rather than a 190x one.

**Repo: from scratch, not a fork.** About 70% of kimi-k3-in-c is K3-specific (KDA, MLA,
attention residuals, latent MoE, trunk streaming, a 112 KB CLI, 30 MB of figures and
fixtures). What transfers is ported file by file under Apache-2.0 with attribution in
`NOTICE`: the safetensors reader, the MXFP4 and bf16 matmul kernels, the expert cache,
the portable I/O shim, the sampler, and the vendored tokenizer/JSON headers.

## Model (normative: OpenAI's `gpt_oss/torch/model.py` and `weights.py`)

Read every number from `config.json` (the `original/` layout). For gpt-oss-20b:
24 layers, hidden 2880, 32 experts, top-4, intermediate 2880, vocab 201088, 64 heads of
dim 64, 8 KV heads, sliding window 128, rope theta 150000, scaling 32, ntk alpha 1,
beta 32, initial context 4096, swiglu limit 7.

Per token position `x` (a vector of `hidden`):

1. `x = embedding[token]`.
2. For each layer `n`: `x = attn(x)`, then `x = mlp(x)`.
3. `logits = unembedding @ rmsnorm(x, norm.scale)` (no bias).

`rmsnorm(x, s) = x * rsqrt(mean(x^2) + 1e-5) * s`.

**Attention.** `t = rmsnorm(x, attn.norm.scale)`; `qkv = qkv.weight @ t + qkv.bias`,
split in this order: q = first `heads*head_dim`, k = next `kv_heads*head_dim`, v = the
last `kv_heads*head_dim`. Query head `h` uses KV head `h / (heads/kv_heads)`. RoPE
rotates q and k per head: split the head's vector into halves `x1, x2`;
`o1 = x1*cos - x2*sin`, `o2 = x2*cos + x1*sin`, with `cos/sin = concentration *
cos/sin(pos * inv_freq[i])`. YaRN: `freq[i] = theta^(2i/head_dim)`; when scaling > 1,
`concentration = 0.1*ln(scaling) + 1`, `low = d_half*ln(ctx0/(beta*2pi))/ln(theta)`,
`high = d_half*ln(ctx0/(alpha*2pi))/ln(theta)`, `ramp = (i - low)/(high - low)`,
`mask = 1 - clamp(ramp, 0, 1)`, `inv_freq = (1/(scaling*freq))*(1 - mask) +
(1/freq)*mask`. Scores `q.k / sqrt(head_dim)` over causal positions; even layers
(0, 2, ...) see only the last `sliding_window` positions including the current one, odd
layers see all. **Sink:** one extra logit `sinks[h]` per head joins the softmax and is
then dropped, so the weights sum to less than 1. Output
`x + out.weight @ concat_heads + out.bias`.

**MoE.** `t = rmsnorm(x, mlp.norm.scale)`; `g = gate.weight @ t + gate.bias`; take the
top-k logits (ties: lowest index), softmax over those k only. For each chosen expert
`e`: `a = mlp1_weight[e] @ t + mlp1_bias[e]` (2*intermediate values, interleaved);
`glu = a[0::2]` clamped to max `limit`, `lin = a[1::2]` clamped to `[-limit, limit]`;
`s = glu * sigmoid(1.702 * glu) * (lin + 1)`; `y_e = mlp2_weight[e] @ s + mlp2_bias[e]`.
Output `x + sum_e w_e * y_e`.

**MXFP4.** `mlpN_weight.blocks` is `[experts, rows, in/32, 16]` U8, `.scales` is
`[experts, rows, in/32]` U8. Low nibble is the even element. Value =
`E2M1[nibble] * 2^(scale - 127)` with `E2M1 = {0, .5, 1, 1.5, 2, 3, 4, 6}` and bit 3 the
sign. Never dequantise an expert into memory.

Tensor names: `embedding.weight`, `unembedding.weight`, `norm.scale`,
`block.N.attn.{norm.scale, qkv.weight, qkv.bias, out.weight, out.bias, sinks}`,
`block.N.mlp.{norm.scale, gate.weight, gate.bias, mlp1_bias, mlp2_bias,
mlp1_weight.blocks, mlp1_weight.scales, mlp2_weight.blocks, mlp2_weight.scales}`.
Everything except the four MXFP4 tensors is BF16.

## Memory design at 8 GB

| component | bytes | residency |
|---|---:|---|
| experts (768 x 13.2 MB) | 10.15 GB | streamed; LRU cache of whole experts |
| attention weights | 1.29 GB | resident, bf16 |
| unembedding | 1.16 GB | resident, bf16 |
| embedding | 1.16 GB | one row read per token |
| KV cache, 4096 positions | 0.4 GB | fp32, every layer keeps every position so `oss_truncate` can rewind |

A token touches 96 experts (1.27 GB). With ~4.3 GB of cache (325 of 768 slots) the
misses are read from NVMe. Expected speed is **an estimate until measured on the box:
roughly 0.5 to 2 tokens/s**. The model directory must live on the SSD, never the HDD.
Cache size changes speed only: logits are identical at every budget, and a test holds
that.

## Layout and contracts

```
include/gptoss/gptoss.h   engine API (token ids in, logits out) -- the contract
src/io/      safetensors reader, positional reads          src/core/   kernels and ops
src/cache/   expert LRU                                     src/model/  config, bind, forward
src/tokenizer/ src/chat/  tokenizer, Harmony, sampler       src/cli/    one binary per file
tools/       Python stdlib oracle and fixture generators    tests/      unit tests, fixtures
```

Headers are included relative to `src/` (`#include "io/oss_st.h"`). The Makefile
discovers sources by wildcard; nobody edits it to add a file.

`bin/gptoss-run --model DIR --tokens 1,2,3 [--logits-out F] [--greedy N]
[--cache-slots N] [--ram-gb X] [--ctx N] [--threads N] [--stats]` feeds the tokens,
writes one little-endian float32 row of `vocab` logits per fed token to `F`, then
generates `N` tokens by argmax (lowest id wins a tie) and prints one id per line on
stdout. Nothing else goes to stdout.

Tiny fixture `tests/fixtures/tiny/`: `config.json`, `model.safetensors`, `golden.json`
= `{"tokens": [...], "logits": [[vocab floats] per token], "router": [[[top-k expert
ids, best first] per layer] per token], "greedy": {"prompt": [...], "out": [...]}}`.
Tiny config: 4 layers, hidden 64, intermediate 96, 8 experts top-2, 8 heads of dim 16,
2 KV heads, vocab 384, sliding window 4, the real RoPE constants.

## Gates

1. `make test` -- unit tests, including MXFP4 matmul against dequantise-then-multiply,
   scalar against SIMD paths, cache eviction, and every heuristic's negative case.
2. `make gate` -- `tools/check_golden.py`: engine logits within `2e-4 * max|logit|` of
   the oracle at every position, argmax equal, greedy ids exactly equal, and the
   logits file byte-identical between the default cache and `--cache-slots <top-k>`.
3. `make sanitize` -- unit tests under ASan and UBSan.
4. On the server only (needs the 13.76 GB checkpoint): AVX2 `make test`, real-vocab
   tokenizer round-trip, a real-weights smoke prompt, and the measured tokens/s.

The oracle is deterministic, so no model verifier stage is needed for the engine; the
orchestrating session reviews the oracle itself line by line against OpenAI's source.

## Tracks

| stage | track | model | owns |
|---|---|---|---|
| A | oracle + fixtures | Sonnet: bounded numeric port with exact acceptance | `tools/`, `tests/fixtures/` |
| A | engine foundation | Sonnet: porting kernels, I/O and cache with self-contained tests | `src/io`, `src/core`, `src/cache`, their tests |
| A | tokenizer + chat | Sonnet: bounded adaptation against a fixed API | `src/tokenizer`, `src/chat`, `src/cli/gptoss-chat.c`, their tests |
| A | server scripts + docs | Haiku: three known files | `scripts/`, `README.md`, `docs/QUICKSTART.md` |
| B | model forward | Sonnet (same agent as foundation): iterate against the oracle | `src/model`, `src/cli/gptoss-run.c` |
| C | integration, gates, review | orchestrator (top model) | merges, final gate |

Each Stage A track runs in its own git worktree and touches only what it owns. Only the
orchestrator runs git.

## Later

- Server bring-up: `scripts/download-model.sh`, `scripts/doctor.sh`, gate 4, tuning.
- An OpenAI-compatible HTTP endpoint and a systemd unit.
- Batched prefill; gpt-oss-120b on the same engine.
