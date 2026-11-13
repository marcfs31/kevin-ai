# Plan: Qwen3.5-4B and Gemma 4 E4B in C99 on an 8 GB CPU box

## Decision

Target: Dell OptiPlex 3050, i5-6500T (4 cores, AVX2, no AVX-512), 8 GB RAM, 238 GB NVMe
SSD, 931 GB 5400 rpm HDD, Ubuntu Server 26.04. Owner's choice (2026-11-13): run
**Qwen3.5-4B** and **Gemma 4 E4B** with a from-scratch C engine in the style of
kimi-k3-in-c, replacing the earlier gpt-oss-20b plan.

| model | params (text) | checkpoint | order | why |
|---|---:|---:|---|---|
| Qwen3.5-4B | ~4.2 B | 9.32 GB bf16, 2 shards | **phase 1** | its Gated DeltaNet layers are the closest thing to the reference repo's KDA code, so the most transfers |
| Gemma 4 E4B | ~4 B core + per-layer embeddings | 15.99 GB bf16, 1 file | phase 2 | needs pieces the reference has none of: per-layer embeddings, KV sharing across layers, logit softcap |

Both Apache-2.0, ungated. Vision, audio and Qwen's MTP head are out of scope (text only).

**Weights: q8 resident, bf16 exact.** bf16 Qwen is 8.4 GB of text weights, more than
RAM. `bin/lm-quant` writes a q8 file (int8, one f32 scale per 32 weights, about
4.75 GB) that the engine mmaps and keeps resident. The original bf16 safetensors also
load directly; on 8 GB they stream through the page cache (the kimi-k3-in-c
technique), slow but exact. Speed is **an estimate until measured on the box: roughly
2 to 4 tokens/s for q8**, bounded by memory bandwidth (~4.75 GB read per token).

**Repo: from scratch, porting by file** from kimi-k3-in-c (Apache-2.0, attribution in
`NOTICE`): safetensors reader, bf16 and int8 matmul kernels, ShortConv and the
delta-rule recurrence (KDA is a channel-wise relative of Gated DeltaNet), portable I/O,
sampler, vendored tokenizer/JSON headers.

## Phase 1 model: Qwen3.5 text (normative: transformers `modeling_qwen3_5.py`)

Reference copy: `/private/tmp/claude-501/-Users-marcfors--claude-spares-spare-4OZGox/
e66db16d-b4f1-4a8b-b27d-cc43b2cb8009/scratchpad/hf2/modeling_qwen3_5.py`, classes
`Qwen3_5TextRotaryEmbedding`, `Qwen3_5RMSNormGated`, `torch_recurrent_gated_delta_rule`,
`Qwen3_5GatedDeltaNet`, `Qwen3_5Attention`, `Qwen3_5MLP`, `Qwen3_5RMSNorm`,
`Qwen3_5DecoderLayer`, `Qwen3_5TextModel`. Where this section and that file disagree,
the file wins.

Every number comes from `config.json` → `text_config`. Qwen3.5-4B: 32 layers, `layer_types`
3 x linear_attention then 1 x full_attention, repeated; hidden 2560; intermediate 9216;
vocab 248320; tied embeddings; rms_norm_eps 1e-6; full attention 16 heads, 4 KV heads,
head_dim 256, `attn_output_gate` true, rope theta 1e7, partial_rotary_factor 0.25;
linear attention 16 key heads x 128, 32 value heads x 128, conv kernel 4.

Tensor prefix `model.language_model.`; ignore `model.visual.*` and `mtp.*`. Embedding
`embed_tokens.weight` [vocab, hidden], also the unembedding (tied). Final `norm.weight`.

**RMSNorm (zero-centred):** `y = x * rsqrt(mean(x^2) + eps) * (1 + w)`. Used for
`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, final `norm`.
**Gated RMSNorm** (`linear_attn.norm`): `y = x * rsqrt(mean(x^2) + eps) * w * silu(z)`
— plain `w`, NOT `1 + w`.

**Decoder layer:** `h = x + mixer(norm1(x))`; `out = h + mlp(norm2(h))`;
`mlp(t) = down(silu(gate(t)) * up(t))`.

**Full attention** (`self_attn`): `q_proj` gives `heads * head_dim * 2`, viewed per head
as `[head_dim query | head_dim gate]` (the split is inside each head, not across the
whole vector). `q = q_norm(query)` and `k = k_norm(k_proj(x))` per head; `v = v_proj(x)`.
RoPE on the first `rot = head_dim * 0.25 = 64` dims of each head, rotate-half style:
for `i < rot/2`, `inv_freq[i] = theta^(-2i/rot)`, angle `pos * inv_freq[i]`,
`o[i] = x[i]*cos - x[i+rot/2]*sin`, `o[i+rot/2] = x[i+rot/2]*cos + x[i]*sin`; dims
`>= rot` pass through. For text the three mRoPE sections share the same position, so
this is plain RoPE. Causal softmax of `q.k / sqrt(head_dim)`, GQA (query head `h` uses KV
head `h / (heads / kv_heads)`), then `attn = attn * sigmoid(gate)` and `o_proj`.

**Gated DeltaNet** (`linear_attn`), one token at a time:
1. `qkv = in_proj_qkv(x)` (size `2*key_dim + value_dim`), `z = in_proj_z(x)`,
   `b = in_proj_b(x)`, `a = in_proj_a(x)` (one per value head).
2. Causal depthwise conv1d over the last 4 positions of `qkv` (weight `[C, 1, 4]`, no
   bias), then SiLU. The conv state holds the previous 3 inputs, zero at position 0.
3. Split into q, k (key heads x 128) and v (value heads x 128). Each key head serves
   `value_heads / key_heads` CONSECUTIVE value heads (repeat_interleave).
4. `q = l2norm(q) / sqrt(128)`, `k = l2norm(k)`, with `l2norm(x) = x * rsqrt(sum(x^2) + 1e-6)`.
5. `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)` per value head.
6. Per value head with state `S` [128 key x 128 value], zero at position 0:
   `S *= exp(g)`; `kv = S^T k`; `delta = (v - kv) * beta`; `S += k outer delta`;
   `o = S^T q`. Order is load-bearing.
7. `o = gated_rmsnorm(o, z)` per value head, then `out_proj`.

## q8 format (owned by `tools/quant_ref.py`, mirrored by `bin/lm-quant`)

Quantise every 2-D text weight matrix (`*.weight` with 2 dims, including the
embedding); keep norms, `A_log`, `dt_bias` and the conv weight in their original dtype.
For a row-major `[out, in]` matrix and each group of 32 consecutive inputs in a row:
`absmax` = max |x| over the bf16-widened group; `scale = (float)(absmax / 127.0)`
computed in double and rounded once to f32; `q = round_half_even((double)x /
(double)scale)` clamped to [-127, 127]; `scale == 0` gives all-zero `q`. `in` must be a
multiple of 32 or the tool fails loudly. Output: one safetensors file, tensor `<name>`
dtype I8 shape `[out, in]` plus `<name>.q8scale` dtype F32 shape `[out, in/32]`,
everything else copied byte-for-byte with its name, header `__metadata__`
`{"format": "lm-q8", "group": "32", "source": "<sha256 of index or file>"}`. The Python
and C writers must produce byte-identical files from the same input.

## Engine layout and contracts

```
include/lm/lm.h           engine API (token ids in, logits out) -- the contract
src/io/      safetensors (sharded via index.json), mmap/pread       src/core/  kernels, ops
src/model/   config, bind, forward per family (qwen35, later gemma4)
src/tokenizer/ src/chat/  tokenizer, chat templates, sampler        src/cli/   one binary per file
tools/       Python stdlib oracle, quantiser reference, fixture generators
tests/       unit tests, fixtures
```

Symbols `lm_`, files `lm_*.c/h`, headers included relative to `src/`. The Makefile
discovers sources by wildcard; nobody edits it to add a file.

`bin/lm-run --model DIR [--weights FILE] --tokens 1,2,3 [--logits-out F] [--greedy N]
[--ctx N] [--threads N]` feeds the tokens, writes one little-endian f32 row of `vocab`
logits per fed token to `F`, then generates `N` tokens by argmax (lowest id wins a tie)
and prints one id per line on stdout. Nothing else goes to stdout.

`bin/lm-quant --model DIR --out FILE` writes the q8 file.

Tiny fixture `tests/fixtures/qwen35-tiny/`: `config.json` (real schema, nested
`text_config`, `model_type` qwen3_5), weights in TWO shards plus
`model.safetensors.index.json` (so sharding is tested), the real tensor names including
a few `visual.*` and `mtp.*` tensors that must be ignored, and:
- `golden_bf16.json` and `golden_q8.json` = `{"tokens": [...], "logits": [[vocab] per
  token], "greedy": {"prompt": [...], "out": [...]}}` — the q8 golden is the oracle run
  on the dequantised q8 weights, so both are exact targets;
- `model.q8.safetensors` from `tools/quant_ref.py`.
Tiny config: 8 layers (2 full), hidden 64, intermediate 128, 4 heads x 32, 2 KV heads,
key heads 2 x 16, value heads 4 x 16, conv 4, vocab 384, partial rotary 0.25, the real
rope theta and eps.

## Gates

1. `make test` — unit tests: kernels against references (bf16, q8, scalar vs SIMD),
   every op against `tests/fixtures/ops/`, and each heuristic's negative case.
2. `make gate` — `tools/check_golden.py`: for bf16 and q8, engine logits within
   `2e-4 * max|logit|` of the oracle at every position, argmax equal, greedy ids
   exactly equal; and `bin/lm-quant` output byte-identical to `model.q8.safetensors`.
3. `make sanitize` — unit tests under ASan and UBSan.
4. On the server only (needs the checkpoint): AVX2 `make test`, real-vocab tokenizer
   round-trip, q8 vs bf16 agreement on a short real prompt (greedy tokens and top-1
   agreement rate reported, not assumed), a smoke chat, measured tokens/s.

The oracle is deterministic, so no model verifier stage is needed for the engine; the
orchestrating session reviews the oracle itself against the transformers source.

## Tracks (phase 1)

| stage | track | model | owns |
|---|---|---|---|
| A | oracle, quantiser reference, fixtures | Sonnet: bounded numeric port, exact acceptance | `tools/`, `tests/fixtures/qwen35-tiny/`, `tests/fixtures/ops/` |
| A | engine foundation | Sonnet: kernels, I/O and ops with self-contained tests | `src/io`, `src/core`, `src/cli/lm-quant.c`, their tests |
| A | tokenizer + chat | Sonnet: bounded adaptation against a fixed API | `src/tokenizer`, `src/chat`, `src/cli/lm-chat.c`, `third_party/tok.h`, their tests |
| A | server scripts + docs | Haiku: four known files | `scripts/`, `README.md`, `docs/QUICKSTART.md` |
| B | Qwen3.5 forward | Sonnet: iterate against the oracle | `src/model`, `src/cli/lm-run.c` |
| C | integration, gates, review | orchestrator | merges, final gate |

## Phase 2: Gemma 4 E4B (after phase 1 passes gate 2)

Reuses io, kernels, q8, tokenizer machinery and chat loop; adds a `gemma4` forward with
per-layer input embeddings (`hidden_size_per_layer_input` 256 x 42 layers, read per
token from the mmapped file), sliding (512) and full attention with different head
dims and RoPE types, KV shared by the last 18 layers, GELU-tanh MLP, final logit
softcap 30, and the Gemma chat template. Its own oracle and tiny fixture, same gates.

## Later

Batched prefill; the MTP head as a speculative draft for Qwen; an OpenAI-compatible
HTTP endpoint and a systemd unit.
