# Plan: Qwen3.5-4B and Gemma 4 E4B in C99 on an 8 GB CPU box

## Decision

Target: Dell OptiPlex 3050, i5-6500T (4 cores, AVX2, no AVX-512), 8 GB RAM, 238 GB NVMe
SSD, 931 GB 5400 rpm HDD, Ubuntu Server 26.04.1 LTS. Owner's choice (2026-11-13): run
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

Upstream reference: `src/transformers/models/qwen3_5/modeling_qwen3_5.py` in the
transformers repository, classes
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
[--ctx N] [--threads N] [--profile]` feeds the tokens, writes one little-endian f32 row of `vocab`
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

## Phase 2: Gemma 4 E4B text (normative: transformers `modeling_gemma4.py`)

Reuses io, kernels, q8 (unchanged format and quantiser), tokenizer machinery, sampler
and chat loop. Upstream reference (transformers main, fetched 2026-11-14), under
`src/transformers/models/gemma4/`:
`modeling_gemma4.py` (classes `Gemma4RMSNorm`, `Gemma4TextMLP`,
`Gemma4TextRotaryEmbedding`, `Gemma4TextAttention`, `eager_attention_forward`,
`Gemma4TextDecoderLayer`, `Gemma4TextScaledWordEmbedding`, `Gemma4TextModel`,
`Gemma4ForCausalLM`), `configuration_gemma4.py` (`kv_sharing_modes`, `__post_init__`),
`modeling_rope_utils.py` (`_compute_proportional_rope_parameters`), `masking_utils.py`
(`sliding_window_overlay`), `config.json`, `generation_config.json`,
`chat_template.jinja`, `tokenizer_config.json`, `gemma_header.json` (real tensor
header), the real `tokenizer.json` from the pinned revision (sha256 91d9cf86…). Where this section
and those files disagree, the files win.

**Config.** `config.json` top-level `model_type` `gemma4`, numbers in `text_config`
(`model_type` `gemma4_text`). E4B: 42 layers; `layer_types` 5 x sliding_attention then
1 x full_attention, repeated (full at 5, 11, ..., 41); hidden 2560; intermediate 10240;
`hidden_activation` gelu_pytorch_tanh; 8 heads, 2 KV heads; `head_dim` 256 (sliding),
`global_head_dim` 512 (full); `sliding_window` 512; `num_kv_shared_layers` 18;
`hidden_size_per_layer_input` 256; `vocab_size` = `vocab_size_per_layer_input` = 262144;
`rms_norm_eps` 1e-6; `final_logit_softcapping` 30; `tie_word_embeddings` true;
`rope_parameters` {sliding_attention: default, theta 1e4; full_attention: proportional,
partial_rotary_factor 0.25, theta 1e6}. Strict like phase 1: every field read, a
missing one is an error. Fail loudly (unsupported, not defaulted) on: `enable_moe_block`
true, `attention_k_eq_v` true, `use_double_wide_mlp` true, `use_bidirectional_attention`
non-null, `num_global_key_value_heads` non-null, an explicit `per_layer_config`, a
`layer_types` whose last entry is not full_attention, `attention_bias` true, a rope
type other than default/proportional, `hidden_size_per_layer_input` 0.

Tensor prefix `model.language_model.`; ignore everything else (vision, audio,
`embed_vision`, `embed_audio`). One file `model.safetensors` (no index).

**Norm.** `rms(x, w) = x * (mean(x^2) + eps)^(-1/2) * w` — plain `w`, NOT `1 + w`
(differs from Qwen). `v_norm` has no weight (`w = 1`).

**Embedding.** `e = embed_tokens[t] * s`, `s = bf16(sqrt(hidden))` — the scale is
rounded to bf16 before use, as the reference does (`embed_scale.to(weight.dtype)`):
50.5 for 2560. Same rule for the per-layer table with `bf16(sqrt(256)) = 16`.

**Per-layer inputs (PLE), once per token before layer 0.** `L` layers, `P` =
`hidden_size_per_layer_input`:
`tok = embed_tokens_per_layer[t] * bf16(sqrt(P))` viewed `[L, P]`;
`ctx = (per_layer_model_projection @ e) * hidden^(-1/2)` (`[L*P, hidden]`, not
bf16-rounded) viewed `[L, P]`, each row `rms(row, per_layer_projection_norm)`;
`ple[l] = (ctx[l] + tok[l]) * 2^(-1/2)`. The table `[vocab, L*P]` (5.6 GB bf16) is read
one row per token, never resident.

**Decoder layer `l`** (`layer_scalar` is a stored `[1]` tensor):
```
h = x + post_attention_layernorm(attn(input_layernorm(x)))
h = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
g = gelu_tanh(per_layer_input_gate @ h) * ple[l]          # [P]
h = h + post_per_layer_input_norm(per_layer_projection @ g)
x' = h * layer_scalar
```
`mlp(t) = down(gelu_tanh(gate(t)) * up(t))`,
`gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.

**Attention.** Head dim `d` = 256 sliding, 512 full. Per head:
`q = rope(rms(q_proj x, q_norm))`, `k = rope(rms(k_proj x, k_norm))`,
`v = rms(v_proj x)` (no weight). Scores `q.k` with scaling **1.0** (no `1/sqrt(d)`), no
attention softcap, GQA head `h` uses KV head `h / (heads / kv_heads)`, softmax, then
`o_proj`. Causal; sliding layers additionally mask key `j` unless `j > i - window` (512
keys including the current one).

**RoPE**, rotate-half over the WHOLE head (pairs `i`, `i + d/2`), angle `pos * f[i]`
for `i < d/2`. Sliding (default): `f[i] = theta^(-2i/d)`. Full (proportional):
`r = int(partial * d / 2)` = 64; `f[i] = theta^(-2i/d)` for `i < r`, `f[i] = 0` for
`r <= i < d/2` (those dims pass through), divided by `factor` when present. Note the
exponent denominator is `d` = 512, not the rotated width (differs from Qwen).

**KV sharing.** `first = layers - num_kv_shared_layers` (24). Layers `>= first` are
"reuse": they have no K/V of their own (their `k_proj`, `v_proj`, `k_norm` tensors exist
in the checkpoint and are ignored) and attend with their own `q` to the K/V cache of the
LAST layer `< first` of the same type (sliding: 22, full: 23), under their own type's
mask. So only layers `< first` keep a KV cache; a sliding cache needs only the last
`window` positions.

**Output.** `logits = cap * tanh((embed_tokens @ rms(x, norm)) / cap)`, `cap` =
`final_logit_softcapping` (the unembedding is the unscaled tied embedding).

**Memory on 8 GB.** Text weights in q8 are about 8.5 GB: the per-layer table is about
3.2 GB but is read one 12 KB row per token (page cache, evictable), and the shared
layers' unused K/V tensors are never touched, so the hot working set is about 5.3 GB.
Tighter than Qwen; same speed caveat: an estimate until measured.

**Tokenizer** (`tokenizer.json` BPE): normalizer replaces `" "` with `"▁"` (U+2581);
the Split pre-tokenizer then never fires, so each text segment between added tokens is
ONE word; BPE with `ignore_merges` false, merges as two-element arrays,
`byte_fallback` true (a char not in the vocab becomes `<0xXX>` tokens, one per UTF-8
byte, that never merge), `fuse_unk` true; no BOS added by the tokenizer (the template
writes `<bos>`). 24 `added_tokens` (all special, `normalized` false) are matched in the
raw text first. Decoder: `▁` -> space, byte-fallback tokens reassembled to UTF-8, no
strip. Must handle long single words without quadratic blow-up on a 2k-token prompt.

**Chat template** (`chat_template.jinja`, text only, no tools):
`<bos>` + [`<|turn>system\n` + trim(system) + `<turn|>\n`] + per message
`<|turn>{user|model}\n` + trim(content) + `<turn|>\n`; generation prompt
`<|turn>model\n`. Thinking off by default (`enable_thinking` false; on adds
`<|think|>\n` at the top of a system turn). Stop ids from `generation_config.json`
`eos_token_id` [1, 106, 50]; sampling defaults temp 1.0, top_k 64, top_p 0.95. Earlier
model turns are re-rendered with `<|channel>...<channel|>` thought blocks stripped —
same verbatim-history default and `--strict-template` option as Qwen.

**Tiny fixture** `tests/fixtures/gemma4-tiny/` (real schema, single
`model.safetensors`, real tensor names incl. ignored vision tensors and the unused
K/V tensors of shared layers): 12 layers, `layer_types` 5 sliding + 1 full twice,
`num_kv_shared_layers` 4 (store layers 7 sliding and 5 full; layers 8–11 reuse),
hidden 64, intermediate 128, 4 heads, 2 KV heads, `head_dim` 16, `global_head_dim` 32,
`sliding_window` 8, `hidden_size_per_layer_input` 32 (the smallest P the q8 group of 32 allows), vocab 384 for both vocab sizes,
`layer_scalar` random in [0.5, 1.5], `final_logit_softcapping` chosen so the cap bites
(max pre-cap |logit| at least cap/2), real rope params. Golden prompt longer than 2 x
window. Same golden files, q8 file and gate as Qwen.

## Tracks (phase 2)

| stage | track | model | owns |
|---|---|---|---|
| A | Gemma oracle, tiny fixture, new op fixtures | Sonnet: bounded numeric port, exact acceptance | `tools/ref_gemma4.py`, `tools/make_tiny_gemma4.py`, `tools/make_ops.py`, `tools/test_tools.py`, `tools/check_golden.py`, `tests/fixtures/gemma4-tiny/`, `tests/fixtures/ops/` |
| A | Gemma tokenizer + template | Sonnet: bounded adaptation, Python reference as oracle | `src/tokenizer`, `src/chat`, `src/cli/lm-chat.c`, `third_party/tok.h`, `tests/fixtures/tok/`, `tools/ref_tok_gemma.py`, their tests |
| B | Gemma forward | Sonnet: iterate against the oracle | `src/model`, `src/core/lm_ops.*`, `src/cli/lm-run.c`, their tests |
| C | integration, gates, review | orchestrator | merges, Makefile `gate`, final gate |

## Later

Batched prefill; the MTP head as a speculative draft for Qwen; an OpenAI-compatible
HTTP endpoint and a systemd unit.
