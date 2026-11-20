# Plan: Qwen3.5-4B and Gemma 4 E4B in C99 on a small CPU box

## Decision

Target: Dell OptiPlex 3050, i5-6500T (4 cores, AVX2, no AVX-512), Ubuntu Server 26.04.1 LTS.
Planned for 8 GB RAM, a 238 GB NVMe SSD and a 931 GB HDD; upgraded 2026-11-16 to about 11 GB
RAM, with the HDD replaced by a 466 GB SATA SSD (spare; the models stay on the NVMe). Owner's choice (2026-11-13): run
**Qwen3.5-4B** and **Gemma 4 E4B** with a from-scratch C engine in the style of
kimi-k3-in-c, replacing the earlier gpt-oss-20b plan.

**Status (2026-11-16):** both models are implemented and run on the real checkpoints on the
target machine. This document is the design and the normative equations. The measured
results are in the README; the build tracks at the end of each phase are the historical plan.

| model | params (text) | checkpoint | order | why |
|---|---:|---:|---|---|
| Qwen3.5-4B | ~4.2 B | 9.32 GB bf16, 2 shards | **phase 1** | its Gated DeltaNet layers are the closest thing to the reference repo's KDA code, so the most transfers |
| Gemma 4 E4B | ~4 B core + per-layer embeddings | 15.99 GB bf16, 1 file | phase 2 | needs pieces the reference has none of: per-layer embeddings, KV sharing across layers, logit softcap |

Both Apache-2.0, ungated. Vision and audio are out of scope (text only); Qwen's MTP head is
used as a speculative draft (see Later).

**Weights: q8 resident, bf16 exact.** bf16 Qwen is 8.4 GB of text weights, more than
the 8 GB the plan was written for. `bin/lm-quant` writes a q8 file (int8, one f32 scale per
32 weights, about 4.75 GB) that the engine mmaps and keeps resident. The original bf16 safetensors also
load directly; on 8 GB they streamed through the page cache (the kimi-k3-in-c
technique), slow but exact; in 11 GB bf16 Qwen stays resident (8.0 GiB measured). Speed was an estimate of 2 to 4
tokens/s for q8; measured on the box it is 2.3 tokens/s generating, bounded by memory bandwidth
(~4.75 GB read per token).

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

Tensor prefix `model.language_model.` (plus the MTP head `mtp.*`, kept since the MTP work);
ignore `model.visual.*`. Embedding
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
[--ctx N] [--threads N] [--profile] [--mtp [--mtp-out F]]` (the flags are described in the
README; this is the original contract) feeds the tokens, writes one little-endian f32 row of `vocab`
logits per fed token to `F`, then generates `N` tokens by argmax (lowest id wins a tie)
and prints one id per line on stdout. Nothing else goes to stdout.

`bin/lm-quant --model DIR --out FILE` writes the q8 file.

Tiny fixture `tests/fixtures/qwen35-tiny/`: `config.json` (real schema, nested
`text_config`, `model_type` qwen3_5), weights in TWO shards plus
`model.safetensors.index.json` (so sharding is tested), the real tensor names including
a few `visual.*` tensors that must be ignored, a full MTP head (`mtp.*`), and:
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

**Memory.** Text weights in q8 are about 8.5 GB: the per-layer table is about
3.2 GB but is read one 12 KB row per token (page cache, evictable), and the shared
layers' unused K/V tensors are never touched, so the hot working set is about 5.3 GB.
Tighter than Qwen. Measured 2026-11-16 with 11 GB: peak 5.7 GiB resident, 1.9 tokens/s
generating. The 3.2 GB table is read one row per token and never resident.

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

## Phase 3: images (normative: transformers vision code)

Both checkpoints carry a vision encoder that the engine does not use yet: Qwen3.5-4B
`model.visual.*` (297 tensors) and Gemma 4 E4B `model.vision_tower.*` plus
`model.embed_vision.*`. Gemma's audio tower and video stay out of scope. Upstream reference:
transformers commit `56547cbb1b39ef75506470902750b4c01fdfff3f` (main, fetched 2026-11-20):
`vision_utils.py`; `models/qwen3_5/modeling_qwen3_5.py` (`Qwen3_5Vision*`,
`Qwen3_5Model.get_rope_index`, `get_vision_position_ids`, `Qwen3_5TextRotaryEmbedding`);
`models/qwen2_vl/image_processing_qwen2_vl.py` (`smart_resize`, `patchify`);
`models/qwen3_vl/processing_qwen3_vl.py`; `models/gemma4/modeling_gemma4.py`
(`Gemma4Vision*`, `Gemma4ClippableLinear`, `Gemma4MultimodalEmbedder`, `Gemma4Model.forward`);
`models/gemma4/image_processing_gemma4.py`, `processing_gemma4.py`. Processor configs at the
pinned revisions: Qwen `preprocessor_config.json` (390 B, sha256 27225450…), Gemma
`processor_config.json` (1689 B, sha256 32bdf45d…). Where this section and those files
disagree, the files win.

**Resampling (both models).** The engine decodes PNG/JPEG to RGB8 and resizes in C with a
separable bicubic filter (Keys a = -0.5, support widened by the scale factor when shrinking,
as PIL does). The reference resizes with torchvision (`antialias=True`), so resized pixels are
not bit-identical; the oracles start from the resized pixel tensor, and the resizer has its
own tests (identity at the same size, a constant image stays constant, symmetry). No resize
happens when the input already has the target size, which is what the web client sends.

### Qwen3.5 vision

- **Grid.** `smart_resize` with factor 32 (patch 16 x merge 2): round each side to a multiple
  of 32; if the area exceeds `max_pixels`, scale by `sqrt(h*w/max_pixels)` and floor to
  multiples of 32; if below `min_pixels` (65536) scale up and ceil. The checkpoint allows
  16,777,216 pixels; the engine caps `max_pixels` at 1,048,576 by default (`--image-max-pixels`),
  which the processor also accepts (`size.longest_edge`). `gh = H/16`, `gw = W/16`.
- **Pixels.** `x/255`, then `(x - 0.5)/0.5` per channel.
- **Patches.** Block-major order over 2x2 merge blocks: `(bh, bw, mh, mw)` with
  `row = 2*bh + mh`, `col = 2*bw + mw`. A patch vector is `[c][t][py][px]` with the frame
  duplicated (`temporal_patch_size` 2), so `patch_embed` (Conv3d 3->1024, kernel 2x16x16, bias)
  equals a 2-D conv with `W[:, :, 0] + W[:, :, 1]`; the engine folds the two slices at load
  (the oracle keeps both, which checks the fold).
- **Position table.** `pos_embed` is 48x48 (2304 rows); per patch, bilinear with
  `align_corners=True` at `(row*(47)/(gh-1), col*(47)/(gw-1))` (taps clamped, size 1 maps to
  0), added after `patch_embed`.
- **Rotary.** head_dim 64, theta 1e4 (config default), 16 inverse frequencies
  `1/1e4^(2i/32)`; per patch angles `[row*f, col*f]` (32) duplicated to 64; `rotate_half` over
  64, applied to q and k in float32.
- **Blocks x24.** `x += proj(attn(LN1(x)))`, `x += fc2(gelu_tanh(fc1(LN2(x))))`; LayerNorm with
  bias, eps 1e-6; `qkv` 1024->3072 with bias (q, k, v in that order, 16 heads of 64); attention
  non-causal over all patches of one image, scale 1/8. Images never attend to each other.
- **Merger.** LayerNorm(1024) per patch, then 4 consecutive patches (one 2x2 block)
  concatenated to 4096, `fc1` 4096->4096, **exact-erf GELU** (`nn.GELU()`, not tanh), `fc2`
  4096->2560. Output rows: `gh*gw/4`, in block order (row-major over the merged grid).
  `deepstack_visual_indexes` is empty: no injection into decoder layers.
- **Into the LM.** Text `<|vision_start|>` (248053), `<|image_pad|>` (248056) x N,
  `<|vision_end|>` (248054), N = `gh*gw/4`; the embeddings of the pad positions are replaced by
  the merger rows in order. The chat template writes the three tokens at the image's place in
  the user content (`add_vision_id` false); system messages may not contain images.
- **M-RoPE positions.** Text token at running position p gets (p, p, p) and p += 1. An image
  block starting at p gets `t = p, h = p + i, w = p + j` for the merged grid `i < gh/2`,
  `j < gw/2` in row-major order, then `p += max(gh, gw)/2`. Decoding continues from p. The
  rotary dim is 64 (`partial_rotary_factor` 0.25 of 256), 32 frequencies, interleaved
  (`mrope_section` [11, 11, 10]): frequency k uses the h position if `k % 3 == 1 && k < 33`,
  the w position if `k % 3 == 2 && k < 30`, else t. With text only all three agree, which is
  why phase 1 needed no change. Linear-attention layers take no positions.
- **Engine consequence.** The RoPE position is no longer the cache index: the state keeps
  `rope_pos` (next position) beside `pos`, snapshots store it, and `lm_reset` zeroes it.

### Gemma 4 vision

- **Resize.** `get_aspect_ratio_preserving_size` exactly: `max_patches = 280 * 9 = 2520`,
  `factor = sqrt(2520*256/(h*w))`, each side floored to a multiple of 48 (patch 16 x pool 3),
  with the reference's zero-side and over-budget cases (an error, not a clamp). Pixels `x/255`
  only (`do_normalize` false); the model then uses `2*(x - 0.5)`.
- **Patches.** Raster order over the patch grid; a patch vector is `[py][px][c]`; position of
  patch (row r, col c) is `(x = c, y = r)`. Padding to 2520 patches in the reference is masked
  out of attention and stripped after pooling, so the engine runs only the real patches.
- **Patch embedding.** `input_proj` 768->768 without bias, plus `table[0][x] + table[1][y]`
  (`position_embedding_table` 2 x 10240 x 768).
- **Layers x16.** `x = x + post_attention_norm(attn(input_norm(x)))`,
  `x = x + post_feedforward_norm(mlp(pre_feedforward_norm(x)))`; RMSNorm with plain `w`
  (eps 1e-6). Attention: 12 heads, 12 KV heads, head_dim 64, `q_norm`/`k_norm` with weight,
  `v_norm` without, scaling 1.0, non-causal over the image's patches. Every linear is a
  clippable linear: clamp the input to `[input_min, input_max]` and the output to
  `[output_min, output_max]` (scalar tensors in the checkpoint). MLP
  `down(gelu_tanh(gate(x)) * up(x))`, 768->3072->768.
- **Rotary.** theta 100 (config, `default` read as `axial`), head_dim 64, 16 inverse
  frequencies `1/100^(2i/32)`; channels 0..31 rotate by x and 32..63 by y, each half with its
  own `rotate_half` over 32 (`apply_multidimensional_rope`).
- **Pooling.** k = 3: the soft token of a patch is `(x/3) + (W/48)*(y/3)` (integer division,
  `W/48` = pooled columns); each soft token is the mean of its 9 patches, then times
  `sqrt(768)` in float32. `standardize` is false. N = patches / 9 rows, raster over the pooled
  grid.
- **Projection.** `embed_vision`: RMSNorm without weight (eps 1e-6) over 768, then
  `embedding_projection` 768->2560 without bias. Not multiplied by the text embedding scale.
- **Into the LM.** `<|image>` (255999), `<|image|>` (258880) x N, `<image|>` (258882); the
  template writes `<|image|>` once and the processor expands it. At image positions the
  per-layer input uses the pad token (id 0) for the table lookup and the pad token's scaled
  embedding for the projection part (`Gemma4Model.forward`). Positions stay sequential, and
  attention stays causal (`use_bidirectional_attention` is null for E4B; the config check
  already refuses a non-null value).

### Engine, quantiser and server contract

- `lm-quant` keeps the vision tensors: 2-D linear weights to q8 like the text model, norms,
  biases, position tables and clamp scalars in their stored dtype. Whether q8 is accurate
  enough for vision is measured (q8 against bf16 encoder output on a real photo, and greedy
  replies), not assumed; if it is not, the vision matrices stay bf16 and that is recorded here.
  A q8 file without vision tensors still loads; vision is then `unavailable`.
- New engine calls (`include/lm/lm.h`): encode an RGB8 image to rows (`lm_image_encode`, with
  the grid it chose) and feed an image block (`lm_prefill_image`), which writes the start
  token, the rows and the end token with the right positions. Images cost memory only while
  encoding (activations for one image), not afterwards.
- `lm-serve`: message content parts `{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}`
  (PNG or JPEG data URLs only; any other URL is a 400, nothing is fetched). Body limit 24 MiB,
  at most 8 images per request, decoded size at most 4096 x 4096 before resizing. Errors are
  400 with `code` `image_unsupported`, `image_invalid`, `image_too_large`, `too_many_images`
  or `context_overflow`. `GET /v1/models` adds `context_length` and
  `capabilities.vision` `{status: ok|unavailable, max_pixels, pixels_per_token, tokens_per_image}`
  (Qwen: `pixels_per_token` 1024; Gemma: `tokens_per_image` up to 280). Usage adds
  `prompt_tokens_details.cached_tokens`.
- Conversation reuse and the prefix cache compare the rendered prompt; an image is rendered
  as its SHA-256, so a different picture never reuses a state built on another.
- Decoding uses vendored `stb_image` (public domain / MIT), PNG and JPEG only, sizes checked
  before allocation, run under `make sanitize` and a short fuzz pass over malformed files.
- `lm-chat --image FILE` (repeatable) attaches images to the first message.

### Gates (phase 3)

1. Oracles `tools/ref_vision_qwen35.py` and `tools/ref_vision_gemma4.py`: standard library,
   float64, ported from the files above; tiny synthetic vision configs in the tiny fixtures.
   `make gate` compares encoder rows (`2e-4 * max|x|`) and the logits after an image block,
   bf16 and q8, plus the M-RoPE positions exactly.
2. Real checkpoints, once, in a throwaway Python environment outside the repo (torch +
   transformers at the commit above): encoder rows for one photo from transformers against
   the engine on the same pixel tensor; the error is recorded here.
3. Server: a photo described correctly by both models, timings and peak RSS recorded.

## Later

Batched prefill, the OpenAI-compatible endpoint (`bin/lm-serve`, with
`scripts/lm-serve.service`) and the prefix cache (`lm_snapshot` / `lm_restore`, one snapshot per
system prompt in `lm_chat`, within a byte budget, `lm-serve --cache-mb`) are done. The context
window is chosen from free memory when not given (`src/model/lm_ctx.c`).

**MTP speculative decoding (Qwen3.5), phase 1 done: engine and `lm-run --mtp`.** The head (as
in vLLM's `qwen3_5_mtp.py`) takes h_p, the main model's final normed hidden at position p, and
the token at p+1: `x = fc([norm_e(embed(t)), norm_h(h_p)])`, one full-attention decoder layer
with its own KV cache at RoPE position p, `norm`, then the shared lm_head; it drafts p+2. All
its norms are zero-centred.
- Invariant: with `pos` tokens fed, the head's cache holds entries 0..pos-2 and `hlast` is
  h_{pos-1}. Every feed or prefill chunk runs the head over the entries its tokens complete.
- `lm_draft(token)` computes entry pos-1 into a provisional slot (the next feed writes the same
  values there) and returns the draft logits; it changes nothing else.
- `lm_verify(tokens, n <= 2)` feeds n tokens as one batch, returns every row, and checkpoints
  the linear-attention state (conv and delta-rule S) after each token but the last.
- `lm_accept(k)` keeps the first k: it restores the checkpoint, `hlast` and `pos`. Full
  attention needs no rollback (slots past pos are rewritten before they are read).
- Oracle: speculative greedy gives exactly the plain greedy ids; `tests/unit/test_mtp.c`
  checks draft, verify and accept bit-exact against plain decoding, and `check_golden.py`
  checks the head's rows against `ref_qwen35.Model.mtp_rows`.
- Measured on the i5 (2026-11-16): 78% of drafts accepted (277 of 353). The speed is 0.80x the
  plain greedy path on main, and 0.91x with the two-token matmul tile (PR 33): the two-token
  verify costs more than one token here, so MTP stays opt-in. The 1.3-1.4x estimate was wrong.
- Next: `lm_chat` / `lm-serve` with speculative sampling (rejection sampling when
  temperature > 0). Not planned while MTP is slower than plain decoding on this CPU.
