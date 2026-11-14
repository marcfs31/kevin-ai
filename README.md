# small-llm-in-c

A C99 CPU inference engine for small language models, written from scratch for an
8 GB machine. Target: a Dell OptiPlex 3050 (Intel i5-6500T, 4 cores, AVX2, no AVX-512),
8 GB RAM, Ubuntu Server 26.04, with the weights on an NVMe SSD.

## Status

- **Qwen3.5-4B (phase 1) and Gemma 4 E4B (phase 2) are both implemented** and pass their
  golden gates on tiny synthetic models, against float64 oracles. The Gemma oracle itself
  agrees with transformers to 5e-9. Nothing here has been run on a real checkpoint yet.
- **Speed is an estimate, not a measurement.** Roughly 2 to 4 tokens/s for the q8 path,
  bounded by memory bandwidth (about 4.75 GB read per token). It has NOT been measured.
- The AVX2 build path has not been run yet, and no real checkpoint has been loaded.

The full plan, with the normative model equations and the test gates, is in
[docs/PLAN.md](docs/PLAN.md).

## The two models

| model | params (text) | checkpoint | order | why |
|---|---:|---:|---|---|
| Qwen3.5-4B | ~4.2 B | 9.32 GB bf16, 2 shards | **phase 1** | its Gated DeltaNet layers are the closest thing to the reference repo's KDA code, so the most transfers |
| Gemma 4 E4B | ~4 B core + per-layer embeddings | 15.99 GB bf16, 1 file | phase 2 (implemented, unmeasured) | needs pieces the reference has none of: per-layer embeddings, KV sharing across layers, logit softcap |

Both are Apache-2.0 and ungated. Vision, audio and Qwen's MTP head are out of scope;
the engine is text only.

## How it works

**Weights: q8 resident, bf16 exact.**

- `bin/lm-quant` writes a q8 file: int8 weights with one f32 scale per 32 weights,
  about 4.75 GB for Qwen3.5-4B. The engine memory-maps it and keeps it resident. This is
  the normal mode on this machine.
- The original bf16 safetensors load directly as well. On 8 GB they stream through the
  page cache. That is slow but exact, and it is the reference for checking q8.

**Long chats stay cheap.** Qwen3.5 mixes linear-attention (Gated DeltaNet) layers with
full-attention layers, three to one. The linear-attention layers keep a fixed-size state
per layer instead of a cache that grows with every token, so memory does not climb
linearly with the length of the conversation. Only the full-attention layers keep a
key/value cache.

**Porting.** The engine is written from scratch and ports code by file from
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) (Apache-2.0; see
[NOTICE](NOTICE)): the safetensors reader, the bf16 and int8 kernels, the delta-rule
recurrence, the sampler, and vendored tokenizer headers.

## Build and test

```sh
make            # build bin/* (one binary per src/cli/*.c)
make test       # build and run every tests/unit/test_*.c
make gate       # compare bin/lm-run against the Python oracle's golden files
make sanitize   # unit tests under ASan and UBSan
make portable   # x86-64 build with -mavx2 -mfma instead of -march=native
```

`make gate` needs the Python oracle and fixtures under `tools/` and `tests/`. Those trees
are not in this worktree yet; `make gate` cannot pass until they are.

Requirements: a C99 compiler (`build-essential` on Ubuntu), `make`, Python 3 for the
oracle, and on the server `curl` and `sha256sum` for the download script.

## Running the server

Step by step, with commands, is in [docs/QUICKSTART.md](docs/QUICKSTART.md). In short:

1. Install the build tools: `sudo apt install build-essential curl` (run by you).
2. Put the source on the server, then `make && make test`.
3. `scripts/doctor.sh` checks CPU flags, memory, the compiler and the model files.
4. `scripts/download-model.sh` fetches the pinned Qwen3.5-4B files to the SSD.
5. `bin/lm-quant --model ~/models/qwen3.5-4b --out ~/models/qwen3.5-4b/model.q8.safetensors`
6. `scripts/doctor.sh ~/models/qwen3.5-4b --read-speed`
7. `bin/lm-chat --model ~/models/qwen3.5-4b --weights ~/models/qwen3.5-4b/model.q8.safetensors --stats`

## Usage

### bin/lm-chat

Chat with the model. With `--prompt` it answers once and exits. Without it, it opens a
REPL; `/reset` clears the conversation and `/exit` quits.

```
bin/lm-chat --model DIR --weights FILE [--prompt TEXT] [--system TEXT]
            [--think | --no-think] [--show-thinking]
            [--temperature T] [--top-p P] [--top-k K] [--seed N]
            [--max-tokens N] [--ctx N] [--threads N] [--stats]
```

- `--think` / `--no-think`: thinking mode. The default is `--no-think`.

### bin/lm-run

Feeds token ids, writes logits, and generates by argmax. This is the tool the golden
gate uses. Nothing but token ids goes to stdout.

```
bin/lm-run --model DIR [--weights FILE] --tokens 1,2,3 [--logits-out F]
           [--greedy N] [--ctx N] [--threads N]
```

`--logits-out F` writes one little-endian f32 row of `vocab` logits per fed token.

### bin/lm-quant

Writes the q8 file from the bf16 checkpoint.

```
bin/lm-quant --model DIR --out FILE
```

## Troubleshooting

- **Out of memory.** Use the q8 file, not bf16 (`--weights` pointing at `model.q8.safetensors`),
  and lower `--ctx`. Close other programs; check `scripts/doctor.sh` for MemAvailable.
- **Slow.** Keep the model directory on the SSD, not the HDD (`doctor.sh` reports this).
  Use q8, not bf16: bf16 streams from disk on this machine and is slower (not yet measured).
- **doctor reports FAIL on a model file.** The size does not match the pinned checkpoint.
  Re-run `scripts/download-model.sh`; it never overwrites a file, so move the bad file aside first.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The Qwen3.5-4B and Gemma 4 E4B
weights are Apache-2.0, published by Qwen and Google respectively.

## Credits

- [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c), the reference this
  engine ports code from by file.
- [Hugging Face transformers](https://github.com/huggingface/transformers), whose
  `modeling_qwen3_5.py` is the normative reference for the Qwen3.5 forward pass.
