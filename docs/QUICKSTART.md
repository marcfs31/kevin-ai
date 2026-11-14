# Quickstart: Qwen3.5-4B on the OptiPlex

Target: Dell OptiPlex 3050, i5-6500T (4 cores, AVX2), 8 GB RAM, Ubuntu Server 26.04,
238 GB NVMe SSD for the system and weights, 931 GB 5400 rpm HDD (do not put weights here).

Status: phase 1. The engine has not yet been run on the real checkpoint or on the AVX2
build, and the speed below is an estimate, not a measurement. See the README for details.

Run each step in order. Commands run on the server unless stated otherwise.

## 1. Install the build tools

Run this yourself, with sudo:

```sh
sudo apt install build-essential curl
```

The scripts never use sudo.

## 2. Get the source onto the server

Copy the repository to the server, for example with `git clone` from your own remote or
with `rsync` from your Mac. Then change into it:

```sh
cd ~/small-llm-in-c
```

## 3. Build and run the unit tests

```sh
make && make test
```

## 4. Pre-flight check (expected to report missing model files at this point)

```sh
scripts/doctor.sh
```

`doctor.sh` is read-only. Before the download it will report FAIL for the missing model
files. That is expected. It exits non-zero on any FAIL.

## 5. Download Qwen3.5-4B to the SSD

```sh
scripts/download-model.sh --model qwen3.5-4b
```

The default MODEL_DIR is `$HOME/models/qwen3.5-4b`. The script needs about 10.3 GB free
(the 9.32 GB checkpoint plus 1 GB). It refuses a rotational disk unless you pass
`--allow-hdd`. It is resumable: if it stops, run the same command again. It never
overwrites or deletes files; a file with a wrong size or sha256 stops it, and you move
that file aside before re-running.

## 6. Make the q8 file

```sh
bin/lm-quant --model ~/models/qwen3.5-4b --out ~/models/qwen3.5-4b/model.q8.safetensors
```

The q8 file is about 4.75 GB and is what the engine keeps resident in RAM.

## 7. Check the model and the read speed

```sh
scripts/doctor.sh ~/models/qwen3.5-4b --read-speed
```

This now checks the model files, the q8 file, and the RAM available for q8. With
`--read-speed` it times a direct read of the first 2 GiB of the q8 file.

## 8. First chat

```sh
bin/lm-chat --model ~/models/qwen3.5-4b --weights ~/models/qwen3.5-4b/model.q8.safetensors --stats
```

With no `--prompt`, this opens a REPL. Type `/reset` to clear the conversation and
`/exit` to quit.

## Using the tools

`bin/lm-chat`:

```
bin/lm-chat --model DIR --weights FILE [--prompt TEXT] [--system TEXT]
            [--think | --no-think] [--show-thinking]
            [--temperature T] [--top-p P] [--top-k K] [--seed N]
            [--max-tokens N] [--ctx N] [--threads N] [--stats]
```

Thinking is off by default (`--no-think`). With `--prompt` it answers once and exits.

`bin/lm-run` (feeds token ids, writes logits, generates by argmax):

```
bin/lm-run --model DIR [--weights FILE] --tokens 1,2,3 [--logits-out F]
           [--greedy N] [--ctx N] [--threads N]
```

`bin/lm-quant`:

```
bin/lm-quant --model DIR --out FILE
```

## Troubleshooting

- **Out of memory.** Use the q8 file (`--weights ~/models/qwen3.5-4b/model.q8.safetensors`)
  rather than the bf16 shards, and lower `--ctx`. Close other programs and re-run
  `scripts/doctor.sh` to see MemAvailable.
- **Slow.** Confirm the model directory is on the SSD (doctor reports the device). Use q8,
  not bf16: bf16 streams from disk on this 8 GB box and is slower.
- **FAIL on a model file.** The size does not match the pinned checkpoint. Move the file
  aside and run `scripts/download-model.sh --model qwen3.5-4b` again.

## Other developer commands

```sh
make gate       # golden comparison against the oracle (needs tools/ and tests/ fixtures)
make sanitize   # unit tests under ASan and UBSan
make portable   # x86-64 build with -mavx2 -mfma
```

`scripts/download-model.sh --model gemma-4-e4b` fetches the Gemma 4 E4B checkpoint (16 GB).
The engine runs it the same way as Qwen (`bin/lm-quant`, then `bin/lm-chat`); it is
implemented and passes its tiny-model gate, but has not been run on the real weights.
q8 Gemma is about 8.5 GB on disk; only about 5.5 GB is read on every token, because the
per-layer embedding table is read one row at a time. Speed is unmeasured.
