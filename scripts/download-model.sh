#!/usr/bin/env bash
# Download the pinned checkpoint files for one supported model into MODEL_DIR.
set -euo pipefail

DEFAULT_MODEL=qwen3.5-4b
ONE_GB=1000000000   # decimal; free-space margin on top of the checkpoint size

usage() {
  cat <<'EOF'
Usage: scripts/download-model.sh [--model NAME] [--allow-hdd] [--help] [MODEL_DIR]

Downloads the pinned checkpoint files for one model into MODEL_DIR.

  --model NAME  qwen3.5-4b (default), gemma-4-e4b, whiterabbitneo-2.5-7b-q4km,
                gemma-4-e4b-uncensored, or one of the ten GGUF ids below
  --allow-hdd   Continue when MODEL_DIR is on a rotational disk. Not recommended: the
                engine reads the weights from this directory and an HDD is far too slow.
  --help        Show this help and exit.

  MODEL_DIR     default: <root>/<NAME>, where <root> is what scripts/models-root.sh prints
                ($LM_MODELS_DIR, else $HOME/models)

qwen3.5-4b (phase 1, Apache-2.0)
  Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    model.safetensors-00001-of-00002.safetensors  5329398688 bytes, sha256 pinned
    model.safetensors-00002-of-00002.safetensors  3990429408 bytes, sha256 pinned
    tokenizer.json                                12807982 bytes, sha256 pinned
    preprocessor_config.json                      390 bytes, sha256 pinned (images)
    config.json, model.safetensors.index.json, tokenizer_config.json, chat_template.jinja
                                                  no pinned size or hash (presence checked)

gemma-4-e4b (phase 2, Apache-2.0)
  google/gemma-4-E4B-it at revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c
    model.safetensors   15992595884 bytes, sha256 pinned
    tokenizer.json      32156428 bytes, sha256 pinned
    processor_config.json  1689 bytes, sha256 pinned (images)
    config.json, tokenizer_config.json, chat_template.jinja, generation_config.json
                        no pinned size or hash (presence checked)

whiterabbitneo-2.5-7b-q4km (GGUF Qwen2.5-Coder 7B fine-tune, Apache-2.0)
  bartowski/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-GGUF at revision 4874da7ad5fa4da4f2584ae64ddc21f5a8e7dfce
  (base model WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B, itself Apache-2.0 on Qwen/Qwen2.5-Coder-7B;
  the licence is also in the file's own metadata, general.license). The Q4_K_M quant, chosen to
  leave an 11 GB machine room for the KV cache (see README):
    WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-Q4_K_M.gguf  4683072032 bytes, sha256 pinned
  The file carries the tokenizer and the chat template; nothing else is needed. Keep exactly
  one .gguf in the directory: the engine treats a directory with one .gguf and no config.json
  as a GGUF model.

gemma-4-e4b-uncensored (GGUF Gemma 4 E4B fine-tune, Gemma Terms of Use; for llama-server)
  HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive at revision 45b6a334b4bcd1d7f37179df58b3b1d66a184e5d
  (base model google/gemma-4-e4b-it; the licence tag on the repo is "gemma"). The Q6_K_P quant,
  about Q8 quality in 5.8 GB, chosen so an 11 GB machine keeps room for the context window and
  the vision projector. llama-server needs both files, the second one for images (--mmproj):
    Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf         6249794528 bytes, sha256 pinned
    mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf      990288832 bytes, sha256 pinned
  scripts/start-remote-lm-serve.sh --id gemma-uncensored starts them. The engine cannot read this
  GGUF (two .gguf files, no Gemma 4 GGUF loader), so lm-serve is not used.

Ten more GGUFs (src/chat/lm_catalog.c). Each --model name is the directory under the models root.
A projector is downloaded only when that row has a matching mmproj. Sizes and sha256 are pinned.

  gemma-4-e2b                 unsloth/gemma-4-E2B-it-GGUF @ 0314792d7f1f7e229411f620751375812bb9faf2
                              gemma-4-E2B-it-Q6_K.gguf + mmproj-F16.gguf (gemma4, llama-server v1.0.0)
  nemotron-3.5-bolt-e4b       LlamaPal/NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL-GGUF @ a66cd4c2f72de88bb83120bc6ac11ca5608f6a9b
                              NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf (gemma4, no projector in that repo)
  granite-4.2-3b              ibm-granite/granite-4.2-3b-GGUF @ c40945d71cd90f249a56985e8155551a9188dc30
                              granite-4.2-3b-Q8_0.gguf (granite)
  spark-x2.6-4b               bartowski/Spark-X2.6-4B-GGUF @ 17682e38297a4510899b12a95ab3d8dcebc674e5
                              Spark-X2.6-4B-Q4_K_M.gguf (spark2_5, llama.cpp b10828 or later)
  lfm3-vl-2b                  LiquidAI/LFM3-VL-2B-GGUF @ 1add221a5ea6c047fcf1eaaf2d247939d7cf8849
                              LFM3-VL-2B-Q8_0.gguf + mmproj-LFM3-VL-2B-F16.gguf (lfm3, llama.cpp b11938 or later)
  gemma-4-e2b-heretic         llmfan46/gemma-4-E2B-it-ultra-uncensored-heretic-GGUF @ e16997210b96c34c4dbd61a452765819db5f0c0e
                              Q6_K weights + the repo's mmproj
  spark-x2.6-4b-heretic       mradermacher/Spark-X2.6-4B-heretic-GGUF @ 52b0599835ab71058515faba28c228c60b7559af
                              Spark-X2.6-4B-heretic.Q6_K.gguf (spark2_5, b10828, text only)
  granite-4.2-3b-heretic      mradermacher/granite-4.2-3b-Heretic-GGUF @ b113a4c18ed0063886211387945bc0254e7dd8af
                              granite-4.2-3b-Heretic.Q8_0.gguf (granite, text only)
  qwen3.5-2b-abliterated      mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF @ 7f5aee7d2dc4df18bd67eab85ea81e0bc2e98a2c
                              the 2B Q8_0 plus its f16 mmproj, not the installed Qwen3.5-4B
  lfm3-vl-2b-abliterated      mradermacher/LFM3-VL-2B-Surgical-Abliterated-GGUF @ 98588af5cc8487c191e896b1970c099221a15eb9
                              Q8_0 plus its f16 mmproj (lfm3, b11938)

Behaviour:
  - Requires curl and sha256sum, and free space of the checkpoint size plus 1 GB.
  - Refuses a rotational disk unless --allow-hdd is given. If the disk type cannot be
    determined, a warning is printed and the script continues.
  - Downloads are resumable (curl -C -): re-run the script after an interruption.
  - A pinned file with the expected size and sha256 is skipped. A pinned file with the
    expected size but the wrong sha256 stops the script.
  - After a download the size and sha256 are checked. On mismatch the script exits
    non-zero and leaves the file in place for inspection.
  - Never deletes or overwrites a file, and never uses sudo.
EOF
}

die() { echo "error: $*" >&2; exit 1; }
usage_error() { echo "error: $*" >&2; echo "see: scripts/download-model.sh --help" >&2; exit 2; }
warn() { echo "warning: $*" >&2; }

MODEL="$DEFAULT_MODEL"
ALLOW_HDD=0
MODEL_DIR_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model)
      [ $# -ge 2 ] || usage_error "--model needs a value"
      MODEL="$2"; shift ;;
    --model=*) MODEL="${1#--model=}" ;;
    --allow-hdd) ALLOW_HDD=1 ;;
    -*) usage_error "unknown option: $1" ;;
    *)
      if [ -n "$MODEL_DIR_ARG" ]; then usage_error "more than one MODEL_DIR given"; fi
      MODEL_DIR_ARG="$1"
      ;;
  esac
  shift
done

# Model table. Each entry is "name|expected_size|expected_sha256"; an empty size or
# sha means the file is not pinned. Every file lives at the root of its revision.
case "$MODEL" in
  qwen3.5-4b)
    BASE_URL="https://huggingface.co/Qwen/Qwen3.5-4B/resolve/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    CHECKPOINT_BYTES=9319828096   # both weight shards
    FILES=(
      "config.json||"
      "model.safetensors.index.json||"
      "model.safetensors-00001-of-00002.safetensors|5329398688|26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"
      "model.safetensors-00002-of-00002.safetensors|3990429408|cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
      "tokenizer.json|12807982|5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
      "preprocessor_config.json|390|27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
      "tokenizer_config.json||"
      "chat_template.jinja||"
    )
    ;;
  gemma-4-e4b)
    BASE_URL="https://huggingface.co/google/gemma-4-E4B-it/resolve/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
    CHECKPOINT_BYTES=15992595884
    FILES=(
      "config.json||"
      "model.safetensors|15992595884|cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503"
      "tokenizer.json|32156428|91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc"
      "processor_config.json|1689|32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c"
      "tokenizer_config.json||"
      "chat_template.jinja||"
      "generation_config.json||"
    )
    ;;
  whiterabbitneo-2.5-7b-q4km)
    BASE_URL="https://huggingface.co/bartowski/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-GGUF/resolve/4874da7ad5fa4da4f2584ae64ddc21f5a8e7dfce"
    CHECKPOINT_BYTES=4683072032
    FILES=(
      "WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-Q4_K_M.gguf|4683072032|3790b0bf2c505fcbd144b6b69354fe45a83ac09238a87469db0082027c127de4"
    )
    ;;
  gemma-4-e4b-uncensored)
    BASE_URL="https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/resolve/45b6a334b4bcd1d7f37179df58b3b1d66a184e5d"
    CHECKPOINT_BYTES=7240082912   # both files
    FILES=(
      "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf|6249794528|f28b0ae262158af10d847c3f01bfdb943161dd31320f3aed20775ee2ad6c67a6"
      "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf|990288832|debad39ab9c1152ab67695a674fb35e8375b2320c57bfd5075835d3ccb16c7db"
    )
    ;;
  gemma-4-e2b)
    BASE_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/0314792d7f1f7e229411f620751375812bb9faf2"
    CHECKPOINT_BYTES=5487375264
    FILES=(
      "gemma-4-E2B-it-Q6_K.gguf|4501721184|f7b8980ecc1c115898fd8d9a3d421d90becbac984194a44b478669c26815843d"
      "mmproj-F16.gguf|985654080|140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa"
    )
    ;;
  nemotron-3.5-bolt-e4b)
    BASE_URL="https://huggingface.co/LlamaPal/NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL-GGUF/resolve/a66cd4c2f72de88bb83120bc6ac11ca5608f6a9b"
    CHECKPOINT_BYTES=5060482240
    FILES=(
      "NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf|5060482240|84ff34f5834b929043f5ae74799e8d562d4056998f6a5923ceb44831e8ad84d6"
    )
    ;;
  granite-4.2-3b)
    BASE_URL="https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF/resolve/c40945d71cd90f249a56985e8155551a9188dc30"
    CHECKPOINT_BYTES=3892651552
    FILES=(
      "granite-4.2-3b-Q8_0.gguf|3892651552|fbe986738041418e26de9e123ba740cb654931f85bf572a71bd01f9e6b85e53d"
    )
    ;;
  spark-x2.6-4b)
    BASE_URL="https://huggingface.co/bartowski/Spark-X2.6-4B-GGUF/resolve/17682e38297a4510899b12a95ab3d8dcebc674e5"
    CHECKPOINT_BYTES=2621104768
    FILES=(
      "Spark-X2.6-4B-Q4_K_M.gguf|2621104768|30b77e2ede85c37da7071048c91fcf47c3e8cbfad51971640530f8ba9753301a"
    )
    ;;
  lfm3-vl-2b)
    BASE_URL="https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/1add221a5ea6c047fcf1eaaf2d247939d7cf8849"
    CHECKPOINT_BYTES=2225234528
    FILES=(
      "LFM3-VL-2B-Q8_0.gguf|1846238496|e138e54189a751d90ea8c0b236360490613ecd465d28d68fa6094d3db7fa6727"
      "mmproj-LFM3-VL-2B-F16.gguf|378996032|1d8e6e58eefc093e09d6332f081516e1ddbd5b35cfca7f35463fb270db230b1e"
    )
    ;;
  gemma-4-e2b-heretic)
    BASE_URL="https://huggingface.co/llmfan46/gemma-4-E2B-it-ultra-uncensored-heretic-GGUF/resolve/e16997210b96c34c4dbd61a452765819db5f0c0e"
    CHECKPOINT_BYTES=4832178912
    FILES=(
      "gemma-4-E2B-it-ultra-uncensored-heretic-Q6_K.gguf|3845345504|cb4fe5236c5750ea700f2d51f5635d3ca1e0c303a11241e5c9a7eb72690190a1"
      "gemma-4-E2B-it-mmproj-BF16.gguf|986833408|d460bb51ce5115232a723a7366694c8ee70d8aa8f60159d182a1da0777d10db3"
    )
    ;;
  spark-x2.6-4b-heretic)
    BASE_URL="https://huggingface.co/mradermacher/Spark-X2.6-4B-heretic-GGUF/resolve/52b0599835ab71058515faba28c228c60b7559af"
    CHECKPOINT_BYTES=3379174368
    FILES=(
      "Spark-X2.6-4B-heretic.Q6_K.gguf|3379174368|468cbfea25ce55a5ed7e87de121695df0497aa80766d98a89bf0d95b86efa6fb"
    )
    ;;
  granite-4.2-3b-heretic)
    BASE_URL="https://huggingface.co/mradermacher/granite-4.2-3b-Heretic-GGUF/resolve/b113a4c18ed0063886211387945bc0254e7dd8af"
    CHECKPOINT_BYTES=3892652640
    FILES=(
      "granite-4.2-3b-Heretic.Q8_0.gguf|3892652640|8e3fbd9a66ebf76e900b5647de12ec3d4c86943bed85268322e01b84e7337491"
    )
    ;;
  qwen3.5-2b-abliterated)
    BASE_URL="https://huggingface.co/mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF/resolve/7f5aee7d2dc4df18bd67eab85ea81e0bc2e98a2c"
    CHECKPOINT_BYTES=2680240192
    FILES=(
      "Huihui-Qwen3.5-2B-abliterated.Q8_0.gguf|2012012992|fb724dfeb9c268e4752b944d35213e959b525cab39db021cc2bdf439901d4deb"
      "Huihui-Qwen3.5-2B-abliterated.mmproj-f16.gguf|668227200|ca72dabcfb10c4954d533dd73e1365f790df0b9fe4941d9030b6a0d2c46abb7c"
    )
    ;;
  lfm3-vl-2b-abliterated)
    BASE_URL="https://huggingface.co/mradermacher/LFM3-VL-2B-Surgical-Abliterated-GGUF/resolve/98588af5cc8487c191e896b1970c099221a15eb9"
    CHECKPOINT_BYTES=2225234816
    FILES=(
      "LFM3-VL-2B-Surgical-Abliterated.Q8_0.gguf|1846238912|0e79bfd4fe30186741050ccff8e702cb7de2bec33a1cef72342813303fbaea67"
      "LFM3-VL-2B-Surgical-Abliterated.mmproj-f16.gguf|378995904|1a0e25010fa894ee476e34dd0359e709f01dedd981dbea5e488e40f9c7d2f098"
    )
    ;;
  *)
    usage_error "unknown model: $MODEL (see --help)"
    ;;
esac

MODEL_DIR="${MODEL_DIR_ARG:-$("$(dirname "${BASH_SOURCE[0]}")/models-root.sh")/$MODEL}"
MIN_FREE_BYTES=$((CHECKPOINT_BYTES + ONE_GB))

for cmd in curl sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || die "required command not found: $cmd"
done

mkdir -p "$MODEL_DIR" || die "cannot create $MODEL_DIR"

file_size() { wc -c < "$1" | tr -d '[:space:]'; }
sha256_of() { sha256sum "$1" | awk '{print $1}'; }

# Prints 0 (non-rotational), 1 (rotational) or unknown for the filesystem holding $1.
# Sets rot_src to the block device it resolved.
rot_src=""
rotational_status() {
  local p="$1" src="" flags name f
  rot_src=""
  if command -v findmnt >/dev/null 2>&1; then
    src=$(findmnt -n -o SOURCE --target "$p" 2>/dev/null | awk 'NR==1 {print; exit}') || src=""
  fi
  if [ -z "$src" ]; then
    src=$(df -P "$p" 2>/dev/null | awk 'NR==2 {print $1}') || src=""
  fi
  src=${src%%\[*}          # strip btrfs subvolume suffix, e.g. /dev/sda2[/@]
  rot_src="$src"
  case "$src" in
    /dev/*) ;;
    *) echo unknown; return 0 ;;
  esac
  if command -v lsblk >/dev/null 2>&1; then
    # -s lists the device and its parents (LVM, LUKS, RAID); any rotational parent counts.
    flags=$(lsblk -sn -o ROTA "$src" 2>/dev/null | tr -d ' ' | awk 'NF') || flags=""
    if [ -n "$flags" ]; then
      if grep -qx 1 <<< "$flags"; then echo 1
      elif grep -qvx 0 <<< "$flags"; then echo unknown
      else echo 0
      fi
      return 0
    fi
  fi
  name=${src#/dev/}
  for f in "/sys/class/block/$name/queue/rotational" "/sys/class/block/$name/../queue/rotational"; do
    if [ -r "$f" ]; then
      flags=$(tr -d ' \n' < "$f")
      case "$flags" in
        0) echo 0 ;;
        1) echo 1 ;;
        *) echo unknown ;;
      esac
      return 0
    fi
  done
  echo unknown
}

# Free space on the filesystem that will hold the files.
avail_kb=$(df -Pk "$MODEL_DIR" | awk 'NR==2 {print $4}')
case "$avail_kb" in
  ''|*[!0-9]*) die "cannot determine free space on $MODEL_DIR" ;;
esac
avail_bytes=$((avail_kb * 1024))
if [ "$avail_bytes" -lt "$MIN_FREE_BYTES" ]; then
  die "need at least $((MIN_FREE_BYTES / ONE_GB)) GB free on the filesystem holding $MODEL_DIR; about $((avail_bytes / ONE_GB)) GB free"
fi
echo "free space: about $((avail_bytes / ONE_GB)) GB on the filesystem holding $MODEL_DIR"

rot=$(rotational_status "$MODEL_DIR")
case "$rot" in
  1)
    if [ "$ALLOW_HDD" -eq 1 ]; then
      warn "$MODEL_DIR is on a rotational disk ($rot_src); continuing because --allow-hdd was given"
    else
      die "$MODEL_DIR is on a rotational disk ($rot_src). Weight reads from an HDD are far too slow. Use a directory on the SSD, or pass --allow-hdd to override."
    fi
    ;;
  0) echo "storage: non-rotational ($rot_src)" ;;
  *) warn "could not determine whether $MODEL_DIR is on a rotational disk (device: ${rot_src:-unknown}); continuing" ;;
esac

# fetch NAME EXPECTED_SIZE EXPECTED_SHA256   (size and sha may be empty = not pinned)
fetch() {
  local name="$1" size="$2" sha="$3"
  local dest="$MODEL_DIR/$name" have got h

  if [ -f "$dest" ]; then
    if [ -z "$size" ]; then
      if [ -s "$dest" ]; then
        echo "skip $name: present (no pinned size or hash; delete it to force a re-fetch)"
        return 0
      fi
    else
      have=$(file_size "$dest")
      if [ "$have" = "$size" ]; then
        h=$(sha256_of "$dest")
        if [ "$h" = "$sha" ]; then
          echo "skip $name: already present and verified"
          return 0
        fi
        die "$dest has the expected size but the wrong sha256 ($h). The script does not delete files; move it aside and re-run."
      fi
      if [ "$have" -gt "$size" ]; then
        die "$dest is $have bytes, larger than the expected $size. The script does not delete files; move it aside and re-run."
      fi
    fi
  fi

  echo "download $name (${size:-size not pinned} bytes)"
  curl -fL -C - --retry 3 --retry-delay 5 -o "$dest" "$BASE_URL/$name" \
    || die "download failed for $name; re-run the script to resume"

  got=$(file_size "$dest")
  if [ -z "$size" ]; then
    [ "$got" -gt 0 ] || die "$name is empty after download"
  else
    if [ "$got" != "$size" ]; then
      die "$name: size is $got bytes, expected $size (left in place for inspection)"
    fi
    h=$(sha256_of "$dest")
    if [ "$h" != "$sha" ]; then
      die "$name: sha256 is $h, expected $sha (left in place for inspection)"
    fi
  fi
  echo "ok $name"
}

for entry in "${FILES[@]}"; do
  IFS='|' read -r name size sha <<< "$entry"
  fetch "$name" "$size" "$sha"
done

echo "done: $MODEL files are in $MODEL_DIR"
