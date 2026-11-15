#!/usr/bin/env bash
# test_lm_serve.sh - bin/lm-serve end to end on the tiny Qwen3.5 fixture over real HTTP (curl):
# the endpoints, streaming, state reuse when the client sends the previous answer back, the
# API key, and the refusals (bad JSON, unsupported parameters, wrong method, unknown path,
# oversized body). Run from the repository root after `make`.
#
# The fixture tokenizer has ids up to 407 but the tiny model 384, so the test builds a model
# directory whose tokenizer moves the special tokens into ids 358..383 (dropping the plain
# tokens there and the merges that use them).
set -u
SERVE=bin/lm-serve FIX=tests/fixtures/qwen35-tiny TOK=tests/fixtures/tok/qwen-tokenizer.json
[ -x "$SERVE" ] || { echo "  FAIL $SERVE not found (run make from the repository root)"; exit 1; }
[ -f "$FIX/config.json" ] || { echo "  FAIL fixture $FIX not found (run from the repository root)"; exit 1; }
command -v curl > /dev/null || { echo "  FAIL curl is required"; exit 1; }
tmp=$(mktemp -d) || exit 1
pid=
cleanup() { [ -n "$pid" ] && kill "$pid" 2> /dev/null; wait 2> /dev/null; find "$tmp" -delete; }
trap cleanup EXIT
checks=0; fails=0
check() { checks=$((checks + 1)); if ! eval "$1"; then fails=$((fails + 1)); echo "  FAIL $2"; fi; }

mkdir "$tmp/model"
cp "$FIX/config.json" "$tmp/model/"
python3 -I - "$TOK" "$tmp/model/tokenizer.json" <<'EOF' || { echo "  FAIL cannot build the test tokenizer"; exit 1; }
import json, sys
t = json.load(open(sys.argv[1]))
added = t["added_tokens"]
base = 384 - len(added)
vocab = {k: v for k, v in t["model"]["vocab"].items() if v < base}
for i, a in enumerate(added):
    a["id"] = base + i
merges = []
for m in t["model"]["merges"]:
    a, b = m.split(" ") if isinstance(m, str) else m
    if a in vocab and b in vocab and a + b in vocab:
        merges.append(m)
t["model"]["vocab"], t["model"]["merges"] = vocab, merges
json.dump(t, open(sys.argv[2], "w"))
EOF
: > "$tmp/body"
ln -s "$PWD/$FIX/model.q8.safetensors" "$tmp/model/model.q8.safetensors"

# j EXPR: evaluate a Python expression over the last response body, parsed as d
j() { python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$tmp/body" "$1"; }
post() { curl -s -o "$tmp/body" -w '%{http_code}' -H 'Content-Type: application/json' "$@"; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

start() { # start the server with extra args; sets $pid and $url
    : > "$tmp/serve.err"
    "$SERVE" --model "$tmp/model" --weights "$tmp/model/model.q8.safetensors" --port 0 --name tiny "$@" 2> "$tmp/serve.err" &
    pid=$!
    url=
    for _ in $(seq 100); do
        url=$(sed -n 's/^lm-serve: listening on \(http:[^ ]*\).*/\1/p' "$tmp/serve.err")
        [ -n "$url" ] && break
        kill -0 "$pid" 2> /dev/null || break
        sleep 0.1
    done
    [ -n "$url" ] || { echo "  FAIL lm-serve did not start: $(cat "$tmp/serve.err")"; exit 1; }
}
stop() { kill "$pid" 2> /dev/null; wait "$pid" 2> /dev/null; rc_stop=$?; pid=; }
chat() { post -d "$1" "$url/v1/chat/completions"; }
# expect STATUS JSON WHAT: POST JSON to chat completions and check the status
expect() {
    local got
    got=$(chat "$2")
    checks=$((checks + 1))
    if [ "$got" != "$1" ]; then fails=$((fails + 1)); echo "  FAIL $3: status $got, $(head -c 300 "$tmp/body")"; fi
}

start
check '[ "$(curl -s "$url/health")" = "{\"status\":\"ok\",\"model\":\"tiny\"}" ]' "/health: $(curl -s "$url/health")"
check '[ "$(post "$url/v1/models")" = 200 ] && [ "$(j "d[\"data\"][0][\"id\"]")" = tiny ]' "/v1/models: $(cat "$tmp/body")"

# a plain completion: an answer, finish_reason length at max_tokens, usage counts
req='{"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0}'
check '[ "$(chat "$req")" = 200 ]' "chat completion status: $(cat "$tmp/body")"
check '[ "$(j "d[\"object\"] == \"chat.completion\" and d[\"choices\"][0][\"finish_reason\"] == \"length\" and d[\"usage\"][\"completion_tokens\"] == 5")" = True ]' \
      "completion shape: $(cat "$tmp/body")"
answer=$(j 'json.dumps(d["choices"][0]["message"]["content"])')
p1=$(j 'd["usage"]["prompt_tokens"]')

# temperature 0 is reproducible
chat "$req" > /dev/null
check '[ "$(j "json.dumps(d[\"choices\"][0][\"message\"][\"content\"])")" = "$answer" ]' "temperature 0 gave two different answers"

# the client sends the answer back with a new message: only the new part is fed
req2='{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":'"$answer"'},{"role":"user","content":"more"}],"max_tokens":3,"temperature":0}'
check '[ "$(chat "$req2")" = 200 ]' "follow-up status: $(cat "$tmp/body")"
check 'tail -1 "$tmp/serve.err" | grep -q "prompt [0-9]*, fed [0-9]* at" && ! tail -1 "$tmp/serve.err" | grep -q "re-fed"' \
      "the follow-up re-fed the conversation: $(tail -1 "$tmp/serve.err")"
check '[ "$(j "d[\"usage\"][\"prompt_tokens\"]")" -gt "$p1" ]' "the follow-up prompt is not longer than the first"
# NEGATIVE: an edited answer cannot reuse the state
chat '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"edited"},{"role":"user","content":"more"}],"max_tokens":3}' > /dev/null
check 'tail -1 "$tmp/serve.err" | grep -q "(re-fed)"' "an edited history was not re-fed: $(tail -1 "$tmp/serve.err")"

# streaming: a role chunk, content chunks, a finish chunk, the usage chunk, [DONE]
sreq='{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],"max_tokens":4,"stream":true,"stream_options":{"include_usage":true}}'
check '[ "$(chat "$sreq")" = 200 ] && [ "$(tail -2 "$tmp/body" | head -1)" = "data: [DONE]" ]' "stream does not end in [DONE]: $(tail -c 200 "$tmp/body")"
check 'grep -q "\"delta\":{\"role\":\"assistant\"" "$tmp/body" && grep -q "\"finish_reason\":\"length\"" "$tmp/body" && grep -q "\"completion_tokens\":4" "$tmp/body"' \
      "stream chunks: $(head -c 600 "$tmp/body")"
check 'grep "^data: {" "$tmp/body" | sed "s/^data: //" | python3 -I -c "import json,sys; [json.loads(l) for l in sys.stdin]"' "a stream chunk is not valid JSON"

# refusals, never silently ignored; neutral values of the same parameters are fine
expect 400 '{not json' "bad JSON"
expect 400 '{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}' "tools"
check 'grep -q "tools is not supported" "$tmp/body"' "the tools refusal does not name the parameter"
expect 400 '{"messages":[{"role":"user","content":"hi"}],"stop":["x"]}' "stop strings"
expect 200 '{"messages":[{"role":"user","content":"hi"}],"stop":null,"n":1,"tools":[],"tool_choice":"none","frequency_penalty":0,"max_tokens":1}' "neutral parameters"
expect 400 '{"messages":[{"role":"assistant","content":"hi"}]}' "a last message not from the user"
expect 400 '{"messages":[{"role":"user","content":"a"},{"role":"system","content":"late"},{"role":"user","content":"b"}]}' "a system message that is not first"
expect 400 '{"messages":[{"role":"tool","content":"hi"}]}' "a tool message"
expect 400 '{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}' "an image part"
expect 200 '{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":1}' "text parts"
expect 400 '{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}' "max_tokens 0"
expect 400 '{"messages":[{"role":"user","content":"hi"}],"temperature":-1}' "a negative temperature"
expect 400 '{"messages":[]}' "no messages"
check '[ "$(code "$url/v1/chat/completions")" = 405 ]' "GET on chat completions"
check '[ "$(code "$url/nope")" = 404 ]' "unknown path"
head -c 4194400 /dev/zero | tr '\0' a > "$tmp/big"
check '[ "$(post --data-binary @"$tmp/big" "$url/v1/chat/completions")" = 413 ]' "a body over 4 MiB accepted"
# the log carries counts, never the text
check '! grep -q "be brief\|edited" "$tmp/serve.err"' "message text in the log"
stop
check '[ "$rc_stop" -eq 0 ] && grep -q "^lm-serve: stopped" "$tmp/serve.err"' "SIGTERM did not stop it cleanly (exit $rc_stop)"

# without max_tokens a reply runs until the model stops or the context fills
start --ctx 64
expect 200 '{"messages":[{"role":"user","content":"hi"}],"temperature":0}' "an unbounded reply"
# 64 positions are fed; the last token is sampled from the 64th position's logits and not fed
check '[ "$(j "d[\"usage\"][\"total_tokens\"] == 65 and d[\"choices\"][0][\"finish_reason\"] == \"length\"")" = True ]' "unbounded reply usage: $(cat "$tmp/body")"
# NEGATIVE: a prompt longer than the context is a 400, not a cut reply
long=$(python3 -I -c 'print("hi " * 200)')
expect 400 '{"messages":[{"role":"user","content":"'"$long"'"}]}' "an over-long prompt"
check 'grep -q context_length_exceeded "$tmp/body"' "the over-long prompt error has no code: $(cat "$tmp/body")"
stop

# the API key: /v1 needs it, /health does not, a wrong key is refused, the log never shows it
LM_SERVE_API_KEY=secret-test-key start
check '[ "$(code "$url/v1/models")" = 401 ]' "no key accepted"
check '[ "$(code -H "Authorization: Bearer secret-test-kez" "$url/v1/models")" = 401 ]' "a wrong key accepted"
check '[ "$(code -H "Authorization: Bearer secret-test-key" "$url/v1/models")" = 200 ]' "the right key refused"
check '[ "$(code "$url/health")" = 200 ]' "/health needs a key"
check '! grep -q secret-test-key "$tmp/serve.err"' "the key appears in the log"
stop

# fail loud: a port in use or a missing model is an error exit, never a fallback
start
"$SERVE" --model "$tmp/model" --weights "$tmp/model/model.q8.safetensors" --port "${url##*:}" 2> "$tmp/busy.err"; rc=$?
check '[ $rc -ne 0 ] && grep -q "cannot listen" "$tmp/busy.err"' "a busy port did not fail (exit $rc): $(cat "$tmp/busy.err")"
stop
"$SERVE" --model "$tmp/nope" --port 0 2> "$tmp/missing.err"; rc=$?
check '[ $rc -eq 1 ]' "a missing model exited $rc"

# the systemd unit only uses flags lm-serve knows (a renamed flag would fail every start)
unit_flags=$(sed -n 's/^ExecStart=//p' scripts/lm-serve.service | tr ' ' '\n' | grep '^--')
for f in $unit_flags; do
    check '"$SERVE" --help | grep -q -- "$f"' "scripts/lm-serve.service uses $f, which lm-serve does not accept"
done
check '[ -n "$unit_flags" ]' "no flags found in scripts/lm-serve.service"

if [ "$fails" -ne 0 ]; then echo "test_lm_serve: $fails of $checks checks FAILED"; exit 1; fi
echo "test_lm_serve: $checks checks passed"
