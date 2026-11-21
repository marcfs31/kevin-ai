#!/usr/bin/env bash
# test_lm_serve.sh - bin/lm-serve end to end on the tiny Qwen3.5 fixture over real HTTP (curl):
# the endpoints, streaming, state reuse when the client sends the previous answer back, the
# API key, and the refusals (bad JSON, unsupported parameters, wrong method, unknown path,
# oversized body), and lm-chat's default reply length on the same model. Run from the
# repository root after `make`.
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

# raw PAYLOAD: send PAYLOAD (a Python bytes expression, so \r\n and repetition are easy) on a
# plain socket and print the status code of the reply, or "reset"/"none"
raw() {
    python3 -I - "${url#http://}" "$1" <<'PY'
import socket, sys
host, port = sys.argv[1].rsplit(":", 1)
s = socket.create_connection((host, int(port)), timeout=10)
try:
    s.sendall(eval(sys.argv[2]))
    data = s.recv(4096)
    print(data.split(b" ")[1].decode() if data.startswith(b"HTTP/1.1 ") else "none")
except ConnectionResetError:
    print("reset")
PY
}

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
# chat JSON: POST to chat completions, print the status, and wait for the request's log line
# (the server logs after the reply is sent, so a check of the log could otherwise race it)
chat() {
    local n
    n=$(wc -l < "$tmp/serve.err")
    post -d "$1" "$url/v1/chat/completions"
    for _ in $(seq 50); do [ "$(wc -l < "$tmp/serve.err")" -gt "$n" ] && break; sleep 0.1; done
}
# expect STATUS JSON WHAT: POST JSON to chat completions and check the status
expect() {
    local got
    got=$(chat "$2")
    checks=$((checks + 1))
    if [ "$got" != "$1" ]; then fails=$((fails + 1)); echo "  FAIL $3: status $got, $(head -c 300 "$tmp/body")"; fi
}

start
check '[ "$(post "$url/health")" = 200 ] && [ "$(j "d[\"status\"]")" = ok ] && [ "$(j "d[\"model\"]")" = tiny ]' \
      "/health: $(cat "$tmp/body")"
check '[ "$(j "d[\"capabilities\"][\"vision\"][\"status\"] in (\"ok\",\"unavailable\")")" = True ]' \
      "/health vision status: $(cat "$tmp/body")"
check '[ "$(post "$url/v1/models")" = 200 ] && [ "$(j "d[\"data\"][0][\"id\"]")" = tiny ]' "/v1/models: $(cat "$tmp/body")"
check '[ "$(j "\"context_length\" in d[\"data\"][0] and d[\"data\"][0][\"context_length\"] > 0")" = True ]' \
      "/v1/models context_length: $(cat "$tmp/body")"
check '[ "$(j "d[\"data\"][0][\"capabilities\"][\"vision\"][\"status\"] in (\"ok\",\"unavailable\") and \"max_pixels\" in d[\"data\"][0][\"capabilities\"][\"vision\"]")" = True ]' \
      "/v1/models vision capabilities: $(cat "$tmp/body")"

# a plain completion: an answer, finish_reason length at max_tokens, usage counts
req='{"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0}'
check '[ "$(chat "$req")" = 200 ]' "chat completion status: $(cat "$tmp/body")"
check '[ "$(j "d[\"object\"] == \"chat.completion\" and d[\"choices\"][0][\"finish_reason\"] == \"length\" and d[\"usage\"][\"completion_tokens\"] == 5")" = True ]' \
      "completion shape: $(cat "$tmp/body")"
check '[ "$(j "d[\"usage\"][\"prompt_tokens_details\"][\"cached_tokens\"] == 0")" = True ]' \
      "usage cached_tokens: $(cat "$tmp/body")"
answer=$(j 'json.dumps(d["choices"][0]["message"]["content"])')
p1=$(j 'd["usage"]["prompt_tokens"]')

# bad image_url (http, not a data URL): 400 with image_unsupported or image_invalid
expect 400 '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}],"max_tokens":1}' \
      "remote image_url"
check 'grep -Eq "image_unsupported|image_invalid" "$tmp/body"' "remote image_url error code: $(cat "$tmp/body")"

# temperature 0 is reproducible
chat "$req" > /dev/null
check '[ "$(j "json.dumps(d[\"choices\"][0][\"message\"][\"content\"])")" = "$answer" ]' "temperature 0 gave two different answers"

# the client sends the answer back with a new message: only the new part is fed
req2='{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":'"$answer"'},{"role":"user","content":"more"}],"max_tokens":3,"temperature":0}'
check '[ "$(chat "$req2")" = 200 ]' "follow-up status: $(cat "$tmp/body")"
check 'tail -1 "$tmp/serve.err" | grep -q "prompt [0-9]*, 0 from cache, fed [0-9]* at" && ! tail -1 "$tmp/serve.err" | grep -q "re-fed"' \
      "the follow-up re-fed the conversation: $(tail -1 "$tmp/serve.err")"
check '[ "$(j "d[\"usage\"][\"prompt_tokens\"]")" -gt "$p1" ]' "the follow-up prompt is not longer than the first"
# reset drops the conversation and the prefix cache, so the same follow-up is fed again
check '[ "$(code "$url/v1/chat/reset")" = 405 ]' "GET /v1/chat/reset was accepted"
check '[ "$(post -d "{}" "$url/v1/chat/reset")" = 200 ] && grep -q "\"status\":\"ok\"" "$tmp/body"' "POST /v1/chat/reset: $(cat "$tmp/body")"
check '[ "$(chat "$req2")" = 200 ]' "chat after reset: $(cat "$tmp/body")"
# nothing was left to extend: the whole prompt is fed, and none of it comes from the cache
check 'tail -1 "$tmp/serve.err" | grep -Eq "prompt ([0-9]+), 0 from cache, fed \\1 at"' \
      "reset left the conversation in memory: $(tail -1 "$tmp/serve.err")"
# NEGATIVE: an edited answer cannot reuse the state
chat '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"edited"},{"role":"user","content":"more"}],"max_tokens":3}' > /dev/null
check 'tail -1 "$tmp/serve.err" | grep -q "(re-fed)"' "an edited history was not re-fed: $(tail -1 "$tmp/serve.err")"

# prefix cache: conversation C, then D with the same system prompt (snapshotted at the shared
# prefix), then C again restored from it -- the same answer as without the cache
creq='{"messages":[{"role":"system","content":"You answer in one short line."},{"role":"user","content":"gamma"}],"max_tokens":6,"temperature":0}'
dreq='{"messages":[{"role":"system","content":"You answer in one short line."},{"role":"user","content":"delta"}],"max_tokens":6,"temperature":0}'
check '[ "$(chat "$creq")" = 200 ] && tail -1 "$tmp/serve.err" | grep -q " 0 from cache"' "conversation C: $(tail -1 "$tmp/serve.err")"
cans=$(j 'json.dumps(d["choices"][0]["message"]["content"])')
check '[ "$(chat "$dreq")" = 200 ] && tail -1 "$tmp/serve.err" | grep -q " 0 from cache"' "conversation D: $(tail -1 "$tmp/serve.err")"
check '[ "$(chat "$creq")" = 200 ] && tail -1 "$tmp/serve.err" | grep -q " [1-9][0-9]* from cache, fed [0-9]* at"' \
      "conversation C again was not restored from the cache: $(tail -1 "$tmp/serve.err")"
check '[ "$(j "json.dumps(d[\"choices\"][0][\"message\"][\"content\"])")" = "$cans" ]' "the cached conversation answered differently"

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
head -c $((24 * 1024 * 1024 + 100)) /dev/zero | tr '\0' a > "$tmp/big"
check '[ "$(post --data-binary @"$tmp/big" "$url/v1/chat/completions")" = 413 ]' "a body over 24 MiB accepted"
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
check '[ "$(post -d "{}" "$url/v1/chat/reset")" = 401 ]' "reset without a key accepted"
check '[ "$(code -H "Authorization: Bearer secret-test-kez" "$url/v1/models")" = 401 ]' "a wrong key accepted"
check '[ "$(code -H "Authorization: Bearer secret-test-key" "$url/v1/models")" = 200 ]' "the right key refused"
check '[ "$(code "$url/health")" = 200 ]' "/health needs a key"
check '! grep -q secret-test-key "$tmp/serve.err"' "the key appears in the log"
check '[ "$(post -H "Authorization: Bearer nope" -d "{}" "$url/v1/chat/completions")" = 401 ]' "a wrong key accepted on chat completions"
check '[ "$(post -H "Authorization: Basic c2VjcmV0LXRlc3Qta2V5" -d "{}" "$url/v1/chat/completions")" = 401 ]' "a non-Bearer scheme accepted"
one='{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
check '[ "$(post -H "Authorization: Bearer secret-test-key" -d "$one" "$url/v1/chat/completions")" = 200 ]' "the right key refused on chat completions"
stop

# protocol refusals on a raw socket: ambiguous framing and malformed headers are a 400, not a guess
start --timeout 2
# (the body is a valid request, so a server that picked one of the headers would answer 200)
ok_body='b"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}"'
check '[ "$(raw "b\"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 60\r\nContent-Length: 3\r\n\r\n\" + $ok_body")" = 400 ]' "a repeated Content-Length was accepted"
check '[ "$(raw "b\"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n\"")" = 411 ]' "chunked transfer encoding was accepted"
check '[ "$(raw "b\"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding : chunked\r\nContent-Length: 60\r\n\r\n\" + $ok_body")" = 400 ]' "whitespace before a header colon was accepted"
check '[ "$(raw "b\"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 60\r\n\r\n\" + $ok_body")" = 200 ]' "the same request with clean headers was refused"
check '[ "$(raw "b\"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: -1\r\n\r\n\"")" = 400 ]' "a negative Content-Length was accepted"
check '[ "$(raw "b\"GET /health HTTP/1.1\r\nX: \" + b\"a\" * 20000 + b\"\r\n\r\n\"")" = 431 ]' "a 20 KB header was accepted"
# a client that sends the body without waiting still gets the 413, not a connection reset
check '[ "$(raw "b\"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 25165924\r\n\r\n\" + b\"a\" * 1024")" = 413 ]' "an oversized body sent at once did not get a 413"
# a request that does not arrive in time is a 408; a slow client does not hold the server
check '[ "$(raw "b\"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 10\r\n\r\n{}\"")" = 408 ]' "an incomplete body did not time out with 408"
python3 -I - "${url#http://}" <<'PY' &
import socket, sys, time
host, port = sys.argv[1].rsplit(":", 1)
s = socket.create_connection((host, int(port)))
for ch in b"GET /health HTTP/1.1\r\nX: " + b"a" * 100:
    try:
        s.send(bytes([ch]))
    except OSError:
        break
    time.sleep(0.25)
PY
trickler=$!
sleep 0.5
t0=$(date +%s)
check '[ "$(code -m 8 "$url/health")" = 200 ] && [ $(( $(date +%s) - t0 )) -le 5 ]' "a trickling client held the server for $(( $(date +%s) - t0 )) s"
kill "$trickler" 2> /dev/null; wait "$trickler" 2> /dev/null
# control bytes in the path do not reach the log
raw "b\"GET /x\x1b[31m HTTP/1.1\r\n\r\n\"" > /dev/null
check '! grep -q "$(printf "\033")" "$tmp/serve.err" && grep -q "GET /x?\[31m 404" "$tmp/serve.err"' "a control byte from the path reached the log"
stop

# a client that goes away stops its reply; a reply cut by shutdown is a 503, not a "stop"
start
curl -s -m 0.3 -o /dev/null -d '{"messages":[{"role":"user","content":"hi"}],"temperature":0}' "$url/v1/chat/completions"
for _ in $(seq 50); do grep -q "499" "$tmp/serve.err" && break; sleep 0.1; done
check 'grep -q "POST /v1/chat/completions 499 (client gone)" "$tmp/serve.err"' "an abandoned reply was not stopped: $(tail -1 "$tmp/serve.err")"
check '[ "$(sed -n "s/.*499 (client gone).*generated \([0-9]*\) at.*/\1/p" "$tmp/serve.err")" -lt 4000 ]' "the abandoned reply ran to the end of the context"
chat '{"messages":[{"role":"user","content":"hi"}],"temperature":0}' > "$tmp/code" &
curl_pid=$!
sleep 0.3
kill "$pid"
wait "$curl_pid"
check '[ "$(cat "$tmp/code")" = 503 ] && grep -q "shutting down" "$tmp/body"' "a reply cut by SIGTERM: status $(cat "$tmp/code"), $(head -c 300 "$tmp/body")"
wait "$pid" 2> /dev/null; pid=

# exposed beyond loopback without a key: a warning; with a key: none
start --host 0.0.0.0
check 'grep -q "warning: reachable from other machines" "$tmp/serve.err"' "no warning for 0.0.0.0 without a key"
stop
LM_SERVE_API_KEY=k start --host 0.0.0.0
check '! grep -q "warning" "$tmp/serve.err"' "a warning although a key is set"
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

# lm-chat on the same model: without --max-tokens an answer runs to the end of the context
# (the old default stopped at 512 tokens, inside this 1024-token window)
bin/lm-chat --model "$tmp/model" --weights "$tmp/model/model.q8.safetensors" --ctx 1024 --temperature 0 \
    --prompt hi > /dev/null 2> "$tmp/chat.err"; rc=$?
check '[ $rc -eq 0 ] && grep -q "context full" "$tmp/chat.err" && ! grep -q "max-tokens reached" "$tmp/chat.err"' \
      "lm-chat without --max-tokens (exit $rc): $(cat "$tmp/chat.err")"
# NEGATIVE: an explicit --max-tokens still stops there
bin/lm-chat --model "$tmp/model" --weights "$tmp/model/model.q8.safetensors" --ctx 1024 --temperature 0 \
    --prompt hi --max-tokens 5 > /dev/null 2> "$tmp/chat.err"
check 'grep -q "max-tokens reached" "$tmp/chat.err"' "lm-chat --max-tokens 5 did not stop: $(cat "$tmp/chat.err")"

# /reset drops the prefix cache. The configured system prompt is put back, and each
# turn is fed from scratch (three "0 from cache" lines, no restore).
printf 'gamma\n/reset\ndelta\n/reset\ngamma\n' | bin/lm-chat --model "$tmp/model" --weights "$tmp/model/model.q8.safetensors" \
    --ctx 256 --temperature 0 --max-tokens 4 --stats --system "You answer in one short line." > /dev/null 2> "$tmp/chat.err"
check '[ "$(grep -c " 0 from cache" "$tmp/chat.err")" = 3 ] && ! grep -q " [1-9][0-9]* from cache" "$tmp/chat.err"' \
      "lm-chat /reset left a prefix snapshot: $(grep "from cache" "$tmp/chat.err")"
check '[ "$(grep -c "^\[context 256 tokens\]$" "$tmp/chat.err")" = 1 ]' "lm-chat --stats did not print the context window once"

# a file log is cut back to LM_SERVE_LOG_MAX before the next line is written
LM_SERVE_LOG_MAX=80 start
check 'grep -q listening "$tmp/serve.err"' "startup was not logged"
check '[ "$(code "$url/health")" = 200 ]' "health under a small log cap"
check '! grep -q listening "$tmp/serve.err" && grep -q "GET /health" "$tmp/serve.err"' \
      "the log was not truncated: $(cat "$tmp/serve.err")"
stop
LM_SERVE_LOG_MAX=0 "$SERVE" --model "$tmp/model" --weights "$tmp/model/model.q8.safetensors" --port 0 \
    >/dev/null 2>"$tmp/logmax.err"; rc=$?
check '[ $rc -eq 2 ] && grep -q LM_SERVE_LOG_MAX "$tmp/logmax.err"' "a bad log cap exited $rc: $(cat "$tmp/logmax.err")"

# --system is wiped from argv once the process is running
secret='ZZ-prompt-not-in-process-list-ZZ'
mkfifo "$tmp/in"
sleep 60 >"$tmp/in" &
holder=$!
bin/lm-chat --model "$tmp/model" --weights "$tmp/model/model.q8.safetensors" --system "$secret" \
    <"$tmp/in" >/dev/null 2>"$tmp/ps.err" &
chatpid=$!
for _ in $(seq 50); do kill -0 "$chatpid" 2>/dev/null && break; sleep 0.05; done
sleep 0.3
check 'kill -0 "$chatpid" 2>/dev/null' "lm-chat exited before the argv check: $(cat "$tmp/ps.err")"
check '! ps -ww -p "$chatpid" -o args= | grep -q "$secret"' "system text stayed on argv: $(ps -ww -p "$chatpid" -o args=)"
printf '/exit\n' >"$tmp/in"
wait "$chatpid" 2>/dev/null || true
kill "$holder" 2>/dev/null || true
wait "$holder" 2>/dev/null || true

if [ "$fails" -ne 0 ]; then echo "test_lm_serve: $fails of $checks checks FAILED"; exit 1; fi
echo "test_lm_serve: $checks checks passed"
