#!/usr/bin/env bash
# setup-remote-searxng.sh: argument checks and the command it would send. No SSH connection is opened.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
script=$root/scripts/setup-remote-searxng.sh
fails=0; checks=0
check() { checks=$((checks + 1)); if ! eval "$1"; then fails=$((fails + 1)); echo "  FAIL $2"; fi; }
fakebin=$(mktemp -d)
trap 'find "$fakebin" -delete' EXIT
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$*"\n' > "$fakebin/ssh"; chmod +x "$fakebin/ssh"
run() { PATH="$fakebin:$PATH" bash "$script" "$@" 2>&1 || true; }

out=$(run)
check '[[ "$out" == *"toor@192.168.1.96"*"bash -s -- "*"opt 8888 c6ee3e9fd0235004a694e63720efc869bb1e3594"* ]]' "defaults (target, opt, port 8888, the pinned commit) missing, got: $out"
out=$(run --port 9100); check '[[ "$out" == *" 9100 c6ee3e9f"* ]]' "--port not passed on, got: $out"
out=$(LM_SEARXNG_SSH=me@host run); check '[[ "$out" == *"me@host"* ]]' "LM_SEARXNG_SSH ignored, got: $out"
for bad in "--port" "--port x" "--port 80" "--port 99999" "--nope"; do
  read -ra words <<< "$bad"; out=$(run "${words[@]}")
  check '[[ "$out" != *"BatchMode"* ]]' "$bad should be refused before ssh, got: $out"
done
# the instance must stay on loopback with the JSON format and no sudo
check 'grep -q "bind_address: \"127.0.0.1\"" "$script" && grep -q "    - json" "$script" && ! grep -q "sudo" "$script"' "loopback, json and no sudo"
# the secret key is generated, never written into the repository
check '! grep -qE "secret_key: \"[0-9a-f]{16,}\"" "$script"' "no fixed secret key"

if [ "$fails" -ne 0 ]; then echo "test_setup_remote_searxng: $fails of $checks checks FAILED"; exit 1; fi
echo "test_setup_remote_searxng: $checks checks passed"
