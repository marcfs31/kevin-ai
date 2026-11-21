#!/usr/bin/env bash
# install-tool-runtimes.sh against fake archives served from file:// URLs: verification, unpacking,
# the links, running twice, and refusing a download whose hash is wrong.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
script=$root/scripts/install-tool-runtimes.sh
tmp=$(mktemp -d)
trap 'find "$tmp" -delete' EXIT
fails=0; checks=0
check() { checks=$((checks + 1)); if ! eval "$1"; then fails=$((fails + 1)); echo "  FAIL $2"; fi; }

# Fake archives shaped like the real ones: node-*/bin/{node,npm,npx} and uv-*/{uv,uvx}.
mkdir -p "$tmp/src/node-fake/bin" "$tmp/src/uv-fake"
for b in node npm npx; do printf '#!/bin/sh\necho fake %s\n' "$b" > "$tmp/src/node-fake/bin/$b"; chmod +x "$tmp/src/node-fake/bin/$b"; done
for b in uv uvx; do printf '#!/bin/sh\necho fake %s\n' "$b" > "$tmp/src/uv-fake/$b"; chmod +x "$tmp/src/uv-fake/$b"; done
tar -cJf "$tmp/node.tar.xz" -C "$tmp/src" node-fake
tar -czf "$tmp/uv.tar.gz" -C "$tmp/src" uv-fake
nsha=$(sha256sum "$tmp/node.tar.xz" | awk '{print $1}')
usha=$(sha256sum "$tmp/uv.tar.gz" | awk '{print $1}')
args=(--node-url "file://$tmp/node.tar.xz" --node-sha256 "$nsha" --uv-url "file://$tmp/uv.tar.gz" --uv-sha256 "$usha")

out=$(bash "$script" --dir "$tmp/opt" "${args[@]}" 2>&1); rc=$?
check '[ $rc -eq 0 ]' "install exited $rc: $out"
check '[[ "$out" == *"verified node.tar.xz"* && "$out" == *"verified uv.tar.gz"* ]]' "both downloads should be verified, got: $out"
for b in node npm npx uv uvx; do
  check '[ "$("$tmp/opt/tool-bin/$b")" = "fake $b" ]' "tool-bin/$b does not run"
done
check '[ -d "$tmp/opt/node-v24.22.1-linux-x64" ] && [ -d "$tmp/opt/uv-0.13.5" ]' "runtime folders have the pinned versions in their names"
check '[ -z "$(ls -A "$tmp/opt/dl")" ]' "the download folder should be empty afterwards"

# running again leaves what is installed alone and does not download
out=$(bash "$script" --dir "$tmp/opt" --node-url "file:///nonexistent" --uv-url "file:///nonexistent" --node-sha256 "$nsha" --uv-sha256 "$usha" 2>&1); rc=$?
check '[ $rc -eq 0 ] && [[ "$out" == *"already in"* ]]' "a second run should find both installed, got: $out"

# NEGATIVE: a download with the wrong hash installs nothing and says what it got
out=$(bash "$script" --dir "$tmp/opt2" --only node --node-url "file://$tmp/node.tar.xz" --node-sha256 0000 2>&1); rc=$?
check '[ $rc -ne 0 ] && [[ "$out" == *"expected 0000"* ]]' "a wrong hash should fail, got rc=$rc: $out"
check '[ ! -e "$tmp/opt2/node-v24.22.1-linux-x64" ] && [ ! -e "$tmp/opt2/tool-bin/node" ]' "a wrong hash must leave nothing installed"
# NEGATIVE: an archive without the binary is refused
mkdir -p "$tmp/src/empty/bin"; tar -cJf "$tmp/bad.tar.xz" -C "$tmp/src" empty
bsha=$(sha256sum "$tmp/bad.tar.xz" | awk '{print $1}')
out=$(bash "$script" --dir "$tmp/opt3" --only node --node-url "file://$tmp/bad.tar.xz" --node-sha256 "$bsha" 2>&1); rc=$?
check '[ $rc -ne 0 ] && [[ "$out" == *"no bin/node"* ]]' "an archive without node should be refused, got: $out"
# NEGATIVE: bad options
out=$(bash "$script" --only python 2>&1); check '[[ "$out" == *"--only must be node or uv"* ]]' "--only python should be refused"
out=$(bash "$script" --nope 2>&1); check '[[ "$out" == *"unknown option"* ]]' "an unknown option should be refused"
# the pinned values in --help are the ones in the script
check 'bash "$script" --help | grep -q "$(sed -n "s/^NODE_VERSION=//p" "$script")" && bash "$script" --help | grep -q "$(sed -n "s/^UV_VERSION=//p" "$script")"' "--help should name the pinned versions"

if [ "$fails" -ne 0 ]; then echo "test_install_tool_runtimes: $fails of $checks checks FAILED"; exit 1; fi
echo "test_install_tool_runtimes: $checks checks passed"
