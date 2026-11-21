#!/usr/bin/env bash
# Install the two runtimes the MCP servers of tools/chat_tools_catalog.json need, into a folder in
# the user's home (no sudo): Node.js (for the npx servers) and uv (for the uvx servers). Each
# download is checked against a pinned SHA-256 before anything is unpacked. Run it on the machine
# that runs tools/chat_tools.py. Safe to run again: a runtime already installed is left alone.
set -euo pipefail

NODE_VERSION=v24.22.1
NODE_URL="https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz"
NODE_SHA256=6c172a9ab6718f7b801fb5a7bd486af2ee5e32758086a04b84fcce2aa77095e6   # 32,026,916 bytes
UV_VERSION=0.13.5
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_SHA256=f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f     # 16,682,639 bytes

usage() {
  cat <<'EOF'
Usage: scripts/install-tool-runtimes.sh [--dir DIR] [--only node|uv] [--help]

  --dir DIR     where to install (default ~/opt). Creates DIR/node-<version>-linux-x64,
                DIR/uv-<version>, and DIR/tool-bin with links to node, npm, npx, uv and uvx.
  --only NAME   install just node or just uv.

Pinned downloads (linux x86_64), each verified with sha256sum before it is unpacked:
  Node.js v24.22.1  https://nodejs.org/dist/v24.22.1/node-v24.22.1-linux-x64.tar.xz  (32,026,916 bytes)
  uv 0.13.5         https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-unknown-linux-gnu.tar.gz  (16,682,639 bytes)

For tests the URLs and hashes can be replaced with --node-url, --node-sha256, --uv-url, --uv-sha256
(file:// URLs work). Needs curl, sha256sum and tar (with xz for Node).
EOF
}

DIR="$HOME/opt"
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dir) [ $# -ge 2 ] || { echo "error: --dir needs a value" >&2; exit 2; }; DIR=$2; shift ;;
    --only) [ $# -ge 2 ] || { echo "error: --only needs a value" >&2; exit 2; }; ONLY=$2; shift ;;
    --node-url) NODE_URL=$2; shift ;;
    --node-sha256) NODE_SHA256=$2; shift ;;
    --uv-url) UV_URL=$2; shift ;;
    --uv-sha256) UV_SHA256=$2; shift ;;
    *) echo "error: unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
case "$ONLY" in ''|node|uv) ;; *) echo "error: --only must be node or uv" >&2; exit 2 ;; esac
for cmd in curl sha256sum tar; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "error: required command not found: $cmd" >&2; exit 1; }
done

mkdir -p "$DIR/tool-bin" "$DIR/dl"
WORK=$(mktemp -d "$DIR/dl/install.XXXXXX")
trap 'find "$WORK" -delete' EXIT

# fetch URL SHA256 FILE: download, verify, and leave the file only when the hash matches.
fetch() {
  local url=$1 sha=$2 out=$3
  curl -fsSL --retry 3 -o "$out" "$url" || { echo "error: download failed: $url" >&2; exit 1; }
  local got
  got=$(sha256sum "$out" | awk '{print $1}')
  if [ "$got" != "$sha" ]; then
    echo "error: $url has sha256 $got, expected $sha; nothing was installed" >&2
    exit 1
  fi
  echo "verified $(basename "$out") ($(wc -c < "$out" | tr -d ' ') bytes, sha256 $sha)"
}

link() { ln -sfn "$1" "$DIR/tool-bin/$2"; }

install_node() {
  local target="$DIR/node-${NODE_VERSION}-linux-x64"
  if [ -x "$target/bin/node" ]; then echo "node ${NODE_VERSION} is already in $target"; else
    fetch "$NODE_URL" "$NODE_SHA256" "$WORK/node.tar.xz"
    mkdir "$WORK/node" && tar -xJf "$WORK/node.tar.xz" -C "$WORK/node" --strip-components=1
    [ -x "$WORK/node/bin/node" ] || { echo "error: the Node archive has no bin/node" >&2; exit 1; }
    mv "$WORK/node" "$target"
    echo "installed node into $target"
  fi
  for b in node npm npx; do link "$target/bin/$b" "$b"; done
}

install_uv() {
  local target="$DIR/uv-${UV_VERSION}"
  if [ -x "$target/uv" ]; then echo "uv ${UV_VERSION} is already in $target"; else
    fetch "$UV_URL" "$UV_SHA256" "$WORK/uv.tar.gz"
    mkdir "$WORK/uv" && tar -xzf "$WORK/uv.tar.gz" -C "$WORK/uv" --strip-components=1
    [ -x "$WORK/uv/uv" ] && [ -x "$WORK/uv/uvx" ] || { echo "error: the uv archive has no uv and uvx" >&2; exit 1; }
    mv "$WORK/uv" "$target"
    echo "installed uv into $target"
  fi
  for b in uv uvx; do link "$target/$b" "$b"; done
}

[ "$ONLY" = uv ] || install_node
[ "$ONLY" = node ] || install_uv
echo "done: put $DIR/tool-bin first on PATH for chat_tools.py (scripts/start-remote-chat-tools.sh does)"
