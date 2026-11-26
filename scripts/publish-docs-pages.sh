#!/usr/bin/env bash
# Build the VitePress docs site and force-push it to the gh-pages branch for GitHub Pages.
# Needs network, Node, and push access to origin. Does not change main.
set -euo pipefail
case "${1:-}" in
  -h|--help) echo "usage: scripts/publish-docs-pages.sh"; exit 0 ;;
  "") ;;
  *) echo "publish-docs-pages.sh: unknown argument $1 (see --help)" >&2; exit 2 ;;
esac
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dist="$root/docs-site/.vitepress/dist"
work=$(mktemp -d)
cleanup() { find "$work" -delete; }
trap cleanup EXIT

[[ -d "$root/docs-site/node_modules" ]] || npm ci --prefix "$root/docs-site" --ignore-scripts --no-audit --no-fund
# Always the project-Pages base, even if someone exported DOCS_BASE=/ for local preview.
DOCS_BASE=/kevin-ai/ npm --prefix "$root/docs-site" run docs:build
[[ -f "$dist/index.html" ]] || { echo "publish-docs-pages.sh: missing $dist/index.html" >&2; exit 1; }

cp -R "$dist"/. "$work"/
touch "$work/.nojekyll"
git -C "$work" init -q -b gh-pages
git -C "$work" config user.email "$(git -C "$root" config user.email)"
git -C "$work" config user.name "$(git -C "$root" config user.name)"
git -C "$work" add -A
git -C "$work" commit -q -m "docs: publish VitePress site to GitHub Pages"
git -C "$work" remote add origin "$(git -C "$root" remote get-url origin)"
git -C "$work" push -u origin gh-pages --force
echo "published https://marcfs31.github.io/kevin-ai/"
