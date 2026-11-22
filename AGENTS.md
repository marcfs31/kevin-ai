# Kevin AI: rules for coding agents

Created by Marc Fors i Soler. The standing rules in `~/.claude/CLAUDE.md` apply; this file adds to them.

## Releases: every merged PR is a version, and every version is tagged

Rule (owner, 2026-11-22). Why: 28 merged PRs sat untagged until v0.2.1 to v0.2.28 were added by hand, and
v0.3.2 and v0.3.3 had to be tagged afterwards the same way.

- **One PR, one version.** Before 1.0 a PR is a patch release (`0.3.4` to `0.3.5`); only a breaking or
  phase change bumps the minor number, and a major ships alone in its own PR.
- **In the PR**, as its own last commit `chore(release): X.Y.Z`: set `version` in `web/package.json` and in
  both places of `web/package-lock.json`, and move the PR's `CHANGELOG.md` entries from `[Unreleased]` to a
  new `## [X.Y.Z] - date` section. One version, identical in every manifest.
- **After the rebase-merge**, tag the merged commit and push only that tag:
  `git tag -a vX.Y.Z <merged sha> -m "vX.Y.Z: <title> (#N)"` then `git push origin vX.Y.Z`.
  Check first that the tagged commit's `web/package.json` says X.Y.Z. Never move or delete a pushed tag.
- A PR that was merged without a tag gets one on its last commit (annotated, message says it was tagged
  afterwards); the next release commit brings the manifests up to date.

## Gate (run before opening a PR, quote the output)

- `make test`: the C tests, the shell tests under `tests/scripts`, the tool service and proxy Python tests,
  and the web lint, typecheck, vitest and build (`tests/scripts/test_web_and_tools.sh`, needs Node).
- `scripts/lighthouse.sh`: Lighthouse on the production build, desktop and mobile; it must stay at 100 in
  every category. Measure that, not the Vite dev server.

## Where to edit

| Change | Where |
| --- | --- |
| Chat UI, styles, strings (en and es together) | `web/src` (`styles.css`, `i18n/en.ts` and `es.ts`) |
| Tools, MCP, web search | `tools/chat_tools.py` (service), `web/src/lib/tool*.ts` (page) |
| The proxy and static server (CSP, gzip) | `tools/chat_server.py` |
| Server scripts (paths use `~/kevin-ai`) | `scripts/` |

## Do not

- Edit `AGENTS.md`, `CLAUDE.md`, `.github/workflows/**` or `dependabot.yml` from an automated session: those
  go in their own PR that the owner merges.
- Rename the repository, move the checkout on the model server or restart the loaded model unasked.
- Add a stylesheet, font, image URL or script from another origin: the CSP is `'self'` only.

## Before you finish

Tests with the change, `CHANGELOG.md` entry, release commit, the gate output, and the tag after the merge.
