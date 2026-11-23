# Kevin AI: rules for coding agents

Created by Marc Fors i Soler. The standing rules in `~/.claude/CLAUDE.md` apply; this file adds to them.

## Releases: every merged PR is a version, and every version is tagged

Rule (owner, 2026-11-22). Why: 28 merged PRs sat untagged until v0.2.1 to v0.2.28 were added by hand, and
v0.3.2 and v0.3.3 had to be tagged afterwards the same way.

- **One PR, one version.** Before 1.0 a PR is a patch release (`0.3.4` to `0.3.5`); a breaking or
  phase change bumps the minor number, and a major ships alone in its own PR.
- **Kevin's SemVer** (owner, 2026-11-23). Why: with one version per PR the patch number had reached 23 within two
  days of `0.3.0`, and it only grows. Patches run 0 to 31 in a minor; the release after `x.y.31` is the next minor,
  `x.(y+1).0`, which carries its own changes, not an empty bump. The first major comes early: the release after
  `0.3.31` is `1.0.0` (no `0.4` to `0.13`). From `1.0.0` on, a minor holds patches 0 to 31 and a major holds minors
  0 to 13, so `1.13.31` is followed by `2.0.0`, and so on. It is still `MAJOR.MINOR.PATCH`, so every tool that reads
  SemVer reads it. Check the last tag first. A major ships alone in its own PR (its version, changelog and migration
  notes); the change that was waiting goes in the next patch, `1.0.1`.
- **In the PR**, as its own last commit `chore(release): X.Y.Z`: set `version` in `web/package.json` and in
  both places of `web/package-lock.json`, and move the PR's `CHANGELOG.md` entries from `[Unreleased]` to a
  new `## [X.Y.Z] - date` section. One version, identical in every manifest.
- **After the rebase-merge**, tag the merged commit and push only that tag:
  `git tag -a vX.Y.Z <merged sha> -m "vX.Y.Z: <title> (#N)"` then `git push origin vX.Y.Z`.
  Check first that the tagged commit's `web/package.json` says X.Y.Z. Never move or delete a pushed tag.
- A PR that was merged without a tag gets one on its last commit (annotated, message says it was tagged
  afterwards); the next release commit brings the manifests up to date.
- Find the next number from the tags, not from memory: `git tag -l 'v*' --sort=version:refname | tail -3`.
  Tag only after CI is green and the PR is merged; never tag a branch commit, which a rebase-merge rewrites.
- State the tag in the closing report: the version, the commit, and that the pushed tag was checked
  (`git ls-remote --tags origin vX.Y.Z`).

## Gate (run before opening a PR, quote the output)

- `make test`: the C tests, the shell tests under `tests/scripts`, the tool service and proxy Python tests,
  and the web lint, typecheck, vitest and build (`tests/scripts/test_web_and_tools.sh`, needs Node).
- `scripts/lighthouse.sh` for any change under `web/`, `tools/chat_server.py` or `index.html` (see below).

## Lighthouse: 100 in every category, desktop and mobile

Rule (owner, 2026-11-22). Why: the first report scored 47 on the Vite dev server; the production build
reached 100 only after four fixes, and a header change dropped mobile to 92 until its height was re-measured.

- A change under `web/`, `tools/chat_server.py` or `index.html` runs `scripts/lighthouse.sh` (needs Chrome and
  the network, so it is not in CI) and quotes its output in the PR. Below 100 is a bug in that PR, not a note.
- Measure the production build as `chat_server.py` serves it, never the Vite dev server (78 unbundled files
  and an HTML answer for every unknown path make its score meaningless).
- What has cost points, so check these first:
  - text sent uncompressed (gzip lives in `chat_server.py`);
  - content that arrives after the first paint pushing the page down: reserve its height (`.top` is 4.75rem,
    the phone `.server` row 7.125rem up to 28rem). Re-measure the loaded height at 360, 412 and 480 px whenever
    a control in that row changes size;
  - a `robots.txt` with `Disallow: /` (Lighthouse reads it as blocked from indexing: ship none);
  - a missing meta description; a heavy dependency loaded up front (lazy-load it, as the PDF reader is).

## Where to edit

| Change | Where |
| --- | --- |
| Chat UI, styles, strings (en and es together) | `web/src` (`styles.css`, `i18n/en.ts` and `es.ts`) |
| Tools, MCP, web search | `tools/chat_tools.py` (service), `web/src/lib/tool*.ts` (page) |
| The proxy and static server (CSP, gzip) | `tools/chat_server.py` |
| Project tools: files, git, commands, remote and pull requests, audit | `tools/chat_projects.py` (registry, dispatch), `chat_jail.py`, `chat_git.py`, `chat_commands.py`, `chat_remote.py`, `chat_audit.py`; a test beside each |
| Project picker, git chip, project rules for the model | `web/src/components/ProjectBar.tsx`, `lib/toolHost.ts`, `lib/projectGit.ts` |
| Server scripts (paths use `~/kevin-ai`) | `scripts/` (the two `start-*-chat-tools.sh` pass the projects base and load the token file) |

## Do not

- Edit `AGENTS.md`, `CLAUDE.md`, `.github/workflows/**` or `dependabot.yml` from an automated session: those
  go in their own PR that the owner merges.
- Rename the repository, move the checkout on the model server or restart the loaded model unasked.
- Add a stylesheet, font, image URL or script from another origin: the CSP is `'self'` only.
- Loosen what the agent can do in a project. Rule (owner, 2026-11-23). Why: the project tools run with the owner's
  rights on a machine that also holds the models, so the registry, not the agent, decides everything. Keep:
  paths relative and jailed (no `..`, no symlink out, nothing under `.git`); no force, no delete, no branch
  other than the current one on push; no commit on the default branch unless `allowMain`; the token only through
  the `GIT_ASKPASS` helper, scrubbed from every result; commands with a fixed argv and no shell; an audit line for
  every change. A tool never takes a URL, a remote, a token or a command line from the model. A new project tool
  ships a negative test for what it must refuse, in the same PR.
- Make the page ask for something that can fail on load. Why: asking `/tools/projects` while no tool service ran
  logged a console error and dropped Lighthouse Best Practices to 96. Ask only while the feature is switched on,
  and add any new `/tools` path to the list in `web/src/privacy.test.ts`.
- Write the project registry, the token file (`~/.config/kevin-ai/tool-service.env`) or a token into the
  repository, a log, a test fixture or a message: they are the owner's. Tests use fake tokens and temp registries.

## Before you finish

Tests with the change, `CHANGELOG.md` entry, release commit, the gate output, and the tag after the merge.
