# Contributing

Thanks for looking at this. The engine is small on purpose: it is written for an
8 GB machine, and every line has to earn its place.

## Build and test

```sh
make            # build bin/*
make test       # unit tests (must pass)
make gate       # golden gates for Qwen3.5 and Gemma 4 (must pass)
make sanitize   # unit tests under ASan and UBSan
```

Run the heavier checks before pushing:

```sh
git config core.hooksPath .githooks   # once per clone; runs scripts/ci-heavy.sh on push
```

Requirements: a C99 compiler, `make`, and Python 3 (standard library only; do not add
third-party Python packages to `tools/`).

## Ground rules

- **No weights in the repository.** Model checkpoints are downloaded with
  `scripts/download-model.sh`. Test fixtures under `tests/fixtures/` are synthetic.
- **Only commercially usable data.** Any corpus or data file must be public domain, CC0,
  MIT, Apache-2.0, BSD, or CC BY/BY-SA with attribution recorded beside it.
- **No secrets.** Never commit `.env`, `.env.local` or any other secret file. Document new
  configuration in `.env.example` only.
- **Fix causes, not signals.** No lint suppressions, skipped tests, lowered coverage or
  silenced warnings.
- **Behaviour changes ship tests.** A bug fix needs a test that fails before the fix and
  passes after it, at the lowest layer that owns the behaviour.
- **Vendored and ported code** keeps its licence header and is recorded in
  [NOTICE](NOTICE). Changes to a vendored file are listed there too.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat`, `fix`, `docs`,
  `test`, `build`, `ci`, `chore`, `refactor`, `perf`.
- One logical change per commit. Keep tests in the same commit as the behaviour they cover.
- Keep pull request branches free of merge commits; rebase onto `main` instead.
- Pull requests are merged by rebase, so every commit must build and pass the tests on its
  own.
- Describe why the change is needed, not only what it does.

## Licence

By contributing you agree that your contribution is licensed under the Apache License 2.0,
the licence of this project.
