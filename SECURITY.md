# Security policy

## Supported versions

Only the latest commit on `main` receives security fixes. There are no released versions yet.

## Reporting a vulnerability

Do not open a public issue for a security problem. Report it privately through GitHub:
go to the repository's **Security** tab and choose **Report a vulnerability**. This
creates a private advisory that only the maintainer can see.

Please include:

- what you found and where (file, function or command)
- steps to reproduce, or a proof of concept
- the impact you expect

You will get an acknowledgement within a week. Once a fix is ready it is released on
`main` and the advisory is published with credit to you, unless you ask otherwise.

## Scope

In scope: the C engine (`src/`, `include/`), its loaders (safetensors, the q8 format),
the tokenizer and chat template handling, and the scripts under `scripts/`.

Out of scope: the upstream model weights and the upstream projects this code was ported
from; report those to their own maintainers.
