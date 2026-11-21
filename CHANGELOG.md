# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-21

### Added
- Web chat: live token counter (tokens read and written per reply and per chat, estimates marked until the server's usage arrives).
- Web chat: compacting of older turns into a model-written summary, automatically near the context limit (also between tool steps) or with the Compact button.
- Web chat: per-conversation model picker over `/v1/models`, showing each context length.
- Web chat: optional tool loop (calculator, current time) with a collapsible trace, a step cap (default 8) and Stop.

## [0.1.0] - 2026-11-21

### Added
- Web chat app for lm-serve (Vite, React, TypeScript) with streaming, attachments, English and Spanish, and the `tools/chat_server.py` proxy.
