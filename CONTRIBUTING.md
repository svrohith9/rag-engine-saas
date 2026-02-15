# Contributing

Thanks for contributing.

## Project Principles
- Prefer correctness and clarity over cleverness.
- Keep the happy path fast; keep the failure mode obvious.
- No secrets in git history (no API keys, tokens, `.env` files).

## Local Development
See:
- `docs/development.md`
- `backend/README.md`
- `frontend/README.md`

## Pull Requests
- Keep PRs small and focused.
- Update docs for behavior changes.
- Add or update tests where it makes sense.

### Checklist
- [ ] I ran the backend locally and exercised the modified endpoints.
- [ ] I ran the frontend locally and verified UI behavior.
- [ ] I did not commit secrets (`.env`, keys, tokens).
- [ ] I updated docs if needed.

## Commit Style
- Use imperative present tense: "Add …", "Fix …", "Refactor …".

## Code Style
- Python: keep functions small, avoid global state, prefer explicit types for public APIs.
- TS/React: keep components focused; move API and types to `src/lib/`.
