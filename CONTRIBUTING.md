# Contributing

Vulcan's contributor contract lives in `docs/ROADMAP.md` §1 ("How to work
this plan"). The short version:

- The four-command quality gate must pass before every push, no exceptions:
  `uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest`,
  `uv run python scripts/smoke.py`.
- Tests are extended, never replaced; nothing in the suite may contact a
  real API (`httpx.MockTransport` for every new upstream surface).
- Every new endpoint, adapter, stream, or CLI output gets leak tests proving
  credentials, prompts, and upstream bodies cannot reach responses, errors,
  or logs.
- Re-verify the non-negotiable invariants checklist (`docs/ROADMAP.md` §3)
  in every PR description.
