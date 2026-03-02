# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install-dev
make test
```

## Workflow

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make changes with passing tests: `make test`
3. Lint before committing: `make lint`
4. Commit using conventional format (see below)
5. Push and open a PR

## Commit Convention

Single-line conventional commits:

```
feat(scope): short description
fix(scope): short description
refactor(scope): short description
chore: short description
docs: short description
test(scope): short description
```

Examples:
- `feat(path): add A* fallback router`
- `fix(energy): clamp negative battery to zero`
- `refactor(core): extract sensing sub-package`
- `chore: pin requirements`

## Project Structure

See [README.md](README.md#architecture) for the full architecture tree.

## Testing

- All tests live in `tests/` and use pytest
- Shared fixtures are in `tests/conftest.py`
- Run with `make test` or `python -m pytest tests/ -v`
