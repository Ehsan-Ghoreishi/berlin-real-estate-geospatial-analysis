# Contributing

Thanks for taking an interest in this project. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/Ehsan-Ghoreishi/berlin-real-estate-geospatial-analysis.git
cd berlin-real-estate-geospatial-analysis
python3 -m venv .venv && source .venv/bin/activate
make install-dev    # runtime + dev dependencies, installs pre-commit hooks
make run            # builds data/processed/ and maps/
```

## Before opening a pull request

```bash
make lint typecheck test
```

CI runs the same checks plus a Docker build, so a green local run means a green PR.

## Conventions

- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `build:`, `ci:`, `chore:`.
  One logical change per commit.
- **Pipeline stages** live in `src/`, expose a `run()` function and are registered
  in `STAGES` in `main.py`. Paths and constants come from `src/config.py`.
- **Logging**, not `print`, via a module-level `logger = logging.getLogger(__name__)`.
- **Tests** go in `tests/` and should use small in-memory frames (see
  `tests/conftest.py`) rather than the full dataset.
- **Results in the README** must come from a pipeline run; update
  `CHANGELOG.md` when they change.
