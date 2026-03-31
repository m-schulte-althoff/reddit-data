# Code Style
- Architecture: Model View Controller.
    - Controller: main.py for structure (only if needed: small helpers in "controllers.py").
    - View: views.py for tables/figures output.
    - Model: synonymous to the program's core logic. The individual modules in the ./src/ folder, such as for example preprocessing.py.
- Tooling:
    - pytest (tests/ folder; Naming: test_$MODULE_*.py).
    - No (expensive) LLM pipelines testing
    - ruff (“no ruff violations”).
    - mypy.
- Prefer explicit named imports when appropriate (e.g. from bar import foo).
- Include type hints (except for tiny helper functions) and short docstrings for all functions.
- Use Python logging module (not print) for pipeline steps.
- Log files stored under logs/ with timestamps in the filename (and via log formatter).
- Make the code easy to read and modify. No larger frameworks (Django/Flask) if not stated explicitly or mandatory for task fulfilment. Keep it minimal.

# Workflow
- Work modularly. Complete one module at a time. After each module, report what you built, show test result.
- Never claim something works without running a test to prove it.
- Prefer running single pytest tests, and not the whole test suite, for performance.
- Raw data is immutable: raw never edited; only derived outputs go to processed/.
- Use python3 instead of just python.
- As environment use "uv run" instead of virtual envs or conda. Create pyproject.toml + uv.lock accordingly.
- pyproject.toml should be the single config hub, among others tool configs for pytest/ruf

## Common commands (examples)
```bash
uv run python3 main.py preprocess

uv run pytest -q
uv run pytest -q tests/test_${MODULE}_*.py

uv run ruff check .
uv run mypy .
```

# Project Setup
Create the following directory structure (if not already existing): 
$PROJECT_ROOT/
├── README.md
├── INSTRUCTIONS.md
├── main.py
├── views.py
├── .gitignore
├── src/
├── data/
│   ├── raw/
│   ├── processed/
├── output/
│   ├── tables/
│   ├── figures/
├── logs/
└── tests/

# Output format
- File naming convention: ${MODULENAME}-${OUTPUTNAME}.${SUFFIX}.
- Preferred formats: `.csv` for tables/data; vector graphics (`.svg`/`.pdf`) over `.png` for figures.
- Determinism: fixed random seeds (where applicable) and stable sorting before writing outputs.
- Summary metrics: e.g., `output/tables/summary.csv`.


# Final Checks
- Documentation: create/update `README.md` explaining:
  (i) Project overview
  (ii) How to reproduce results
  (iii) Data sources
  (iv) File structure
- Update the .gitignore accordingly and with regards to tool use (all data: "data", "data/", uv files etc.).

