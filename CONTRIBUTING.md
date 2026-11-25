# Contributing

Thanks for looking at the predictive-maintenance pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m maintenance.pipeline
streamlit run dashboard/app.py
```

## Checks

```bash
ruff check src tests scripts dashboard
pytest
python -m maintenance.cli check
```

## Guidelines

- Keep sensor units unchanged: Kelvin, rpm, Nm and minutes.
- Add derived features in `features.py` and document the physical intuition.
- Do not tune the threshold on the test split; use validation only.
- Update `docs/MAINTENANCE_POLICY.md` if the cost ratio changes.
- Commit generated artifacts with the code that produced them.
