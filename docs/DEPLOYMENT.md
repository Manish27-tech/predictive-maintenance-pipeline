# Deployment

## Local

```bash
pip install -e ".[dev]"
python -m maintenance.pipeline
streamlit run dashboard/app.py
```

## Docker

```bash
docker build -t predictive-maintenance .
docker run --rm -p 8501:8501 predictive-maintenance
```

## GitHub Pages

The static site is published from `docs/` on `main`. Re-run the pipeline after a data or model change and commit `docs/` and `artifacts/` together.

## Retraining checklist

1. Refresh `data/raw/ai4i2020.csv`.
2. Run `python -m maintenance.pipeline`.
3. Run `ruff check src tests scripts dashboard`.
4. Run `pytest`.
5. Run `python -m maintenance.cli check`.
6. Review failure-mode recall and the tool-wear gap.
7. Publish only if calibration and cost are acceptable.

## Monitoring

- Score distribution and calibration by product type.
- Failure-mode coverage, especially tool wear.
- Sensor drift in temperature, torque and speed.
- False-alarm burden on the maintenance team.
- Cost ratio changes from finance and operations.
