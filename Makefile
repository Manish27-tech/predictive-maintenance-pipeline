.PHONY: install data train test lint check dashboard clean

install:
	pip install -e ".[dev]"

data:
	python scripts/fetch_data.py

train:
	python -m maintenance.pipeline

test:
	pytest

lint:
	ruff check src tests scripts dashboard

check:
	python -m maintenance.cli check

dashboard:
	streamlit run dashboard/app.py

clean:
	rm -rf .pytest_cache .ruff_cache artifacts/figures docs/figures reports/figures
