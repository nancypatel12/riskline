PY := .venv/bin/python

.PHONY: install train serve test lint drift bench privacy sweep sweep-seeds fairness calibration docker clean

install:
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.lock
	$(PY) -m pip install -q --no-deps -e .

train:
	$(PY) -m riskline.train

serve:
	$(PY) -m uvicorn riskline.serve:app --port 8000

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

drift:
	$(PY) scripts/simulate_drift.py

bench:
	$(PY) scripts/benchmark.py

privacy:
	$(PY) scripts/privacy_audit.py

sweep:
	$(PY) scripts/leakage_sweep.py

# The same sweep on two more seeds, added to reports/leakage.json. Run after sweep; about 18 minutes.
sweep-seeds:
	$(PY) scripts/leakage_sweep.py --extra-seeds 43 44

fairness:
	$(PY) scripts/fairness_audit.py

calibration:
	$(PY) scripts/calibration_compare.py

docker:
	docker build -t riskline .
	docker run --rm -p 8000:8000 riskline

clean:
	rm -rf models reports/*.json data
