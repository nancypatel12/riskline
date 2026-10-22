# riskline

[![CI](https://github.com/nancypatel12/riskline/actions/workflows/ci.yml/badge.svg)](https://github.com/nancypatel12/riskline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Credit default risk, end to end. One config, one seed, one command per stage — from a checksummed public dataset to a versioned, calibrated model behind a validated API, with drift alerts when the world changes and a privacy audit before it ships.

CPU only. No cloud. Every number below is reproduced by `make train drift bench privacy`.

## Run it

```bash
make install
make train
make serve        # http://localhost:8000/docs
```

Or `make docker` — the image trains on the real data at build time and serves as a non-root user.

## Results

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350): 30,000 Taiwanese cardholders, 22.1% default next month. Stratified 60/20/20 split; the decision threshold is chosen on the calibration split, never on test.

| Model (test, n = 6,000) | ROC-AUC | PR-AUC | Brier | ECE | Cost / applicant |
|---|---:|---:|---:|---:|---:|
| Approve everyone | — | — | — | — | 1.106 |
| Logistic regression | 0.742 | 0.505 | 0.142 | 0.014 | 0.587 |
| **Gradient boosting, calibrated** | **0.778** | **0.557** | **0.135** | **0.013** | **0.568** |

Cost: a missed default costs 5, a wrongly flagged good customer costs 1 (`config.yaml`). The cost-optimal threshold is **0.14** — close to the Bayes rule 1/(1+5) = 0.17, as it should be for calibrated scores. Against the naive 0.5 cutoff it cuts expected cost **25%** (0.754 → 0.568) and catches **81%** of defaults.

| Serving (1 uvicorn worker, M3 Pro) | p50 | p95 | p99 |
|---|---:|---:|---:|
| `POST /predict` | 6.1 ms | 7.8 ms | 8.9 ms |
| `POST /predict/batch` (100 rows) | 8.2 ms | 9.2 ms | 10.8 ms |

Batch throughput: ~12,000 rows/s on one core.

| Drift check (vs. training reference) | Alerts | Mean score | Flag rate |
|---|---:|---:|---:|
| Held-out test data | 0 | 0.223 | 54% |
| Simulated downturn | 8 | 0.274 | 73% |

The downturn cuts limits 30%, grows balances 30%, and pushes 35% of customers one month further behind. PSI and KS flag the delay and limit features and the score distribution itself; clean data raises nothing.

| Privacy audit ([trainleak](https://github.com/nancypatel12/trainleak), 2,000 members vs 2,000 non-members) | Test acc. | Train acc. | LiRA AUC | TPR @ 1% FPR | Risk |
|---|---:|---:|---:|---:|---:|
| **Production model** | 0.815 | 0.827 | 0.560 | 2.3% | moderate |
| Overfit control (same features, no regularization) | 0.809 | 1.000 | 0.866 | 7.8% | severe |

Can an attacker with the model's scores tell whether a customer was in the training set? Against the production model, the strongest attack (LiRA, 16 reference models) is only slightly better than a coin flip. The overfit control scores almost the same on test accuracy, yet the same attack reaches AUC 0.87. Accuracy alone would ship it; the privacy gate would not. `make privacy` exits non-zero above "moderate".

Raw numbers: [`reports/`](reports).

## Architecture

```mermaid
flowchart LR
    A[UCI zip<br/>SHA-256 pinned] --> B[validate<br/>schema + ranges]
    B --> C[split<br/>train / calib / test]
    C --> D[FeatureBuilder<br/>+ gradient boosting]
    D --> E[calibrate<br/>on calib split]
    E --> F[cost-optimal<br/>threshold]
    F --> G[(registry<br/>v1, v2, …)]
    G -->|production| H[FastAPI<br/>/predict · /predict/batch]
    G -->|reference| I[drift<br/>PSI + KS]
    G -->|members vs test| K[privacy gate<br/>trainleak LiRA]
    J[live traffic] --> H
    J --> I
```

## What's inside

| | |
|---|---|
| `riskline/schema.py` | One contract. Feature names and valid ranges drive both data validation and the Pydantic API models. |
| `riskline/data.py` | Fetch once, verify SHA-256 every time, report every validation failure at once. |
| `riskline/features.py` | Stateless sklearn transformer: folds undocumented category codes, adds utilization, repayment ratio, delinquency counts. |
| `riskline/train.py` | Seeded split → `HistGradientBoostingClassifier` (native categoricals, early stopping) → sigmoid calibration → threshold → model card. |
| `riskline/registry.py` | Versioned artifacts (`model.joblib`, `card.json`, `reference.parquet`). A new version is promoted only if its PR-AUC matches or beats production. |
| `riskline/serve.py` | `/health`, `/model`, `/predict`, `/predict/batch` (≤ 1,000 rows). Unknown fields and out-of-range values return 422. Every response carries `X-Latency-Ms`. |
| `riskline/drift.py` | Population Stability Index (quantile bins; value bins for discrete features) and two-sample KS per feature. |
| `riskline/privacy.py` | Membership inference audit through [trainleak](https://github.com/nancypatel12/trainleak): adapts the pipeline to trainleak's numpy interface and retrains reference models for LiRA. |

Each model card records the Python and scikit-learn versions, config hash, data hash, git commit and seed. Same inputs, same model.

## Decisions

- **Calibration method.** Sigmoid, not isotonic: isotonic creates ties that cost 2.6 points of PR-AUC here. Boosted trees on log-loss were already well calibrated (ECE 1.3%); the step stays as a guard and is cheap.
- **scikit-learn over LightGBM.** Same histogram-boosting algorithm, no OpenMP system dependency — a smaller image and a simpler CI.
- **Alert rule.** PSI ≥ 0.2 or KS ≥ 0.1. With thousands of rows every KS p-value is tiny; the statistic is what matters.
- **Privacy metric.** TPR at 1% FPR, not AUC alone: an attack that confidently identifies a few customers is a real leak even when its average looks harmless. Members are the rows the trees were fit on; non-members are the untouched test split.

## Tests

32 tests, ~9 s, no network: data validation rejects each failure mode, training is deterministic and beats chance, the threshold search recovers the Bayes rule, registry round-trips and gates, the API rejects bad input, PSI separates shifted from unshifted data, the privacy audit separates an overfit model from a regularized one. CI runs lint and tests on Python 3.12 and 3.13, then builds the Docker image — training on the real dataset — and smoke-tests it.

## License

[MIT](LICENSE)
