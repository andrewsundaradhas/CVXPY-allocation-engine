# Allocation Engine

Portfolio allocation engine: given a universe of tickers and a risk preference,
returns optimal weights plus risk/return statistics.

Layers:
- **Data** — yfinance loader, returns, features.
- **Stats** — Ledoit-Wolf shrinkage covariance.
- **ML** — gradient-boosted expected-return forecaster, blended with a prior.
- **Optimizer** — CVXPY mean-variance / risk-parity / max-Sharpe with
  long-only, budget, sector, and turnover constraints.
- **Interface** — FastAPI REST + Streamlit dashboard.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify CVXPY + CLARABEL:

```bash
python -c "import cvxpy; print(cvxpy.__version__); print(cvxpy.installed_solvers())"
```

## Tests

```bash
pytest
```

## Status

- [x] Phase 0 — environment
- [x] Phase 1 — data layer (loader, returns)
- [ ] Phase 2 — covariance + features
- [ ] Phase 3 — ML expected returns
- [ ] Phase 4 — CVXPY optimizer
- [ ] Phase 5 — engine orchestrator
- [ ] Phase 6 — FastAPI + Streamlit
