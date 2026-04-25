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

## Run the API

```bash
uvicorn allocator.api:app --reload --port 8000
```

Then:

```bash
curl -X POST localhost:8000/allocate -H 'Content-Type: application/json' \
  -d '{"tickers":["AAPL","MSFT","JPM","XOM","PG"],"objective":"mean_variance","risk_aversion":4}'
```

## Status

- [x] Phase 0 — environment
- [x] Phase 1 — data layer (loader, returns)
- [x] Phase 2 — covariance (Ledoit-Wolf) + features
- [x] Phase 3 — ML expected returns + shrinkage blend
- [x] Phase 4 — CVXPY optimizer (mean-variance, min-variance, max-Sharpe, risk-parity)
- [x] Phase 5 — engine orchestrator
- [x] Phase 6a — FastAPI service (`/health`, `/allocate`)
- [ ] Phase 6b — Streamlit dashboard
