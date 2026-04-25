from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .engine import AllocationRequest, run_allocation
from .schemas import AllocationIn, AllocationOut

app = FastAPI(title="CVXPY Allocation Engine", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/allocate", response_model=AllocationOut)
def allocate(req: AllocationIn):
    try:
        out = run_allocation(AllocationRequest(**req.model_dump()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AllocationOut(
        tickers=out["tickers"],
        weights=out["weights"].tolist() if hasattr(out["weights"], "tolist") else list(out["weights"]),
        expected_return=out["expected_return"],
        volatility=out["volatility"],
        sharpe=out["sharpe"],
        status=out["status"],
    )
