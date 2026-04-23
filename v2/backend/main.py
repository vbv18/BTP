"""
Demand Forecasting API - FastAPI Backend
"""

import io
import os
import sys
import uuid
import warnings
import traceback
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
import openpyxl
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Demand Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ──────────────────────────────────────────────────────────────────
FORECAST_WEEKS = 26
INTERMITTENT_THRESHOLD = 0.6


# ═══════════════════════════════════════════════════════════════════════════
# FORECASTING LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def load_df(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    required = {"date", "sku_id", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["sku_id", pd.Grouper(key="date", freq="W")])["demand"]
        .sum()
        .reset_index()
        .sort_values(["sku_id", "date"])
        .reset_index(drop=True)
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    processed = []
    for sku in df["sku_id"].unique():
        s = df[df["sku_id"] == sku].copy()
        all_weeks = pd.date_range(s["date"].min(), s["date"].max(), freq="W")
        s = s.set_index("date").reindex(all_weeks)
        s["sku_id"] = sku
        s["demand"] = s["demand"].fillna(0)
        s = s.reset_index().rename(columns={"index": "date"})
        processed.append(s)
    return pd.concat(processed, ignore_index=True)


def classify_skus(df: pd.DataFrame):
    classification, zero_ratios = {}, {}
    for sku in df["sku_id"].unique():
        d = df[df["sku_id"] == sku]["demand"]
        zr = (d == 0).sum() / len(d)
        zero_ratios[sku] = zr
        classification[sku] = "intermittent" if zr > INTERMITTENT_THRESHOLD else "smooth"
    return classification, zero_ratios


def sku_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for sku in df["sku_id"].unique():
        d = df[df["sku_id"] == sku]["demand"]
        stats[sku] = {
            "avg_demand": d.mean(),
            "std_demand": d.std() if len(d) > 1 else 0,
            "zero_ratio": (d == 0).sum() / len(d),
        }
    return stats


def create_features(df: pd.DataFrame, stats: dict):
    df = df.copy()
    le = LabelEncoder()
    df["sku_encoded"] = le.fit_transform(df["sku_id"])
    sku_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["trend"] = df.groupby("sku_id").cumcount()

    featured = []
    for sku in df["sku_id"].unique():
        s = df[df["sku_id"] == sku].copy()
        s["lag_1"] = s["demand"].shift(1)
        s["lag_2"] = s["demand"].shift(2)
        s["lag_4"] = s["demand"].shift(4)
        s["rolling_mean_2"] = s["demand"].shift(1).rolling(2, min_periods=1).mean()
        s["rolling_mean_4"] = s["demand"].shift(1).rolling(4, min_periods=1).mean()
        s["sku_avg_demand"] = stats[sku]["avg_demand"]
        s["sku_std_demand"] = stats[sku]["std_demand"]
        s["sku_zero_ratio"] = stats[sku]["zero_ratio"]
        featured.append(s)

    result = pd.concat(featured, ignore_index=True).fillna(0)
    return result, le, sku_mapping


FEATURES = [
    "sku_encoded", "week_of_year", "month", "week_sin", "week_cos", "trend",
    "lag_1", "lag_2", "lag_4", "rolling_mean_2", "rolling_mean_4",
    "sku_avg_demand", "sku_std_demand", "sku_zero_ratio",
]


def train_model(train_df: pd.DataFrame, quantile: float):
    clean = train_df.dropna(subset=["lag_4"]).copy().sort_values("date")
    if len(clean) < 10:
        return None
    split = int(len(clean) * 0.9)
    tr, val = clean.iloc[:split], clean.iloc[split:]
    weights = np.where(tr["demand"] > 0, 2.0, 1.0)
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=quantile,
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, random_state=42, verbosity=-1,
    )
    model.fit(
        tr[FEATURES], tr["demand"],
        sample_weight=weights,
        eval_set=[(val[FEATURES], val["demand"])] if len(val) > 0 else None,
        callbacks=[lgb.early_stopping(20, verbose=False)] if len(val) > 0 else None,
    )
    return model


def forecast_smooth(sku_data, stats, sku_encoded, m10, m50, m90):
    history = list(sku_data["demand"].values)
    last_date = sku_data["date"].max()
    sku = sku_data["sku_id"].iloc[0]
    sa, ss, sz = stats[sku]["avg_demand"], stats[sku]["std_demand"], stats[sku]["zero_ratio"]
    init_trend = len(sku_data)
    rows = []

    for i in range(FORECAST_WEEKS):
        fd = last_date + timedelta(weeks=i + 1)
        l1 = history[-1] if len(history) >= 1 else 0
        l2 = history[-2] if len(history) >= 2 else 0
        l4 = history[-4] if len(history) >= 4 else 0
        r2 = np.mean(history[-2:]) if len(history) >= 2 else np.mean(history)
        r4 = np.mean(history[-4:]) if len(history) >= 4 else np.mean(history)
        woy = fd.isocalendar()[1]
        X = pd.DataFrame({
            "sku_encoded": [sku_encoded], "week_of_year": [woy],
            "month": [fd.month], "week_sin": [np.sin(2 * np.pi * woy / 52)],
            "week_cos": [np.cos(2 * np.pi * woy / 52)], "trend": [init_trend + i],
            "lag_1": [l1], "lag_2": [l2], "lag_4": [l4],
            "rolling_mean_2": [r2], "rolling_mean_4": [r4],
            "sku_avg_demand": [sa], "sku_std_demand": [ss], "sku_zero_ratio": [sz],
        })
        p10 = max(0, m10.predict(X)[0])
        p50 = max(0, m50.predict(X)[0])
        p90 = max(0, m90.predict(X)[0])
        recent_mean = np.mean(history[-4:]) if len(history) >= 4 else np.mean(history)
        p50 = min(p50, 3 * max(recent_mean, 1))
        p10 = min(p10, p50)
        p90 = max(p90, p50)
        p10 = max(0, p50 - 1.2 * abs(p50 - p10))
        p90 = min(p50 + 1.2 * abs(p90 - p50), 5 * max(p50, 1))
        rows.append({"date": fd, "forecast_p10": p10, "forecast_p50": p50, "forecast_p90": p90})
        history.append(p50)

    return pd.DataFrame(rows)


def forecast_intermittent(sku_data):
    demand = sku_data["demand"].values
    non_zero = demand[demand > 0]
    last_date = sku_data["date"].max()
    dates = pd.date_range(last_date + timedelta(weeks=1), periods=FORECAST_WEEKS, freq="W")

    if len(non_zero) == 0:
        return pd.DataFrame({"date": dates, "forecast_p10": 0, "forecast_p50": 0, "forecast_p90": 0})

    recent_nz = non_zero[-10:] if len(non_zero) > 10 else non_zero
    avg = np.mean(recent_nz)
    std = np.std(recent_nz) if len(recent_nz) > 1 else avg * 0.3
    prob = (demand[-10:] > 0).mean() if len(demand) >= 10 else (demand > 0).mean()
    sku = sku_data["sku_id"].iloc[0]
    np.random.seed(hash(str(sku)) % 10000)

    rows = []
    for fd in dates:
        sims = [
            max(0, np.random.normal(avg, std)) if np.random.random() < prob else 0
            for _ in range(100)
        ]
        rows.append({
            "date": fd,
            "forecast_p10": np.percentile(sims, 10),
            "forecast_p50": np.percentile(sims, 50),
            "forecast_p90": np.percentile(sims, 90),
        })
    return pd.DataFrame(rows)


def run_pipeline(demand_bytes: bytes, actual_bytes: bytes):
    # Load & prep
    df = load_df(demand_bytes)
    df = aggregate_weekly(df)
    df = preprocess(df)
    classification, _ = classify_skus(df)
    stats = sku_stats(df)
    df, le, sku_mapping = create_features(df, stats)

    # Train
    m10 = train_model(df, 0.1)
    m50 = train_model(df, 0.5)
    m90 = train_model(df, 0.9)
    use_global = all(m is not None for m in [m10, m50, m90])

    # Forecast
    all_forecasts = []
    for sku in df["sku_id"].unique():
        sd = df[df["sku_id"] == sku].copy()
        if classification[sku] == "smooth" and use_global:
            fdf = forecast_smooth(sd, stats, sku_mapping[sku], m10, m50, m90)
        else:
            fdf = forecast_intermittent(sd)
        fdf["sku_id"] = sku
        all_forecasts.append(fdf)

    forecasts = pd.concat(all_forecasts, ignore_index=True)[
        ["date", "sku_id", "forecast_p10", "forecast_p50", "forecast_p90"]
    ]

    # Evaluate
    actual_df = load_df(actual_bytes)
    actual_weekly = (
        actual_df.groupby(["sku_id", pd.Grouper(key="date", freq="W")])["demand"]
        .sum()
        .reset_index()
    )
    merged = forecasts.merge(actual_weekly, on=["date", "sku_id"], how="inner")

    metrics_rows = []
    preview_rows = []

    if len(merged) > 0:
        # Overall metrics
        mae = mean_absolute_error(merged["demand"], merged["forecast_p50"])
        rmse = float(np.sqrt(mean_squared_error(merged["demand"], merged["forecast_p50"])))
        denom = merged["demand"].sum()
        wape = float(np.abs(merged["demand"] - merged["forecast_p50"]).sum() / denom * 100) if denom > 0 else 0.0
        coverage = float(
            ((merged["demand"] >= merged["forecast_p10"]) & (merged["demand"] <= merged["forecast_p90"])).mean() * 100
        )

        metrics_rows.append({
            "sku_id": "ALL",
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "WAPE (%)": round(wape, 2),
            "Coverage P10-P90 (%)": round(coverage, 2),
        })

        # Per-SKU metrics
        for sku in sorted(merged["sku_id"].unique()):
            sd = merged[merged["sku_id"] == sku]
            mae_s = mean_absolute_error(sd["demand"], sd["forecast_p50"])
            rmse_s = float(np.sqrt(mean_squared_error(sd["demand"], sd["forecast_p50"])))
            d = sd["demand"].sum()
            wape_s = float(np.abs(sd["demand"] - sd["forecast_p50"]).sum() / d * 100) if d > 0 else 0.0
            cov_s = float(
                ((sd["demand"] >= sd["forecast_p10"]) & (sd["demand"] <= sd["forecast_p90"])).mean() * 100
            )
            metrics_rows.append({
                "sku_id": str(sku),
                "MAE": round(mae_s, 2),
                "RMSE": round(rmse_s, 2),
                "WAPE (%)": round(wape_s, 2),
                "Coverage P10-P90 (%)": round(cov_s, 2),
            })

        # Preview (first 10 rows of merged)
        preview = merged.head(10).copy()
        preview["date"] = preview["date"].dt.strftime("%d-%m-%Y")
        for col in ["forecast_p10", "forecast_p50", "forecast_p90", "demand"]:
            preview[col] = preview[col].round(2)
        preview_rows = preview.to_dict(orient="records")

    metrics_df = pd.DataFrame(metrics_rows)

    # Build Excel
    out_path = OUTPUTS_DIR / f"forecast_{uuid.uuid4().hex[:8]}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        forecast_out = forecasts.copy()
        forecast_out["date"] = forecast_out["date"].dt.strftime("%d-%m-%Y")
        for col in ["forecast_p10", "forecast_p50", "forecast_p90"]:
            forecast_out[col] = forecast_out[col].round(2)
        forecast_out.to_excel(writer, sheet_name="Forecast", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    return {
        "excel_path": str(out_path),
        "excel_name": out_path.name,
        "preview": preview_rows,
        "metrics": metrics_rows,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/forecast")
async def forecast(
    demand_file: UploadFile = File(...),
    actual_file: UploadFile = File(...),
):
    try:
        demand_bytes = await demand_file.read()
        actual_bytes = await actual_file.read()
        result = run_pipeline(demand_bytes, actual_bytes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}\n{traceback.format_exc()}")


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )


# ─── Entry ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=False)
