# Demand Forecast — Setup & Run

## Requirements
- Windows OS
- Python 3.13

## Run (one command)

```powershell
run.bat
```

Then open: http://localhost:8000

---

## CSV Format

Both `demand.csv` and `actual.csv` must have:

| Column  | Format       |
|---------|--------------|
| date    | DD-MM-YYYY   |
| sku_id  | string/int   |
| demand  | numeric      |

---

## Output Excel (2 sheets)

| Sheet     | Columns                                      |
|-----------|----------------------------------------------|
| Forecast  | date, sku_id, forecast_p10, forecast_p50, forecast_p90 |
| Metrics   | sku_id, MAE, RMSE, WAPE (%), Coverage P10-P90 (%) |

---

## Project Structure

```
demand_forecast/
├── backend/
│   └── main.py          ← FastAPI app + full pipeline
├── frontend/
│   └── index.html       ← Single-page UI
├── outputs/             ← Excel files saved here
├── requirements.txt
├── run.bat              ← Windows start script
└── README.md
```
