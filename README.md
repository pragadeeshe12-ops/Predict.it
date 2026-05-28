# Pridict.it

Stock market analysis and next-day **range prediction** using XGBoost. The pipeline downloads historical OHLCV data, engineers technical indicators, trains a single model on multiple Indian equities, and publishes the trained model to Hugging Face Hub.

---

## What this project does

1. **Fetch data** — Downloads daily price history for a list of NSE tickers (e.g. `RELIANCE.NS`, `TCS.NS`).
2. **Process data** — Computes indicators (RSI, EMA, volume spikes, fair value gaps, etc.) and writes a combined dataset with a `stock_name` column per row.
3. **Train** — Fits an **XGBoost regressor** to predict `target_range` (next day’s high − low).
4. **Publish model** — Saves the model locally and uploads it to a Hugging Face model repository.

The model predicts **expected daily range**, not a single exact future price. Trading signals (BUY / WEAK BUY / NO TRADE) can be derived from range plus RSI, volume, and support logic when prediction helpers are enabled.

---

## Recent changes

| Area | Change |
|------|--------|
| **Multi-stock training** | `main.py` loops over tickers in `constant.stock_price`, builds one combined `processed_data/processed_data.csv`. |
| **`stock_name` column** | Each row is tagged with its ticker in `data_source.add_technical_indicators()`. |
| **XGBoost + categorical stocks** | `encode_stock_features()` in `Xboost.py` one-hot encodes `stock_name` (e.g. `stock_RELIANCE.NS`) so training no longer fails on string columns. |
| **Feature alignment** | `train_model()` stores `model.feature_columns` for consistent inference column order. |
| **Training entrypoint** | Training orchestration moved to `main/main.py`; `Xboost.py` holds ML utilities and save/upload. |
| **Hugging Face upload** | `save_model()` writes `main/xgb_model.pkl` and uploads to Hub repo `praga-deesh/predict.id`. |

---

## External data sources

| Source | Library | Purpose |
|--------|---------|---------|
| **Yahoo Finance** | [`yfinance`](https://pypi.org/project/yfinance/) | Historical OHLCV (Open, High, Low, Close, Volume) for Indian tickers such as `*.NS` |

No other external APIs are required for the current training pipeline. Data is pulled at runtime when you run `main.py` (network required).

---

## Where the ML model is stored

| Location | Path / URL | Notes |
|----------|------------|--------|
| **Local (default)** | `main/xgb_model.pkl` | Created by `joblib.dump()` in `save_model()` |
| **Hugging Face Hub** | [praga-deesh/predict.id](https://huggingface.co/praga-deesh/predict.id) | Uploaded via `huggingface_hub.upload_file` as `xgb_model.pkl` |

To load from Hugging Face (inference / API):

```python
from huggingface_hub import hf_hub_download
import joblib

path = hf_hub_download(
    repo_id="praga-deesh/predict.id",
    filename="xgb_model.pkl",
)
model = joblib.load(path)
# If saved with feature_columns on the model object:
# feature_columns = model.feature_columns
```

Authenticate before upload/download:

```bash
pip install -r requirements.txt
huggingface-cli login

```

---

## Project structure

```
Pridict.it/
├── main/
│   ├── main.py              # Fetch all stocks → train → evaluate → save/upload
│   ├── data_source.py       # yfinance download + technical indicators
│   ├── Xboost.py            # Load/prepare features, train, evaluate, save_model
│   ├── constant.py          # Tickers list, paths, BASE_DIR
│   └── xgb_model.pkl        # Trained model (local artifact)
├── processed_data/
│   ├── historical_data.csv  # Last downloaded raw OHLCV (per run, overwritten)
│   └── processed_data.csv   # Combined multi-stock dataset with indicators
├── requirements.txt
└── README.md
```

---

## Configuration

Edit tickers in `main/constant.py`:

```python
stock_price = ["BEL.NS", "RELIANCE.NS", "TCS.NS", ...]
```

- `INPUT_CSV` — intermediate raw file: `processed_data/historical_data.csv`
- Combined training file — `processed_data/processed_data.csv`

---

## How to run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
python main/main.py
```

This will:

1. Download and process each ticker in `stock_price`
2. Append rows into `processed_data/processed_data.csv` (with `stock_name`)
3. Train XGBoost with an 80/20 time split
4. Print MAE / RMSE
5. Save `main/xgb_model.pkl` and upload to Hugging Face

---

## Model inputs and target

**Target (what we predict):**

- `target_range` — next trading day’s `High - Low`

**Features (examples):**

- Price/volume: `Close`, `Open`, `High`, `Low`, `Volume`, lags, returns
- Indicators: `rsi_14`, `ema_9`, `ema_21`, `volume_spike`, `rolling_high_10`, `rolling_low_10`, `atr_14`, FVG flags, etc.
- **Stock identity:** `stock_name` → one-hot columns like `stock_RELIANCE.NS` at training time

**Predicting one stock** (when inference code is used): filter `processed_data.csv` by `stock_name`, sort by `Date`, take the **last row**, then call the model with `feature_columns` aligned to training.

---

## Dependencies

See `requirements.txt`:

- `yfinance`, `pandas`, `numpy`
- `xgboost`, `scikit-learn`, `joblib`
- `huggingface_hub`

---

## Notes and limitations

- Predictions are **not financial advice**; the model is experimental.
- `yfinance` data quality and delays apply; verify tickers (`.NS` suffix for NSE).
- Training uses a simple global time split on the combined CSV; for production, consider splitting by date across all symbols.
- `get_stock_data()` currently removes `processed_data.csv` when starting each ticker download in the loop—only the last ticker’s fetch remains in `historical_data.csv` until appended; the combined file is built via append in `add_technical_indicators()`.

---

## License

Add your license here if applicable.
