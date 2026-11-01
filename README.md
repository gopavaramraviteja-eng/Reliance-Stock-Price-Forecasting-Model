# RELIANCE Direction Forecast — Streamlit app

A Streamlit app that forecasts Reliance Industries' (NSE: RELIANCE) closing
price with a classical time-series model (ARIMA, optionally Prophet) and
turns that into an UP/DOWN directional call for N trading days ahead, with a
walk-forward backtest of how reliable that call has actually been.

## Files

- `app.py` — the Streamlit UI (run this).
- `nse_data.py` — fetches historical OHLCV data from NSE (with a CSV-upload
  fallback built into the app itself).
- `forecasting.py` — ARIMA/Prophet fitting, direction-call logic, and the
  walk-forward backtest.
- `requirements.txt` — Python dependencies.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional: to enable the Prophet model as well as ARIMA, also run
`pip install prophet` (uncomment the line in `requirements.txt` if you want
it pinned). Prophet has heavier native build dependencies, so it's kept
optional — the app works fine with ARIMA alone.

## Run

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`. In the sidebar:

1. **Data** — either "Fetch from NSE" with a date range, or "Upload CSV" (see
   below for why you might need this).
2. **Model** — ARIMA (always available) or Prophet (if installed).
3. **Forecast horizon** — how many trading days ahead to call.
4. **Backtest windows** — how many past calls to re-check for accuracy.

Click **Run**.

## Important: why the NSE fetch might fail

NSE (`nseindia.com`) actively blocks a lot of scripted and datacenter
traffic — this is a known, longstanding limitation of scraping their site,
not a bug in this app. In practice:

- It's most likely to work when you run the app **locally from an Indian IP**
  (which matches how NSE expects traffic to look).
- It's most likely to **fail** if you deploy this to Streamlit Community
  Cloud or another cloud host — NSE frequently blocks the IP ranges those
  platforms use.
- The app tries two official NSE sources in order (the historical-data JSON
  API, then the daily Bhavcopy archive), but if both are blocked, don't fight
  it — use the manual path below instead.

**Manual fallback:** go to the NSE quote page you started from
(`nseindia.com/get-quote/equity/RELIANCE`), open the **"Historical Data"**
panel, pick your date range, and download the CSV. Upload that file in the
app's sidebar ("Upload CSV"). The app auto-detects NSE's usual column names,
but if it complains, make sure the file has columns for `Date`, `Open`,
`High`, `Low`, `Close`, and `Volume`.

## Notes on the model

- ARIMA is fit on log-closing-price (to stabilize variance), with the
  `(p, d, q)` order chosen automatically by AIC over a small grid — no manual
  tuning needed, but it also means it's not the highest-ceiling model
  available; it's a solid, fast, interpretable baseline.
- The directional call is derived from the price forecast: if the model's
  forecasted close at the target date is above the last actual close, the
  call is UP, otherwise DOWN. The 80% interval band shown around the
  forecast is a useful sanity check — a call inside a wide, straddling band
  is a weak call even if the point forecast leans one way.
- The backtest is walk-forward (never trains on future data) and compares
  the model's accuracy to a naive "always predict the majority direction"
  baseline. If the model doesn't beat that baseline over your backtest
  window, its calls aren't adding information — that's a useful, honest
  signal, not a bug to hide.
- This is a statistical model of past price behavior, not investment advice.

## Extending it

- Add more features (volume, RSI, moving averages) by switching from a pure
  price time-series model to a classifier (e.g. logistic regression /
  gradient boosting on lagged features) trained directly on the UP/DOWN
  label — `forecasting.py` is structured so a new model just needs a
  `fit_forecast_<name>()` function returning a `ForecastResult`.
- Widen the backtest by lowering `min_train_size` in
  `walk_forward_backtest()`, at the cost of noisier early windows.
- Cache NSE responses to disk in `nse_data.py` if you're iterating a lot and
  want to avoid re-hitting NSE on every run.
