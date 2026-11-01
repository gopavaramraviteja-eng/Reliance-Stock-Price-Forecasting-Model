"""
RELIANCE (Reliance Industries) direction-forecasting app.

Run locally with:
    streamlit run app.py

What it does:
- Loads historical daily OHLCV data for RELIANCE (NSE, series EQ) either by
  fetching it from NSE directly, or from a CSV you upload/download yourself
  from the NSE quote page's "Historical Data" panel.
- Fits a classical time-series model (ARIMA, or Prophet if installed) on the
  closing price.
- Reports a directional call (UP/DOWN) for N trading days ahead, with a
  price target and confidence band.
- Backtests that directional call walk-forward over recent history so you
  can see how reliable it has actually been, compared to a naive baseline.

See README.md for setup notes, especially why the NSE fetch can fail (NSE
blocks a lot of scripted/datacenter traffic) and what to do about it.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import PROPHET_AVAILABLE, direction_call, forecast, walk_forward_backtest
from nse_data import NSEFetchError, REQUIRED_COLUMNS, get_history, validate_ohlcv

st.set_page_config(page_title="RELIANCE Direction Forecast", layout="wide")

st.title("Reliance Industries (RELIANCE) — Direction Forecast")
st.caption(
    "Classical time-series forecasting on NSE closing-price history, "
    "turned into an UP / DOWN call with a walk-forward backtest."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data")
    data_mode = st.radio(
        "Source",
        ["Fetch from NSE", "Upload CSV"],
        help=(
            "NSE frequently blocks automated requests from cloud/datacenter "
            "IPs. If the fetch fails, download history yourself from the "
            "'Historical Data' section on the NSE RELIANCE quote page and "
            "upload it here instead."
        ),
    )

    today = dt.date.today()
    if data_mode == "Fetch from NSE":
        start_date = st.date_input("Start date", value=today - dt.timedelta(days=365 * 2))
        end_date = st.date_input("End date", value=today)
    else:
        uploaded = st.file_uploader(
            "OHLCV CSV (columns: Date, Open, High, Low, Close, Volume)", type="csv"
        )

    st.header("2. Model")
    model_options = ["ARIMA"] + (["Prophet"] if PROPHET_AVAILABLE else [])
    if not PROPHET_AVAILABLE:
        st.caption("Prophet not installed — see README.md to enable it. Using ARIMA only.")
    model_name = st.selectbox("Model", model_options)

    horizon = st.slider("Forecast horizon (trading days ahead)", 1, 20, 5)

    st.header("3. Backtest")
    n_windows = st.slider("Backtest windows (how many past calls to re-check)", 5, 60, 20)

    run = st.button("Run", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame | None, str | None]:
    if data_mode == "Fetch from NSE":
        try:
            with st.spinner("Fetching history from NSE..."):
                df, source = get_history(start_date, end_date)
            return df, source
        except NSEFetchError as e:
            st.error(
                f"Could not fetch data from NSE: {e}\n\n"
                "This is a known NSE limitation (it blocks a lot of scripted "
                "and datacenter traffic, especially from cloud-hosted apps). "
                "Switch 'Source' to 'Upload CSV' in the sidebar and use the "
                "'Historical Data' download on the NSE RELIANCE quote page "
                "instead."
            )
            return None, None
        except Exception as e:
            st.error(f"Unexpected error fetching NSE data: {e}")
            return None, None
    else:
        if uploaded is None:
            st.info("Upload a CSV in the sidebar to continue.")
            return None, None
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read that CSV: {e}")
            return None, None
        # tolerate common NSE export column names
        df.columns = [c.strip() for c in df.columns]
        rename_map = {
            "Date ": "Date", "OPEN": "Open", "HIGH": "High", "LOW": "Low",
            "close": "Close", "CLOSE": "Close", "ltp": "Close", "LTP": "Close",
            "VOLUME": "Volume", "Shares Traded": "Volume", "No. of Shares": "Volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        problems = validate_ohlcv(df)
        if problems:
            st.error("Uploaded CSV has problems:\n\n" + "\n".join(f"- {p}" for p in problems))
            st.caption(f"Required columns: {', '.join(REQUIRED_COLUMNS)}")
            return None, None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
        return df, "Uploaded CSV"


if not run:
    st.info("Set your options in the sidebar and click **Run**.")
    st.stop()

df, source = load_data()
if df is None or df.empty:
    st.stop()

if len(df) < 60:
    st.warning(
        f"Only {len(df)} rows of history loaded — forecasts and the backtest "
        "will be unreliable with this little data. Widen the date range or "
        "upload a longer history."
    )

st.success(f"Loaded {len(df)} trading days from **{source}** "
           f"({df['Date'].min().date()} → {df['Date'].max().date()}).")

# ---------------------------------------------------------------------------
# Price chart
# ---------------------------------------------------------------------------
st.subheader("Price history")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                   xaxis_title=None, yaxis_title="₹")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Forecast + direction call
# ---------------------------------------------------------------------------
st.subheader(f"{horizon}-trading-day forecast ({model_name})")

try:
    with st.spinner(f"Fitting {model_name}..."):
        res = forecast(model_name, df["Close"], df["Date"], horizon)
except Exception as e:
    st.error(f"Model fitting failed: {e}")
    st.stop()

last_close = float(df["Close"].iloc[-1])
call = direction_call(last_close, res)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last close", f"₹{last_close:,.2f}", help=f"as of {df['Date'].iloc[-1].date()}")
col2.metric(
    f"Predicted close ({call['target_date'].date()})",
    f"₹{call['target_price']:,.2f}",
    f"{call['pct_move']:+.2f}%",
)
col3.metric("Direction call", call["direction"])
col4.metric("80% interval", f"₹{call['lower']:,.0f} – ₹{call['upper']:,.0f}")

fig2 = go.Figure()
hist_tail = df.tail(60)
fig2.add_trace(go.Scatter(x=hist_tail["Date"], y=hist_tail["Close"], mode="lines", name="History"))
fig2.add_trace(go.Scatter(x=res.dates, y=res.yhat, mode="lines+markers",
                           name=f"{res.model_label} forecast", line=dict(dash="dash")))
fig2.add_trace(go.Scatter(
    x=list(res.dates) + list(res.dates[::-1]),
    y=list(res.yhat_upper) + list(res.yhat_lower[::-1]),
    fill="toself", fillcolor="rgba(99,110,250,0.15)", line=dict(width=0),
    name="80% interval", showlegend=True,
))
fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="₹")
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "This is a statistical extrapolation of past price behaviour, not "
    "investment advice. Treat the direction call as one input among many."
)

# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------
st.subheader("Walk-forward backtest of the directional call")
st.caption(
    f"Re-runs the same {horizon}-day-ahead {model_name} call at {n_windows} "
    "past points in history (never looking ahead) and checks whether the "
    "UP/DOWN call matched what actually happened next."
)

with st.spinner("Backtesting — this refits the model once per window..."):
    bt = walk_forward_backtest(df, model_name, horizon, n_windows)

if bt.empty:
    st.warning("Not enough history to run a backtest at this horizon / window count.")
else:
    accuracy = bt["correct"].mean()
    naive_baseline = (bt["actual_direction"] == "UP").mean()
    naive_baseline = max(naive_baseline, 1 - naive_baseline)  # always-predict-majority-class

    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Model directional accuracy", f"{accuracy:.0%}", f"n={len(bt)}")
    bcol2.metric("Naive 'always predict majority' baseline", f"{naive_baseline:.0%}")
    bcol3.metric("Edge over baseline", f"{(accuracy - naive_baseline)*100:+.0f} pts")

    if accuracy <= naive_baseline:
        st.warning(
            "The model isn't beating the trivial baseline of always guessing "
            "the more common direction over this backtest window — treat its "
            "directional calls with caution, or try a different horizon / "
            "date range."
        )

    st.dataframe(
        bt.assign(
            as_of_date=bt["as_of_date"].dt.date,
            target_date=bt["target_date"].dt.date,
            predicted_price=bt["predicted_price"].round(2),
            actual_price=bt["actual_price"].round(2),
        ),
        use_container_width=True,
        hide_index=True,
    )
