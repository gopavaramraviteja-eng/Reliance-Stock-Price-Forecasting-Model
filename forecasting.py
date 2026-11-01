"""
Forecasting models for RELIANCE closing price, plus the logic that turns a
price forecast into an up/down direction call.

Two classical time-series models are supported:
- ARIMA (statsmodels) -- always available, order picked automatically by a
  small AIC grid search over a handful of (p, d, q) combinations.
- Prophet (Meta's forecasting library) -- optional; the app checks
  `PROPHET_AVAILABLE` and hides the option if it isn't installed, since
  Prophet has heavier native dependencies than the rest of this app.

Both models are wrapped behind the same interface so the app can treat them
interchangeably:
    forecast(df, horizon) -> DataFrame[Date, yhat, yhat_lower, yhat_upper]
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from prophet import Prophet  # type: ignore

    PROPHET_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional install
    PROPHET_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

ARIMA_ORDER_GRID = [
    (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (0, 1, 2), (2, 1, 1), (1, 1, 2),
]


@dataclass
class ForecastResult:
    dates: pd.Series
    yhat: np.ndarray
    yhat_lower: np.ndarray
    yhat_upper: np.ndarray
    model_label: str


def _business_days_after(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n)


def fit_forecast_arima(close: pd.Series, dates: pd.Series, horizon: int) -> ForecastResult:
    """Fit ARIMA on log-close (stabilises variance) and forecast `horizon` steps ahead."""
    log_close = np.log(close.astype(float).values)

    best_aic = np.inf
    best_fit = None
    best_order = None
    for order in ARIMA_ORDER_GRID:
        try:
            model = ARIMA(log_close, order=order)
            fit = model.fit()
            if fit.aic < best_aic:
                best_aic, best_fit, best_order = fit.aic, fit, order
        except Exception:
            continue

    if best_fit is None:
        # Last-resort simple order if the grid search failed entirely (e.g. very short series)
        best_fit = ARIMA(log_close, order=(1, 1, 0)).fit()
        best_order = (1, 1, 0)

    fc = best_fit.get_forecast(steps=horizon)
    mean_log = fc.predicted_mean
    ci_log = fc.conf_int(alpha=0.2)  # 80% interval

    yhat = np.exp(mean_log)
    yhat_lower = np.exp(ci_log[:, 0])
    yhat_upper = np.exp(ci_log[:, 1])

    future_dates = _business_days_after(pd.Timestamp(dates.iloc[-1]), horizon)
    return ForecastResult(
        dates=pd.Series(future_dates),
        yhat=yhat,
        yhat_lower=yhat_lower,
        yhat_upper=yhat_upper,
        model_label=f"ARIMA{best_order}",
    )


def fit_forecast_prophet(close: pd.Series, dates: pd.Series, horizon: int) -> ForecastResult:
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed in this environment.")
    train = pd.DataFrame({"ds": pd.to_datetime(dates.values), "y": close.astype(float).values})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
                     interval_width=0.8)
    model.fit(train)
    future_dates = _business_days_after(pd.Timestamp(dates.iloc[-1]), horizon)
    future = pd.DataFrame({"ds": future_dates})
    fc = model.predict(future)
    return ForecastResult(
        dates=pd.Series(future_dates),
        yhat=fc["yhat"].values,
        yhat_lower=fc["yhat_lower"].values,
        yhat_upper=fc["yhat_upper"].values,
        model_label="Prophet",
    )


def forecast(model_name: str, close: pd.Series, dates: pd.Series, horizon: int) -> ForecastResult:
    if model_name == "ARIMA":
        return fit_forecast_arima(close, dates, horizon)
    elif model_name == "Prophet":
        return fit_forecast_prophet(close, dates, horizon)
    raise ValueError(f"Unknown model: {model_name}")


def direction_call(last_actual_close: float, forecast_result: ForecastResult) -> dict:
    """Turn a price forecast into an UP/DOWN call for the final forecasted day."""
    target = float(forecast_result.yhat[-1])
    pct_move = (target - last_actual_close) / last_actual_close * 100
    return {
        "direction": "UP" if target >= last_actual_close else "DOWN",
        "target_price": target,
        "last_close": last_actual_close,
        "pct_move": pct_move,
        "lower": float(forecast_result.yhat_lower[-1]),
        "upper": float(forecast_result.yhat_upper[-1]),
        "target_date": forecast_result.dates.iloc[-1],
    }


def walk_forward_backtest(
    df: pd.DataFrame,
    model_name: str,
    horizon: int,
    n_windows: int,
    min_train_size: int = 60,
) -> pd.DataFrame:
    """
    Walk-forward validation of the directional call:
    for each of the last `n_windows` points where a `horizon`-day-ahead
    actual is known, train on everything before it and check whether the
    model's UP/DOWN call for that date matched what actually happened.

    Returns a DataFrame with one row per test point: date, predicted vs
    actual direction, predicted vs actual price, and a `correct` flag.
    """
    dates = df["Date"].reset_index(drop=True)
    close = df["Close"].reset_index(drop=True)
    n = len(df)

    last_possible_train_end = n - horizon - 1
    first_train_end = max(min_train_size, last_possible_train_end - n_windows + 1)

    records = []
    for train_end in range(first_train_end, last_possible_train_end + 1):
        train_close = close.iloc[: train_end + 1]
        train_dates = dates.iloc[: train_end + 1]
        actual_idx = train_end + horizon
        if actual_idx >= n:
            continue
        try:
            res = forecast(model_name, train_close, train_dates, horizon)
        except Exception:
            continue
        last_close = float(train_close.iloc[-1])
        call = direction_call(last_close, res)
        actual_close = float(close.iloc[actual_idx])
        actual_direction = "UP" if actual_close >= last_close else "DOWN"
        records.append({
            "as_of_date": train_dates.iloc[-1],
            "target_date": dates.iloc[actual_idx],
            "predicted_direction": call["direction"],
            "actual_direction": actual_direction,
            "predicted_price": call["target_price"],
            "actual_price": actual_close,
            "correct": call["direction"] == actual_direction,
        })
    return pd.DataFrame(records)
