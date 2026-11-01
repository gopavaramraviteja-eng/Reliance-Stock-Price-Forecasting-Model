"""
Data access layer for NSE Reliance Industries (RELIANCE) historical data.

Design goals:
- Prefer NSE's own official sources (no third-party price vendors).
- Degrade gracefully: NSE actively blocks scripted/datacenter traffic, so every
  automated path can fail. The Streamlit app always offers a manual CSV upload
  as a last resort (see app.py), so this module never needs to be the only
  way to get data into the app.
- Cache whatever is fetched to disk so re-running the app doesn't re-hit NSE.

Two automated strategies, tried in order:
1. fetch_via_api      -> NSE's historical-data JSON API (fast, but the first
                          to get blocked from cloud/datacenter IPs).
2. fetch_via_bhavcopy -> NSE's full daily Bhavcopy archive, reconstructed
                          day-by-day. Slower (one HTTP request per trading
                          day) but a different endpoint family, so it
                          sometimes works when the API path doesn't.
"""
from __future__ import annotations

import datetime as dt
import io
import time

import pandas as pd
import requests

SYMBOL = "RELIANCE"
SERIES = "EQ"

BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


class NSEFetchError(RuntimeError):
    """Raised when an NSE data source can't be reached or returns nothing usable."""


def _new_session() -> requests.Session:
    """A requests.Session primed with cookies the way a real browser would get them.

    NSE's APIs reject requests that don't carry cookies set by first visiting
    the site's HTML pages, so we hit the homepage and the stock's quote page
    before calling the JSON endpoints.
    """
    s = requests.Session()
    s.headers.update(BASE_HEADERS)
    s.get("https://www.nseindia.com", timeout=10)
    s.get(f"https://www.nseindia.com/get-quote/equity/{SYMBOL}", timeout=10)
    return s


def _normalise_api_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "CH_TIMESTAMP": "Date",
        "CH_OPENING_PRICE": "Open",
        "CH_TRADE_HIGH_PRICE": "High",
        "CH_TRADE_LOW_PRICE": "Low",
        "CH_CLOSING_PRICE": "Close",
        "CH_TOT_TRADED_QTY": "Volume",
    }
    df = df.rename(columns=rename)
    keep = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    return df


def fetch_via_api(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Pull daily OHLCV history from NSE's own historical-data API.

    NSE caps each call to roughly a year of data, so multi-year ranges are
    fetched in yearly chunks and concatenated.
    """
    s = _new_session()
    url = "https://www.nseindia.com/api/historical/cm/equity"
    frames = []
    cur = start
    while cur <= end:
        try:
            chunk_end = min(end, cur.replace(year=cur.year + 1) - dt.timedelta(days=1))
        except ValueError:
            # cur is Feb 29 on a leap year; step back a day before adding a year
            chunk_end = min(end, (cur - dt.timedelta(days=1)).replace(year=cur.year + 1))
        params = {
            "symbol": SYMBOL,
            "series": '["EQ"]',
            "from": cur.strftime("%d-%m-%Y"),
            "to": chunk_end.strftime("%d-%m-%Y"),
        }
        r = s.get(url, params=params, timeout=15)
        if r.status_code != 200:
            raise NSEFetchError(f"NSE API returned HTTP {r.status_code}")
        try:
            payload = r.json().get("data", [])
        except ValueError as e:
            raise NSEFetchError("NSE API returned a non-JSON response (likely blocked)") from e
        if payload:
            frames.append(pd.DataFrame(payload))
        cur = chunk_end + dt.timedelta(days=1)
        time.sleep(0.4)  # be polite between chunk requests
    if not frames:
        raise NSEFetchError("NSE API returned no rows for the requested range")
    return _normalise_api_frame(pd.concat(frames, ignore_index=True))


def fetch_via_bhavcopy(start: dt.date, end: dt.date, max_days: int = 400) -> pd.DataFrame:
    """Fallback: reconstruct history day-by-day from NSE's full Bhavcopy archive.

    One HTTP request per calendar day in range, so this is capped at
    `max_days` to keep runtime sane -- use the API path or a manual CSV for
    longer backtests.
    """
    span = (end - start).days
    if span > max_days:
        raise NSEFetchError(
            f"Requested range is {span} days; Bhavcopy fallback is capped at "
            f"{max_days} days to avoid thousands of requests. Narrow the date "
            f"range or upload a CSV instead."
        )
    s = requests.Session()
    s.headers.update(BASE_HEADERS)
    rows = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # trading days only; holidays 404 and are skipped
            fname = cur.strftime("%d%m%Y")
            url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{fname}.csv"
            try:
                r = s.get(url, timeout=15)
                if r.status_code == 200 and r.text.strip():
                    day_df = pd.read_csv(io.StringIO(r.text))
                    day_df.columns = [c.strip() for c in day_df.columns]
                    day_df = day_df[
                        (day_df["SYMBOL"].astype(str).str.strip() == SYMBOL)
                        & (day_df["SERIES"].astype(str).str.strip() == SERIES)
                    ]
                    if not day_df.empty:
                        rows.append(day_df.iloc[0])
            except requests.RequestException:
                pass
            time.sleep(0.25)
        cur += dt.timedelta(days=1)
    if not rows:
        raise NSEFetchError("Bhavcopy fallback returned no rows for the requested range")
    df = pd.DataFrame(rows)
    rename = {
        "DATE1": "Date",
        "OPEN_PRICE": "Open",
        "HIGH_PRICE": "High",
        "LOW_PRICE": "Low",
        "CLOSE_PRICE": "Close",
        "TTL_TRD_QNTY": "Volume",
    }
    df = df.rename(columns=rename)
    keep = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return df


def get_history(start: dt.date, end: dt.date) -> tuple[pd.DataFrame, str]:
    """Try the fast API first, then fall back to Bhavcopy reconstruction.

    Returns (dataframe, source_label). Raises NSEFetchError if both fail --
    the caller (the Streamlit app) should catch this and offer CSV upload.
    """
    try:
        df = fetch_via_api(start, end)
        if len(df) > 5:
            return df, "NSE historical API (nseindia.com)"
    except Exception:
        pass
    df = fetch_via_bhavcopy(start, end)
    return df, "NSE Bhavcopy full archive (nsearchives.nseindia.com)"


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with a user-supplied OHLCV dataframe (empty = OK)."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"Missing required column(s): {', '.join(missing)}")
        return problems
    if len(df) < 30:
        problems.append("Need at least ~30 rows of history for a meaningful model.")
    if df["Close"].isna().all():
        problems.append("'Close' column has no usable numeric values.")
    return problems
