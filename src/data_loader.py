"""Download, standardize, and persist market OHLCV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _standardize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, date-sorted dataframe with standard OHLCV columns."""
    if data.empty:
        raise ValueError("Market data is empty.")

    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        price_level = next(
            (
                level
                for level in range(df.columns.nlevels)
                if {"Open", "High", "Low", "Close", "Volume"}.issubset(
                    set(df.columns.get_level_values(level))
                )
            ),
            None,
        )
        if price_level is None:
            raise ValueError("Could not identify OHLCV fields in MultiIndex columns.")
        df.columns = df.columns.get_level_values(price_level)

    if not isinstance(df.index, pd.RangeIndex):
        index_name = df.index.name or "date"
        df = df.reset_index().rename(columns={index_name: "date"})

    df.columns = [
        str(column).strip().lower().replace(" ", "_") for column in df.columns
    ]
    if "date" not in df.columns:
        for candidate in ("datetime", "index"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break

    if "close" not in df.columns and "adj_close" in df.columns:
        df = df.rename(columns={"adj_close": "close"})

    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_localize(None)
    for column in REQUIRED_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return (
        df.dropna(subset=REQUIRED_COLUMNS)
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def download_data(
    symbol: str,
    start_date: str,
    end_date: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted OHLCV data from Yahoo Finance."""
    if not symbol.strip():
        raise ValueError("symbol must be a non-empty string.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to download market data.") from exc

    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    return _standardize_ohlcv(data)


def download_recent_data(
    symbol: str,
    period: str = "5d",
    interval: str = "5m",
    include_prepost: bool = True,
) -> pd.DataFrame:
    """Download recent intraday OHLCV data for a lightweight live snapshot."""
    if not symbol.strip():
        raise ValueError("symbol must be a non-empty string.")
    if not period.strip() or not interval.strip():
        raise ValueError("period and interval must be non-empty strings.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to download market data.") from exc

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        actions=False,
        prepost=include_prepost,
        threads=False,
    )
    return _standardize_ohlcv(data)


def summarize_live_market_data(
    data: pd.DataFrame,
    symbol: str,
    interval: str = "5m",
    stale_after_minutes: float = 15,
    fetched_at: pd.Timestamp | str | None = None,
) -> dict:
    """Summarize the latest available intraday candle without implying a forecast."""
    if stale_after_minutes <= 0:
        raise ValueError("stale_after_minutes must be positive.")
    clean = _standardize_ohlcv(data)
    latest = clean.iloc[-1]
    latest_timestamp = pd.Timestamp(latest["date"])
    resolved_fetched_at = (
        pd.Timestamp(fetched_at)
        if fetched_at is not None
        else pd.Timestamp.now(tz="UTC").tz_localize(None)
    )
    if resolved_fetched_at.tz is not None:
        resolved_fetched_at = resolved_fetched_at.tz_convert("UTC").tz_localize(None)
    session = clean.loc[clean["date"].dt.normalize() == latest_timestamp.normalize()]
    session_open = float(session.iloc[0]["open"])
    current_price = float(latest["close"])
    age_minutes = max(
        0.0,
        float((resolved_fetched_at - latest_timestamp).total_seconds() / 60),
    )
    return {
        "symbol": symbol,
        "interval": interval,
        "timestamp": latest_timestamp,
        "fetched_at": resolved_fetched_at,
        "current_price": current_price,
        "session_open": session_open,
        "session_high": float(session["high"].max()),
        "session_low": float(session["low"].min()),
        "session_volume": float(session["volume"].sum()),
        "intraday_change": (
            float(current_price / session_open - 1) if session_open > 0 else 0.0
        ),
        "data_age_minutes": age_minutes,
        "is_stale": age_minutes > stale_after_minutes,
        "source": "Yahoo Finance intraday",
    }


def save_raw_data(
    df: pd.DataFrame, symbol: str, folder: str | Path = "data/raw"
) -> str:
    """Standardize and save raw OHLCV data to a CSV file."""
    return _save_data(df, symbol, folder)


def load_raw_data(symbol: str, folder: str | Path = "data/raw") -> pd.DataFrame:
    """Load and standardize saved raw OHLCV data."""
    return _load_data(symbol, folder)


def save_processed_data(
    df: pd.DataFrame, symbol: str, folder: str | Path = "data/processed"
) -> str:
    """Save a processed market dataframe to a CSV file."""
    path = get_symbol_path(symbol, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def load_processed_data(
    symbol: str, folder: str | Path = "data/processed"
) -> pd.DataFrame:
    """Load a processed market dataframe and parse its date column."""
    path = get_symbol_path(symbol, folder)
    if not path.exists():
        raise FileNotFoundError(f"Processed data file does not exist: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _save_data(df: pd.DataFrame, symbol: str, folder: str | Path) -> str:
    clean_df = _standardize_ohlcv(df)
    path = get_symbol_path(symbol, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(path, index=False)
    return str(path)


def _load_data(symbol: str, folder: str | Path) -> pd.DataFrame:
    path = get_symbol_path(symbol, folder)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file does not exist: {path}")
    return _standardize_ohlcv(pd.read_csv(path, parse_dates=["date"]))


def get_symbol_path(symbol: str, folder: str | Path) -> Path:
    """Return the safe CSV path used to persist a market symbol."""
    safe_symbol = symbol_slug(symbol)
    return Path(folder) / f"{safe_symbol}.csv"


def symbol_slug(symbol: str) -> str:
    """Return a filesystem-safe, uppercase market symbol."""
    safe_symbol = (
        symbol.strip()
        .upper()
        .replace("/", "_")
        .replace("=", "_")
        .replace("^", "")
    )
    if not safe_symbol:
        raise ValueError("symbol must be a non-empty string.")
    return safe_symbol
