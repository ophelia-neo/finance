# ================================
# Macro Trade Charts (BRL/JPY Pitch)
# ================================
# Charts:
# 1) Brazil vs Japan real policy rates (policy - CPI)
# 2) Brazil vs Japan trade balance
# 3) Brazil vs Japan 5Y CDS
# 4) BRL/JPY price with 200-DMA (plus separate RSI figure)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIG: put your local CSV file paths here --------
# Expected format: 2 columns [Date, Value]; any header names are OK (auto-detected).
POLICY_FILES = {
    "Brazil": "/Users/ophelianeo/Downloads/10ybry_b_m.csv",  # Selic (%)
    "Japan":  "/Users/ophelianeo/Downloads/10yjpy_b_m.csv",   # BoJ policy rate (%)
}
CPI_FILES = {
    "Brazil": "/Users/ophelianeo/Downloads/cpimbr_m_m.csv",      # CPI YoY (%)
    "Japan":  "/Users/ophelianeo/Downloads/cpimjp_m_m.csv",       # CPI YoY (%)
}
TRADE_FILES = {
    "Brazil": "/path/to/brazil_trade_balance_usd_bn.csv",  # USD bn (monthly)
    "Japan":  "/path/to/japan_trade_balance_usd_bn.csv",   # USD bn (monthly)
}
CDS_FILES = {
    "Brazil": "/path/to/brazil_5y_cds_bps.csv",   # bps or %
    "Japan":  "/path/to/japan_5y_cds_bps.csv",    # bps or %
}

START = "2019-01-01"   # adjust as needed
END   = None           # or e.g. "2025-08-17"

# -------- robust CSV loader (auto-detects date/value cols; normalizes units) --------
def load_series_csv(path: str, name: str, to_percent: bool | None = None,
                    freq: str | None = None) -> pd.Series:
    """
    Read a 2+ column CSV, auto-detect date + numeric value column.
    - Normalizes units: if max > 50 and to_percent is True → divide by 100 (bps -> %).
    - If to_percent is False and max < 1 → multiply by 100 (decimals -> %).
    - If freq is 'M' or 'D', resamples to that frequency with last observation.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    if df.empty:
        raise ValueError(f"{name}: CSV empty: {path}")

    # detect date column
    date_scores, parsed = {}, {}
    for col in df.columns:
        try:
            dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            date_scores[col] = dt.notna().mean()
            parsed[col] = dt
        except Exception:
            date_scores[col] = 0
    date_col = max(date_scores, key=date_scores.get)
    dt = parsed[date_col]

    # detect numeric column
    num_scores, nums = {}, {}
    for col in df.columns:
        if col == date_col:
            continue
        raw = df[col].astype(str).str.strip()
        cleaned = (raw.str.replace("%", "", regex=False)
                        .str.replace("\u2212", "-", regex=False)
                        .str.replace("−", "-", regex=False)
                        .str.replace(",", "", regex=False))
        num = pd.to_numeric(cleaned, errors="coerce")
        num_scores[col] = num.notna().mean()
        nums[col] = num
    if not num_scores:
        raise ValueError(f"{name}: no numeric column found in {path}")
    val_col = max(num_scores, key=num_scores.get)
    s = pd.Series(nums[val_col].values, index=dt, name=name).dropna().sort_index()

    # normalize units if asked
    if to_percent is True:
        # if looks like bps (e.g., 235), convert to %; if already %, leave as is
        if s.max() > 50:
            s = s / 100.0
    elif to_percent is False:
        # if decimals (0.023) → to % for readability
        if s.max() < 1:
            s = s * 100.0

    # resample
    if freq == "M":
        s = s.resample("M").last()
    elif freq == "D":
        s = s.resample("D").last()

    # date window
    if START:
        s = s[s.index >= pd.to_datetime(START)]
    if END:
        s = s[s.index <= pd.to_datetime(END)]

    return s.dropna()

# -------- 1) Real policy rates: Brazil vs Japan --------
def chart_real_policy_rates():
    br_policy = load_series_csv(POLICY_FILES["Brazil"], "Brazil Policy (%)", to_percent=None, freq="M")
    jp_policy = load_series_csv(POLICY_FILES["Japan"],  "Japan Policy (%)",  to_percent=None, freq="M")
    br_cpi = load_series_csv(CPI_FILES["Brazil"], "Brazil CPI YoY (%)", to_percent=None, freq="M")
    jp_cpi = load_series_csv(CPI_FILES["Japan"],  "Japan CPI YoY (%)",  to_percent=None, freq="M")

    # align monthly
    df = pd.concat([br_policy, br_cpi, jp_policy, jp_cpi], axis=1).dropna()
    df.columns = ["BR_Policy", "BR_CPI", "JP_Policy", "JP_CPI"]
    df["BR_Real"] = df["BR_Policy"] - df["BR_CPI"]
    df["JP_Real"] = df["JP_Policy"] - df["JP_CPI"]

    plt.figure(figsize=(10, 5))
    df[["BR_Real", "JP_Real"]].plot()
    plt.title("Real Policy Rates: Brazil vs Japan")
    plt.ylabel("Percent")
    plt.tight_layout()
    plt.show()

# -------- 2) Trade balance: Brazil vs Japan --------
def chart_trade_balance():
    br_tb = load_series_csv(TRADE_FILES["Brazil"], "Brazil Trade Balance (USD bn)", to_percent=None, freq="M")
    jp_tb = load_series_csv(TRADE_FILES["Japan"],  "Japan Trade Balance (USD bn)",  to_percent=None, freq="M")

    df = pd.concat([br_tb, jp_tb], axis=1).dropna()
    df.columns = ["Brazil", "Japan"]

    plt.figure(figsize=(10, 5))
    df.plot()
    plt.title("Trade Balance (USD bn): Brazil vs Japan")
    plt.ylabel("USD bn (monthly or 12m rolling if pre-rolled)")
    plt.tight_layout()
    plt.show()

# -------- 3) 5Y CDS: Brazil vs Japan --------
def chart_cds():
    br_cds = load_series_csv(CDS_FILES["Brazil"], "Brazil 5Y CDS", to_percent=True, freq="D")
    jp_cds = load_series_csv(CDS_FILES["Japan"],  "Japan 5Y CDS",  to_percent=True, freq="D")
    # Now both in % (e.g., 1.35% = 135 bps)
    df = pd.concat([br_cds, jp_cds], axis=1).dropna()
    df.columns = ["Brazil 5Y CDS (%)", "Japan 5Y CDS (%)"]

    plt.figure(figsize=(10, 5))
    df.plot()
    plt.title("5Y Sovereign CDS: Brazil vs Japan")
    plt.ylabel("Percent (≈ bps / 100)")
    plt.tight_layout()
    plt.show()

# -------- 4) BRL/JPY with 200-DMA and RSI (two separate figures) --------
def chart_brljpy_with_tech(period="2y"):
    try:
        import yfinance as yf
        have_yf = True
    except Exception:
        have_yf = False

    if not have_yf:
        print("[INFO] yfinance not installed; skip BRL/JPY download.")
        return

    px = yf.download("BRLJPY=X", period=period, auto_adjust=True, progress=False)
    if px.empty:
        print("[INFO] No BRLJPY=X data returned.")
        return

    close = px["Close"].dropna()
    ma200 = close.rolling(200).mean()

    # --- Figure A: Price (candles if mplfinance available) + 200-DMA ---
    try:
        import mplfinance as mpf
        df = px.copy()
        df = df[["Open","High","Low","Close"]].dropna()
        mpf.plot(df, type="candle", mav=(200,), volume=False, title="BRL/JPY: Candles with 200-DMA", style="yahoo")
    except Exception:
        # fallback: simple line + 200-DMA
        plt.figure(figsize=(10, 5))
        close.plot()
        ma200.plot()
        plt.title("BRL/JPY: Close with 200-DMA")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.show()

    # --- Figure B: RSI(14) ---
    ret = close.pct_change()
    gain = ret.where(ret > 0, 0).rolling(14).mean()
    loss = (-ret.where(ret < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    plt.figure(figsize=(10, 3.5))
    rsi.plot()
    plt.axhline(70); plt.axhline(50); plt.axhline(30)
    plt.title("BRL/JPY RSI(14)")
    plt.ylabel("RSI")
    plt.tight_layout()
    plt.show()

# ---- run what you want ----
# Uncomment the lines below after you fill the CSV paths.

chart_real_policy_rates()
chart_trade_balance()
chart_cds()
chart_brljpy_with_tech(period="3y")
