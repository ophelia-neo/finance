# === Macro Visual Pack (1–6) ===
# Creates six figures exactly as requested.
# Requires: yfinance, pandas, numpy, matplotlib
# NOTE: Yahoo doesn’t host many official policy-rate & non-US yield series.
# Fill the CSV paths below (simple two-column files) to plot the real series.
# Otherwise, the script will plot what it can (e.g., US 10Y, currencies) and
# print a helpful message for anything missing.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

plt.rcParams["figure.figsize"] = (10, 6.5)

# ----------------------------- Helpers -----------------------------

def get_close_series(ticker: str, period="5y"):
    """Return a 1D Close Series for a single Yahoo ticker (robust to MultiIndex)."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"][ticker]
    else:
        s = df["Close"]
    return s.dropna().astype(float).rename(ticker)

def try_any(tickers, period="5y"):
    """Try a list of tickers; return the first Series that works or None."""
    for t in tickers:
        try:
            return get_close_series(t, period=period)
        except Exception:
            continue
    return None

def index_1(df_or_s):
    df = df_or_s.copy()
    return df / df.iloc[0]

def load_rate_csv(path, name, date_col="Date", value_col="Rate"):
    """CSV with columns: Date, Rate (in %, or level)."""
    df = pd.read_csv(path, parse_dates=[date_col])
    return df.set_index(date_col)[value_col].astype(float).rename(name).sort_index()

def load_yield_csv(path, name, date_col="Date", value_col="Yield"):
    """CSV with columns: Date, Yield (in %)."""
    df = pd.read_csv(path, parse_dates=[date_col])
    return df.set_index(date_col)[value_col].astype(float).rename(name).sort_index()

# ---------------------- 1) Policy Rate Comparison ------------------
# Fill paths if you have them; leave as None to skip a line.
POLICY_FILES = {
    # "Fed":    "fed_funds_effective.csv",     # Date,Rate
    # "ECB":    "ecb_deposit_rate.csv",
    # "BoJ":    "boj_policy_rate.csv",
    # "Brazil": "bcb_selic_target.csv",
    # "China":  "pboc_1y_lpr.csv",
}

policy_series = []
for name, path in POLICY_FILES.items():
    try:
        s = load_rate_csv(path, name)
        policy_series.append(s)
    except Exception as e:
        print(f"[Policy] Skipped {name}: {e}")

if policy_series:
    panel = pd.concat(policy_series, axis=1).dropna(how="all")
    panel = panel[panel.index >= "2020-01-01"]
    panel.plot(title="Policy Rates (2020–present)")
    plt.ylabel("%")
    plt.tight_layout(); plt.show()
else:
    print("[Policy] No CSVs provided → Add central-bank CSVs to plot policy rates.")

# -------- 2) 10-Year Government Bond Yields (US, DE, JP, BR, CN) ---
us10 = get_close_series("^TNX", period="5y")/10.0  # ^TNX is in %*10
us10 = us10.rename("US 10Y")

# Fill CSVs for other 10Y yields (or leave to skip)
Y10_FILES = {
    # "Germany 10Y": "de10y.csv",  # Date,Yield
    # "Japan 10Y":   "jp10y.csv",
    # "Brazil 10Y":  "br10y.csv",
    # "China 10Y":   "cn10y.csv",
}

y10_series = [us10]
for name, path in Y10_FILES.items():
    try:
        y10_series.append(load_yield_csv(path, name))
    except Exception as e:
        print(f"[10Y] Skipped {name}: {e}")

panel10 = pd.concat(y10_series, axis=1).dropna(how="all")
panel10.plot(title="10Y Government Bond Yields")
plt.ylabel("%"); plt.tight_layout(); plt.show()

# ------------- 3) Currency Performance (USD vs majors/EM) ----------
usd_proxy = try_any(["DX-Y.NYB", "UUP"], period="3y")  # DXY or UUP ETF
eurusd = get_close_series("EURUSD=X", period="3y")
usdjpy = get_close_series("JPY=X", period="3y")
usdbrl = get_close_series("BRL=X", period="3y")
usdcny = get_close_series("CNY=X", period="3y")

usd_vs = pd.DataFrame({
    "USD (proxy)": usd_proxy,
    "USD/EUR": 1/eurusd,   # invert EUR/USD
    "USD/JPY": usdjpy,
    "USD/BRL": usdbrl,
    "USD/CNY": usdcny
}).dropna()
index_1(usd_vs).plot(title="Currency Performance: USD vs EUR, JPY, BRL, CNY (Indexed)")
plt.ylabel("Index"); plt.tight_layout(); plt.show()

# -------- 4) Yield Curve Steepness (10Y–2Y): US (+ CSV add-ons) ----
us10y = get_close_series("^TNX", period="5y")/10.0
us2y  = try_any(["^UST2Y"], period="5y")
if us2y is not None:
    us2y = (us2y/100.0) if us2y.max() > 20 else us2y
    us_steep = (us10y - us2y).rename("US 10s–2s")
else:
    us5y = get_close_series("^FVX", period="5y")/10.0
    us_steep = (us10y - us5y).rename("US 10s–5s (proxy)")

# Optionally add Germany/Brazil with your CSVs:
CURVE_FILES = {
    # "Germany 2Y": "de2y.csv", "Germany 10Y": "de10y.csv",
    # "Brazil 2Y":  "br2y.csv", "Brazil 10Y":  "br10y.csv",
}
curve_panels = [us_steep]
try:
    if all(k in CURVE_FILES for k in ["Germany 2Y","Germany 10Y"]):
        de2 = load_yield_csv(CURVE_FILES["Germany 2Y"], "DE 2Y")
        de10= load_yield_csv(CURVE_FILES["Germany 10Y"], "DE 10Y")
        curve_panels.append((de10 - de2).rename("Germany 10s–2s"))
    if all(k in CURVE_FILES for k in ["Brazil 2Y","Brazil 10Y"]):
        br2 = load_yield_csv(CURVE_FILES["Brazil 2Y"], "BR 2Y")
        br10= load_yield_csv(CURVE_FILES["Brazil 10Y"], "BR 10Y")
        curve_panels.append((br10 - br2).rename("Brazil 10s–2s"))
except Exception as e:
    print(f"[Curve] Skipped extra curves: {e}")

pd.concat(curve_panels, axis=1).dropna(how="all").plot(title="Yield Curve Steepness (10Y – 2Y)")
plt.ylabel("pp"); plt.tight_layout(); plt.show()

# --------- 5) Volatility Indices (MOVE, VIX, DBCVIX) – proxies -----
# VIX is available; MOVE/DBCVIX are proxied by 30D realized vol.
vix = get_close_series("^VIX", period="5y")
tnx_ret = get_close_series("^TNX", period="5y").pct_change()
usd_ret = (usd_proxy or get_close_series("UUP", period="5y")).pct_change()

def realized_vol(ret, win=30, name="RV"):
    return (ret.rolling(win).std() * np.sqrt(252)).rename(name)

bond_rv = realized_vol(tnx_ret, 30, "US10Y RV (30D)")
fx_rv   = realized_vol(usd_ret, 30, "USD Proxy RV (30D)")

pd.concat([vix.rename("VIX"), bond_rv, fx_rv], axis=1).dropna().plot(
    title="Volatility: VIX vs Bond RV vs USD RV"
)
plt.ylabel("Level / Ann. Vol"); plt.tight_layout(); plt.show()

# ---- 6) FX Carry Trade Returns (spot/cumulative price proxies) -----
# BRL/JPY (Yahoo cross if available; else build from legs). Plus USD/JPY, USD/CNY.
brl_usd_3y = 1/get_close_series("BRL=X", period="3y")
jpy_usd_3y = get_close_series("JPY=X", period="3y")  # USD/JPY
brl_jpy_cross = (brl_usd_3y / (1/jpy_usd_3y)).rename("BRL/JPY (cross)")
brl_jpy_yahoo = try_any(["BRLJPY=X"], period="3y")

carry_df = pd.concat([brl_jpy_cross, brl_jpy_yahoo], axis=1).dropna(how="all")
index_1(carry_df).plot(title="Carry Trade Price Proxies (BRL/JPY)")
plt.ylabel("Index (spot only)"); plt.tight_layout(); plt.show()

index_1(jpy_usd_3y.to_frame(name="USD/JPY")).plot(title="USD/JPY (Indexed)")
plt.ylabel("Index"); plt.tight_layout(); plt.show()

index_1(get_close_series("CNY=X", period="3y").to_frame(name="USD/CNY")).plot(
    title="USD/CNY (Indexed)"
)
plt.ylabel("Index"); plt.tight_layout(); plt.show()
