import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Download BRL/JPY (3y daily) ---
px = yf.download("BRLJPY=X", period="3y", interval="1d",
                 auto_adjust=False, progress=False)  # keep raw OHLC for candles
if px.empty:
    raise SystemExit("No data for BRLJPY=X")

close = px["Close"].dropna()
ma200 = close.rolling(window=200, min_periods=200).mean()

# --- Figure A: Candlesticks with 200-DMA (or fallback line chart) ---
try:
    import mplfinance as mpf
    df_mpf = px[["Open", "High", "Low", "Close"]].dropna()
    ap = [mpf.make_addplot(ma200, width=1.4)]  # labeled automatically as 'Close' MA
    mpf.plot(df_mpf, type="candle", addplot=ap, mav=None,
             volume=False, title="BRL/JPY — Candles with 200-DMA", style="yahoo")
except Exception as e:
    print(f"[info] mplfinance not available → fallback line plot ({e})")
    fig, ax = plt.subplots(figsize=(10,5))
    close.plot(ax=ax, label="BRL/JPY Exchange Rate", linewidth=1.4)
    ma200.plot(ax=ax, label="200-Day Moving Average", linewidth=1.8)

    ax.set_title("BRL/JPY — Close with 200-Day Moving Average")
    ax.set_ylabel("Price"); ax.legend(["BRL/JPY Exchange Rate", "200-Day Moving Average"]); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.show()

# --- Figure B: RSI(14), Wilder smoothing ---
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
roll = 14
avg_gain = gain.ewm(alpha=1/roll, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/roll, adjust=False).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

fig, ax = plt.subplots(figsize=(10,3.5))
rsi.plot(ax=ax, label="RSI(14)")
ax.axhline(70, linestyle="--", linewidth=1)
ax.axhline(50, linestyle="--", linewidth=1)
ax.axhline(30, linestyle="--", linewidth=1)
ax.set_title("BRL/JPY — RSI(14)")
ax.set_ylabel("RSI"); ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()
