import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def get_close_series(ticker, start="2020-01-01", end=None, auto_adjust=True):
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    # Handle both single-index and MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer exact ('Close', ticker) if present
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            # Fallback: take the 'Close' level and the first column
            close_block = df.xs("Close", axis=1, level=0)
            s = close_block.iloc[:, 0]
    else:
        s = df["Close"]
    s = s.dropna()
    s.name = ticker  # set the series name without using rename()
    return s

# === Your sticky breakeven proxy: TIP / IEF ===
tip = get_close_series("TIP", start="2020-01-01")
ief = get_close_series("IEF", start="2020-01-01")

both = pd.concat([tip, ief], axis=1, join="inner").dropna()
assert not both.empty, "Aligned TIP/IEF is empty. Check tickers or network."

ratio = (both["TIP"] / both["IEF"]).dropna()

print("TIP points:", tip.size, "IEF points:", ief.size, "Aligned:", both.shape[0])
ratio.plot(title="US Breakeven Proxy: TIP / IEF", color="#20545c")
plt.ylabel("Ratio"); plt.tight_layout(); plt.show()
