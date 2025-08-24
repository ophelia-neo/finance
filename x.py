# === 10Y Sovereign Yields from LOCAL CSVs only ===
# Outputs:
# 1) 10-Year Sovereign Yields chart
# 2) Turkey–Germany 10Y spread (if both present)
# 3) Egypt–Germany 10Y spread (if both present)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# --- add this near the top (after imports) ---
PAL_TEAL = {
    "Germany 10Y":  "#20545c",  # deep teal (anchor)
    "Turkey 10Y":   "#317F89",
    "Egypt 10Y":    "#6BAAAE",
    "Singapore 10Y": "#ADD5D7"
}
TITLE = "#20545c"


# ---------- CONFIG: map label -> CSV path ----------
CSV_FILES = {
    "Germany 10Y":  "/Users/ophelianeo/Downloads/10ydey_b_d.csv",
    "Turkey 10Y":   "/Users/ophelianeo/Downloads/10ytry_b_d.csv",
    "Egypt 10Y":    "/Users/ophelianeo/Downloads/10yegy_b_d.csv",
    "Singapore 10Y": "/Users/ophelianeo/Downloads/10ysgy_b_d.csv"
}
START = "2022-01-01"   # inclusive; set None to keep all dates
END   = None           # set e.g. "2025-08-17" to cap the range

# ---------- helpers ----------
def to_daily_ffill(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index().asfreq("D").ffill()

def load_yield_csv_generic(path: str, name: str,
                           date_col: str | None = None,
                           value_col: str | None = None) -> pd.Series:
    """
    Reads a CSV with columns [Date, Yield] (order/headers can vary).
    - Auto-detects delimiter and column names (or pass date_col/value_col)
    - Cleans %, commas; converts bps→% if needed; converts decimals to % if needed
    - Returns DAILY forward-filled series in percent
    """
    df = pd.read_csv(path, sep=None, engine="python")
    if date_col is None:  date_col  = df.columns[0]
    if value_col is None: value_col = df.columns[1]

    vals = (df[value_col].astype(str)
                      .str.replace("%", "", regex=False)
                      .str.replace(",", "", regex=False)
                      .str.strip())
    y = pd.to_numeric(vals, errors="coerce")
    d = pd.to_datetime(df[date_col], errors="coerce")

    s = pd.Series(y.values, index=d, name=name).dropna().sort_index()

    # Normalize units → percent
    if s.max() > 50:      # looks like bps (e.g., 235)
        s = s / 100.0
    elif s.max() < 1:     # looks like decimals (0.023)
        s = s * 100.0

    return to_daily_ffill(s)

# ---------- load series ----------
series = {}
for label, path in CSV_FILES.items():
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Missing file for {label}: {p}")
        continue
    try:
        s = load_yield_csv_generic(str(p), label)
        series[label] = s
    except Exception as e:
        print(f"[ERROR] Could not load {label} from {p}: {e}")

df = pd.DataFrame(series).dropna(how="all")

# date filter
if START: df = df[df.index >= pd.to_datetime(START)]
if END:   df = df[df.index <= pd.to_datetime(END)]

if df.empty:
    raise SystemExit("[ERROR] No data loaded. Check CSV paths and columns.")

# ---------- plots (with teal hues) ----------
# Order lines by our palette; only keep series that exist
plot_order = [k for k in PAL_TEAL if k in df.columns]

fig, ax = plt.subplots(figsize=(10, 6.5))
for name in plot_order:
    ax.plot(df.index, df[name], label=name, color=PAL_TEAL[name], linewidth=2.5)
ax.set_title("10-Year Sovereign Yields (Local CSVs)", color=TITLE, pad=10)
ax.set_ylabel("Yield (%)", color=TITLE)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout(); plt.show()

# spreads (use distinct teal shades)
if {"Turkey 10Y","Germany 10Y"}.issubset(df.columns):
    spread_tr = (df["Turkey 10Y"] - df["Germany 10Y"]).rename("Turkey – Germany 10Y")
    ax = spread_tr.plot(
        figsize=(10, 4.8),
        title="Turkey–Germany 10Y Sovereign Spread",
        ylabel="Spread (pp)",
        color=PAL_TEAL.get("Turkey 10Y", "#20545c"),
        lw=2.5, grid=True
    )
    plt.tight_layout(); plt.show()
else:
    print("[INFO] Turkey–Germany spread not plotted (need both CSVs).")

if {"Egypt 10Y","Germany 10Y"}.issubset(df.columns):
    spread_eg = (df["Egypt 10Y"] - df["Germany 10Y"]).rename("Egypt – Germany 10Y")
    ax = spread_eg.plot(
        figsize=(10, 4.8),
        title="Egypt–Germany 10Y Sovereign Spread",
        ylabel="Spread (pp)",
        color=PAL_TEAL.get("Egypt 10Y", "#6BAAAE"),
        lw=2.5, grid=True
    )
    plt.tight_layout(); plt.show()
else:
    print("[INFO] Egypt–Germany spread not plotted (need both CSVs).")
