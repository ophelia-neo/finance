import pandas as pd

# === Paths ===
tickers_file = "/Users/ophelianeo/Documents/work!/finance/Tickers.xlsx"
ratios_file = "/Users/ophelianeo/Documents/work!/finance/Tickers_with_ratios_20250905_215807.xlsx"
output_file = "/Users/ophelianeo/Documents/work!/finance/Tickers_with_ratios_ordered.xlsx"

# === Load tickers from column D ===
df_tickers = pd.read_excel(tickers_file)
tickers = df_tickers.iloc[:, 3].dropna().astype(str).str.strip().tolist()

# === Load ratios file ===
df_ratios = pd.read_excel(ratios_file)

# If the ratios file has a 'Yahoo_Ticker' or similar column, use that for reindex
ticker_col = "Yahoo_Ticker" if "Yahoo_Ticker" in df_ratios.columns else df_ratios.columns[0]
df_ratios = df_ratios.set_index(ticker_col)

# === Reorder according to tickers file ===
df_ratios_ordered = df_ratios.reindex(tickers)

# === Merge back with tickers file ===
df_combined = pd.concat([df_tickers, df_ratios_ordered.reset_index(drop=True)], axis=1)

# === Save result ===
df_combined.to_excel(output_file, index=False)

print(f"✅ Ordered ratios saved to {output_file}")
