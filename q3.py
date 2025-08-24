# Q3b — Regime-Aware Strategy (Full Python Code, Runnable End-to-End)
# -------------------------------------------------------------------
# This script:
# 1) Builds macro regimes from CPI YoY and GDP YoY (Macro sheet)
# 2) Resamples asset prices/yields to month-end and computes monthly returns
# 3) Analyzes asset performance by regime (return/vol/Sharpe/corr)
# 4) Backtests a simple regime-aware allocation vs a 60/40 benchmark
#
# DATA EXPECTATION (as in your uploaded workbook):
# - Excel file: 'Dataset for Q3b.xlsx'
#   Sheets:
#     Macro: [Date, GDP YOY, CPI YOY]  (monthly CPI YOY; GDP YOY quarterly but forward-filled)
#     Prices: [Date, S&P 500, Gold, USD Index Spot Rate] (daily or irregular)
#     Yield:  [Date, US 10YR Bonds]  (daily 10y yield level, in %)
#
# OUTPUTS:
# - CSVs with ';' delimiter saved in current directory
# - Basic charts and dataframes displayed
#
# NOTES:
# - Bond total return is approximated from yield changes using a duration-based
#   approach: TR ≈ -Duration * Δy + 0.5*Convexity*(Δy^2) + carry (y/12).
#   We assume Duration=8 and Convexity=60 as rough constants for a 10Y Treasury.
#   This is a common teaching proxy when full TR indices are not available.
#
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def display_dataframe_to_user(title, df):
    print(f"\n{title}")
    print("-" * len(title))
    print(df)

# ---------------------------
# CONFIG
# ---------------------------
# Updated file paths - modify these to match your actual file locations
EXCEL_PATH = '/Users/ophelianeo/Downloads/Dataset for Q3b.xlsx'  # Adjust this path as needed
OUTPUT_FILE = '/Users/ophelianeo/Downloads/Dataset for Q3b_output.xlsx'  # Single Excel output file

# Create output directory if it doesn't exist
output_dir = Path(OUTPUT_FILE).parent
output_dir.mkdir(exist_ok=True)

BOND_DURATION = 8.0     # rough effective duration for 10Y
BOND_CONVEXITY = 60.0   # rough convexity (bp^-2 scaled for decimal)
RF_MONTHLY = 0.0        # set to 0 unless you add a T-bill proxy (decimal monthly)

try:
    # ---------------------------
    # LOAD DATA
    # ---------------------------
    print("Loading data from Excel file...")
    
    if not os.path.exists(EXCEL_PATH):
        print(f"Error: Excel file '{EXCEL_PATH}' not found!")
        print("Please ensure the Excel file is in the same directory as this script.")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")
    
    xls = pd.ExcelFile(EXCEL_PATH)
    print(f"Sheet names found: {xls.sheet_names}")
    
    macro = pd.read_excel(xls, sheet_name='Macro')
    prices = pd.read_excel(xls, sheet_name='Prices')
    yld = pd.read_excel(xls, sheet_name='Yield')

    for df in (macro, prices, yld):
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

    print("Data loaded successfully!")
    print(f"Macro data shape: {macro.shape}")
    print(f"Prices data shape: {prices.shape}")
    print(f"Yield data shape: {yld.shape}")

    # ---------------------------
    # MACRO: Build Regimes
    # ---------------------------
    print("\nBuilding macro regimes...")
    
    macro = macro.sort_values('Date').reset_index(drop=True)
    macro['GDP YOY'] = macro['GDP YOY'].ffill()

    # 6m changes for trend signals
    macro['CPI_YoY_6m_chg'] = macro['CPI YOY'] - macro['CPI YOY'].shift(6)
    macro['GDP_YoY_6m_chg'] = macro['GDP YOY'] - macro['GDP YOY'].shift(6)

    def trend_state(x):
        if pd.isna(x):
            return np.nan
        return 'Rising' if x > 0 else ('Falling' if x < 0 else 'Flat')

    macro['InflationTrend'] = macro['CPI_YoY_6m_chg'].apply(trend_state)
    macro['GrowthTrend']    = macro['GDP_YoY_6m_chg'].apply(trend_state)
    macro['RecessionFlag']  = (macro['GDP YOY'] <= 0).astype(int)

    def label_regime(row):
        if row['RecessionFlag'] == 1:
            return 'Recessionary Pressure'
        it = row['InflationTrend']
        gt = row['GrowthTrend']
        if pd.isna(it) or pd.isna(gt):
            return np.nan
        if it == 'Rising' and gt == 'Falling':
            return 'High Inflation / Slowing Growth'
        if it == 'Falling' and gt == 'Rising':
            return 'Disinflationary Expansion'
        if it == 'Falling' and gt == 'Falling':
            return 'Recessionary Pressure'
        if it == 'Rising' and gt == 'Rising':
            return 'Reflation / Overheating'
        return 'Neutral/Transition'

    macro['Regime'] = macro.apply(label_regime, axis=1)
    regimes = macro[['Date','GDP YOY','CPI YOY','InflationTrend','GrowthTrend','RecessionFlag','Regime']].dropna(subset=['Regime']).reset_index(drop=True)

    # Prepare regime tables for Excel output
    regime_counts = regimes['Regime'].value_counts().rename_axis('Regime').reset_index(name='Months')
    regime_ranges = regimes.groupby('Regime')['Date'].agg(['min','max']).reset_index()

    # ---------------------------
    # ASSETS: Resample to Month-End & Compute Returns
    # ---------------------------
    print("\nProcessing asset data...")
    
    # Prices: S&P 500, Gold, USD Index (assume price levels → pct returns)
    prices_m = prices.set_index('Date').resample('M').last()
    asset_cols = [c for c in prices_m.columns if c.lower() != 'date']

    # Yields: 10Y yield level in %, to month-end
    yld_m = yld.set_index('Date').resample('M').last()

    # Compute price-based monthly returns
    ret_px = prices_m.pct_change()

    # Bond TR approximation from yield level
    # Convert % to decimal yields
    y = (yld_m['US 10YR Bonds'] / 100.0).copy()
    dy = y.diff().fillna(0.0)
    # Δy is in decimals; duration/convexity terms work in decimals
    # Add simple monthly carry (y/12)
    bond_tr = (-BOND_DURATION * dy) + (0.5 * BOND_CONVEXITY * (dy**2)) + (y.shift(1) / 12.0)
    bond_tr.name = 'US 10YR TR (approx)'

    rets = ret_px.copy()
    rets[bond_tr.name] = bond_tr

    # Align to regimes (month-end). Regimes are monthly already, but align to end of month
    regimes_m = regimes.set_index('Date').reindex(rets.index).copy()

    # ---------------------------
    # PERFORMANCE BY REGIME
    # ---------------------------
    print("\nAnalyzing performance by regime...")
    
    def ann_metrics(x, periods_per_year=12):
        mu = x.mean() * periods_per_year
        sd = x.std() * np.sqrt(periods_per_year)
        sharpe = (mu - RF_MONTHLY * periods_per_year) / (sd if sd != 0 else np.nan)
        return pd.Series({'AnnReturn': mu, 'AnnVol': sd, 'Sharpe': sharpe})

    by_regime_stats = (
        rets.join(regimes_m['Regime'])
            .dropna(subset=['Regime'])
            .groupby('Regime')
            .apply(lambda df: df.drop(columns=['Regime'], errors='ignore').apply(ann_metrics), include_groups=False)
    )

    # Save per-regime performance (will be saved to Excel later)
    # Flatten MultiIndex for Excel output
    by_regime_stats = by_regime_stats.stack(0).unstack(1).sort_index()

    # Correlations by regime (returns) - store for Excel output
    corrs = {}
    corr_sheets = {}
    for r in regimes['Regime'].dropna().unique():
        df = rets.join(regimes_m['Regime'])
        sub = df[df['Regime'] == r].drop(columns=['Regime'], errors='ignore')
        if len(sub) > 3:
            corrs[r] = sub.corr()
            safe_name = r.replace("/", "-").replace(" ", "_")[:25]  # Excel sheet name limit
            corr_sheets[f'Corr_{safe_name}'] = corrs[r]

    # ---------------------------
    # REGIME-AWARE ALLOCATION & BACKTEST
    # ---------------------------
    print("\nBuilding regime-aware allocation strategy...")
    
    # Define weights by regime (stocks, gold, USD, bonds)
    # Weights sum to 1.0
    weight_map = {
        'High Inflation / Slowing Growth': {'S&P 500': 0.10, 'Gold': 0.45, 'USD Index Spot Rate': 0.25, bond_tr.name: 0.20},
        'Disinflationary Expansion':       {'S&P 500': 0.55, 'Gold': 0.10, 'USD Index Spot Rate': 0.05, bond_tr.name: 0.30},
        'Recessionary Pressure':           {'S&P 500': 0.15, 'Gold': 0.20, 'USD Index Spot Rate': 0.15, bond_tr.name: 0.50},
        'Reflation / Overheating':         {'S&P 500': 0.50, 'Gold': 0.25, 'USD Index Spot Rate': 0.05, bond_tr.name: 0.20},
        'Neutral/Transition':              {'S&P 500': 0.35, 'Gold': 0.20, 'USD Index Spot Rate': 0.10, bond_tr.name: 0.35},
    }

    # Use previous month's regime to avoid look-ahead
    regime_lag = regimes_m['Regime'].shift(1)

    # Build weight time series
    assets_for_w = ['S&P 500', 'Gold', 'USD Index Spot Rate', bond_tr.name]
    W = pd.DataFrame(index=rets.index, columns=assets_for_w, dtype=float)

    for dt in W.index:
        r = regime_lag.loc[dt]
        if r in weight_map:
            for a in assets_for_w:
                W.loc[dt, a] = weight_map[r].get(a, 0.0)
        else:
            W.loc[dt] = np.nan

    W = W.ffill()

    # Compute portfolio returns
    common_cols = [c for c in assets_for_w if c in rets.columns]
    port_ret = (W[common_cols] * rets[common_cols]).sum(axis=1)
    port_ret.name = 'Regime-Aware'

    # 60/40 benchmark (S&P 500 / Bonds)
    bench_ret = 0.60 * rets['S&P 500'] + 0.40 * rets[bond_tr.name]
    bench_ret.name = '60/40'

    # Summary stats
    def perf_summary(r):
        cagr = (1 + r.dropna()).prod()**(12/len(r.dropna())) - 1 if len(r.dropna())>0 else np.nan
        vol = r.std() * np.sqrt(12)
        dd = (1 + r.fillna(0)).cumprod()
        peak = dd.cummax()
        mdd = ((dd/peak) - 1).min()
        sharpe = (r.mean()*12 - RF_MONTHLY*12) / (r.std()*np.sqrt(12)) if r.std() != 0 else np.nan
        return pd.Series({'CAGR': cagr, 'AnnVol': vol, 'Sharpe': sharpe, 'MaxDD': mdd})

    summary = pd.concat([perf_summary(port_ret), perf_summary(bench_ret)], axis=1)
    summary.columns = ['Regime-Aware', '60/40']

    # Equity curves
    eq = pd.concat([ (1+port_ret.fillna(0)).cumprod(),
                     (1+bench_ret.fillna(0)).cumprod() ], axis=1)
    eq.columns = ['Regime-Aware', '60/40']

    # ---------------------------
    # SAVE ALL RESULTS TO SINGLE EXCEL FILE
    # ---------------------------
    print(f"\nSaving all results to: {OUTPUT_FILE}")
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # Main sheets
        regimes.to_excel(writer, sheet_name='Regimes', index=False)
        regime_counts.to_excel(writer, sheet_name='Regime_Counts', index=False)
        regime_ranges.to_excel(writer, sheet_name='Regime_Ranges', index=False)
        by_regime_stats.round(4).to_excel(writer, sheet_name='Per_Regime_Stats')
        summary.round(4).to_excel(writer, sheet_name='Strategy_Summary')
        eq.to_excel(writer, sheet_name='Equity_Curves')
        
        # Returns data
        rets.to_excel(writer, sheet_name='Monthly_Returns')
        
        # Correlation matrices by regime
        for sheet_name, corr_df in corr_sheets.items():
            corr_df.round(3).to_excel(writer, sheet_name=sheet_name)

    # ---------------------------
    # DISPLAY KEY RESULTS
    # ---------------------------
    print("\n" + "="*60)
    print("KEY RESULTS")
    print("="*60)
    
    display_dataframe_to_user("Q3b — Regimes (Monthly) - First 24 rows", regimes.head(24))
    display_dataframe_to_user("Q3b — Regime Counts", regime_counts)
    display_dataframe_to_user("Q3b — Per-Regime Stats (Annualized)", by_regime_stats.round(3))
    display_dataframe_to_user("Q3b — Strategy Summary", summary.round(3))

    # Plot equity curves
    plt.figure(figsize=(12,6))
    plt.plot(eq.index, eq['Regime-Aware'], label='Regime-Aware', linewidth=2)
    plt.plot(eq.index, eq['60/40'], label='60/40 Benchmark', linewidth=2)
    plt.title('Equity Curves: Regime-Aware Strategy vs 60/40 Benchmark', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Growth of $1', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("OUTPUT SAVED")
    print("="*60)
    print(f"All analysis results saved to: {OUTPUT_FILE}")
    print("\nExcel sheets created:")
    print("- Regimes: Monthly regime classifications")
    print("- Regime_Counts: Number of months in each regime") 
    print("- Regime_Ranges: Date ranges for each regime")
    print("- Per_Regime_Stats: Performance statistics by regime")
    print("- Strategy_Summary: Backtest performance comparison")
    print("- Equity_Curves: Portfolio growth over time")
    print("- Monthly_Returns: Raw monthly return data")
    print("- Corr_[Regime]: Correlation matrices by regime")
    
    print(f"\nAnalysis complete! Open {OUTPUT_FILE} to view all results.")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("\nTo run this script, you need:")
    print("1. The Excel file 'Dataset for Q3b.xlsx' in the same directory")
    print("2. The Excel file should have three sheets: 'Macro', 'Prices', and 'Yield'")
    
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()