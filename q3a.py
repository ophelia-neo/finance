# %% [markdown]
# # Q3b — Regime-Aware Investment Strategy Analysis
# 
# This notebook implements a comprehensive regime-aware investment strategy that:
# 1. Builds macro regimes from CPI YoY and GDP YoY data
# 2. Analyzes asset performance across different economic regimes
# 3. Backtests a dynamic allocation strategy vs a 60/40 benchmark
# 
# **Key Features:**
# - Economic regime classification based on inflation and growth trends
# - Bond total return approximation using duration/convexity model
# - Risk-adjusted performance metrics by regime
# - Dynamic asset allocation with regime-specific weights

# %% [markdown]
# ## Setup and Configuration

# %%
from readline import redisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries loaded successfully!")

# %%
# Configuration Parameters
EXCEL_PATH = '/Users/ophelianeo/Downloads/Dataset for Q3b.xlsx'  # Adjust this path as needed
OUTPUT_FILE = '/Users/ophelianeo/Downloads/Dataset for Q3b_output.xlsx'

# Bond modeling parameters
BOND_DURATION = 8.0     # Effective duration for 10Y Treasury
BOND_CONVEXITY = 60.0   # Convexity assumption
RF_MONTHLY = 0.0        # Risk-free rate (monthly)

print(f"Configuration set:")
print(f"- Excel input: {EXCEL_PATH}")
print(f"- Excel output: {OUTPUT_FILE}")
print(f"- Bond duration: {BOND_DURATION}")
print(f"- Bond convexity: {BOND_CONVEXITY}")

# %% [markdown]
# ## Data Loading and Preprocessing

# %%
def load_and_validate_data(excel_path):
    """Load data from Excel file and perform basic validation"""
    try:
        if not os.path.exists(excel_path):
            available_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
            raise FileNotFoundError(f"Excel file '{excel_path}' not found! Available Excel files: {available_files}")
        
        xls = pd.ExcelFile(excel_path)
        print(f"Sheet names found: {xls.sheet_names}")
        
        # Load all sheets
        macro = pd.read_excel(xls, sheet_name='Macro')
        prices = pd.read_excel(xls, sheet_name='Prices')
        yld = pd.read_excel(xls, sheet_name='Yield')
        
        # Convert dates and sort
        for df in (macro, prices, yld):
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)
        
        print("✓ Data loaded successfully!")
        print(f"- Macro data: {macro.shape[0]} rows, {macro.shape[1]} columns")
        print(f"- Prices data: {prices.shape[0]} rows, {prices.shape[1]} columns") 
        print(f"- Yield data: {yld.shape[0]} rows, {yld.shape[1]} columns")
        
        return macro, prices, yld
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

# Load the data
macro, prices, yld = load_and_validate_data(EXCEL_PATH)

# %%
# Display data samples
if macro is not None:
    print("Macro Data Sample:")
    redisplay(macro.head())
    print("\nPrices Data Sample:")
    redisplay(prices.head())
    print("\nYield Data Sample:")
    redisplay(yld.head())

# %% [markdown]
# ## Regime Classification System

# %%
def build_macro_regimes(macro_df):
    """Build economic regimes from macro indicators"""
    macro = macro_df.copy()
    macro = macro.sort_values('Date').reset_index(drop=True)
    macro['GDP YOY'] = macro['GDP YOY'].ffill()
    
    # Calculate 6-month changes for trend identification
    macro['CPI_YoY_6m_chg'] = macro['CPI YOY'] - macro['CPI YOY'].shift(6)
    macro['GDP_YoY_6m_chg'] = macro['GDP YOY'] - macro['GDP YOY'].shift(6)
    
    def trend_state(x):
        if pd.isna(x):
            return np.nan
        return 'Rising' if x > 0 else ('Falling' if x < 0 else 'Flat')
    
    macro['InflationTrend'] = macro['CPI_YoY_6m_chg'].apply(trend_state)
    macro['GrowthTrend'] = macro['GDP_YoY_6m_chg'].apply(trend_state)
    macro['RecessionFlag'] = (macro['GDP YOY'] <= 0).astype(int)
    
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
    return macro

if macro is not None:
    regimes_df = build_macro_regimes(macro)
    regimes = regimes_df[['Date','GDP YOY','CPI YOY','InflationTrend','GrowthTrend','RecessionFlag','Regime']].dropna(subset=['Regime']).reset_index(drop=True)
    
    print("✓ Regimes built successfully!")
    print(f"Total periods with regime classification: {len(regimes)}")
    print("\nRegime Distribution:")
    regime_counts = regimes['Regime'].value_counts()
    display(regime_counts.to_frame('Count'))

# %%
# Visualize regime timeline
if 'regimes' in locals():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: GDP and CPI trends
    ax1.plot(regimes['Date'], regimes['GDP YOY'], label='GDP YoY (%)', linewidth=2, color='blue')
    ax1.plot(regimes['Date'], regimes['CPI YOY'], label='CPI YoY (%)', linewidth=2, color='red')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Economic Indicators Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime timeline
    regime_colors = {
        'High Inflation / Slowing Growth': 'red',
        'Disinflationary Expansion': 'green', 
        'Recessionary Pressure': 'orange',
        'Reflation / Overheating': 'purple',
        'Neutral/Transition': 'gray'
    }
    
    for i, regime in enumerate(regimes['Regime'].unique()):
        mask = regimes['Regime'] == regime
        ax2.scatter(regimes.loc[mask, 'Date'], [i] * mask.sum(), 
                   label=regime, c=regime_colors.get(regime, 'black'), alpha=0.7, s=20)
    
    ax2.set_ylabel('Regime')
    ax2.set_title('Economic Regime Timeline', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime duration histogram
    regime_counts.plot(kind='bar', ax=ax3, color=[regime_colors.get(regime, 'black') for regime in regime_counts.index])
    ax3.set_title('Regime Frequency (Months)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Months')
    ax3.set_xlabel('Economic Regime')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Asset Return Calculations

# %%
def calculate_asset_returns(prices_df, yield_df, duration=8.0, convexity=60.0):
    """Calculate monthly returns for all assets including bond total returns"""
    
    # Resample to month-end
    prices_m = prices_df.set_index('Date').resample('M').last()
    yld_m = yield_df.set_index('Date').resample('M').last()
    
    # Calculate price-based returns
    asset_cols = [c for c in prices_m.columns if c.lower() != 'date']
    ret_px = prices_m[asset_cols].pct_change()
    
    # Bond total return approximation from yield changes
    y = (yld_m['US 10YR Bonds'] / 100.0).copy()  # Convert % to decimal
    dy = y.diff().fillna(0.0)
    
    # TR = -Duration * Δy + 0.5*Convexity*(Δy^2) + carry
    bond_tr = (-duration * dy) + (0.5 * convexity * (dy**2)) + (y.shift(1) / 12.0)
    bond_tr.name = 'US 10YR TR (approx)'
    
    # Combine all returns
    rets = ret_px.copy()
    rets[bond_tr.name] = bond_tr
    
    return rets

if prices is not None and yld is not None:
    returns = calculate_asset_returns(prices, yld, BOND_DURATION, BOND_CONVEXITY)
    
    print("✓ Asset returns calculated!")
    print(f"Return series shape: {returns.shape}")
    print("\nAsset columns:")
    for col in returns.columns:
        print(f"- {col}")
    
    # Display return statistics
    print("\nReturn Statistics (Monthly):")
    stats = pd.DataFrame({
        'Mean': returns.mean(),
        'Std': returns.std(), 
        'Min': returns.min(),
        'Max': returns.max()
    }).round(4)
    display(stats)

# %%
# Visualize return distributions
if 'returns' in locals():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(returns.columns):
        if i < 4:  # Plot first 4 assets
            returns[col].hist(bins=50, ax=axes[i], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col} - Monthly Returns', fontweight='bold')
            axes[i].set_xlabel('Return')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_ret = returns[col].mean()
            std_ret = returns[col].std()
            axes[i].axvline(mean_ret, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_ret:.3f}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.show()

# %% [markdown] 
# ## Performance Analysis by Regime

# %%
def analyze_performance_by_regime(returns_df, regimes_df):
    """Analyze asset performance across different economic regimes"""
    
    # Align returns with regimes
    regimes_m = regimes_df.set_index('Date').reindex(returns_df.index)
    
    def ann_metrics(x, periods_per_year=12):
        """Calculate annualized performance metrics"""
        mu = x.mean() * periods_per_year
        sd = x.std() * np.sqrt(periods_per_year)
        sharpe = (mu - RF_MONTHLY * periods_per_year) / (sd if sd != 0 else np.nan)
        return pd.Series({'AnnReturn': mu, 'AnnVol': sd, 'Sharpe': sharpe})
    
    # Calculate performance by regime
    performance_data = returns_df.join(regimes_m['Regime']).dropna(subset=['Regime'])
    by_regime_stats = (
        performance_data.groupby('Regime')
        .apply(lambda df: df.drop(columns=['Regime'], errors='ignore').apply(ann_metrics), include_groups=False)
    )
    
    # Flatten MultiIndex for better display
    by_regime_stats = by_regime_stats.stack().unstack(0)
    
    # Calculate correlations by regime
    correlations = {}
    for regime in regimes_df['Regime'].dropna().unique():
        regime_data = performance_data[performance_data['Regime'] == regime].drop(columns=['Regime'], errors='ignore')
        if len(regime_data) > 3:
            correlations[regime] = regime_data.corr()
    
    return by_regime_stats, correlations, regimes_m

if 'returns' in locals() and 'regimes' in locals():
    regime_performance, regime_correlations, regimes_monthly = analyze_performance_by_regime(returns, regimes)
    
    print("✓ Performance analysis by regime completed!")
    print(f"Analyzed {len(regime_correlations)} regimes with sufficient data")
    
    print("\nPerformance by Regime (Annualized):")
    display(regime_performance.round(3))

# %%
# Visualize performance by regime
if 'regime_performance' in locals():
    # Create performance heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    metrics = ['AnnReturn', 'AnnVol', 'Sharpe']
    titles = ['Annualized Returns', 'Annualized Volatility', 'Sharpe Ratios']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        data_for_heatmap = regime_performance.loc[metric].T
        sns.heatmap(data_for_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0, ax=axes[i], cbar_kws={'shrink': 0.8})
        axes[i].set_title(title, fontweight='bold', fontsize=12)
        axes[i].set_ylabel('Assets')
        axes[i].set_xlabel('Economic Regimes')
    
    plt.tight_layout()
    plt.show()

# %%
# Analyze correlations by regime
if 'regime_correlations' in locals():
    n_regimes = len(regime_correlations)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (regime, corr_matrix) in enumerate(regime_correlations.items()):
        if i < 6:  # Show up to 6 regimes
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, ax=axes[i], cbar_kws={'shrink': 0.8})
            axes[i].set_title(f'Correlations: {regime}', fontweight='bold', fontsize=10)
    
    # Hide empty subplots
    for j in range(i+1, 6):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Regime-Aware Allocation Strategy

# %%
def build_regime_strategy(returns_df, regimes_monthly_df):
    """Build and backtest regime-aware allocation strategy"""
    
    # Define regime-specific asset weights
    weight_map = {
        'High Inflation / Slowing Growth': {
            'S&P 500': 0.10, 'Gold': 0.45, 'USD Index Spot Rate': 0.25, 'US 10YR TR (approx)': 0.20
        },
        'Disinflationary Expansion': {
            'S&P 500': 0.55, 'Gold': 0.10, 'USD Index Spot Rate': 0.05, 'US 10YR TR (approx)': 0.30
        },
        'Recessionary Pressure': {
            'S&P 500': 0.15, 'Gold': 0.20, 'USD Index Spot Rate': 0.15, 'US 10YR TR (approx)': 0.50
        },
        'Reflation / Overheating': {
            'S&P 500': 0.50, 'Gold': 0.25, 'USD Index Spot Rate': 0.05, 'US 10YR TR (approx)': 0.20
        },
        'Neutral/Transition': {
            'S&P 500': 0.35, 'Gold': 0.20, 'USD Index Spot Rate': 0.10, 'US 10YR TR (approx)': 0.35
        }
    }
    
    # Use lagged regime to avoid look-ahead bias
    regime_lag = regimes_monthly_df['Regime'].shift(1)
    
    # Build dynamic weights
    assets_for_w = ['S&P 500', 'Gold', 'USD Index Spot Rate', 'US 10YR TR (approx)']
    weights = pd.DataFrame(index=returns_df.index, columns=assets_for_w, dtype=float)
    
    for dt in weights.index:
        regime = regime_lag.loc[dt] if dt in regime_lag.index else np.nan
        if regime in weight_map:
            for asset in assets_for_w:
                weights.loc[dt, asset] = weight_map[regime].get(asset, 0.0)
        else:
            weights.loc[dt] = np.nan
    
    # Forward fill weights for missing regimes
    weights = weights.ffill()
    
    # Calculate portfolio returns
    common_assets = [c for c in assets_for_w if c in returns_df.columns]
    portfolio_returns = (weights[common_assets] * returns_df[common_assets]).sum(axis=1)
    portfolio_returns.name = 'Regime-Aware Strategy'
    
    # 60/40 benchmark
    benchmark_returns = 0.60 * returns_df['S&P 500'] + 0.40 * returns_df['US 10YR TR (approx)']
    benchmark_returns.name = '60/40 Benchmark'
    
    return portfolio_returns, benchmark_returns, weights

if 'returns' in locals() and 'regimes_monthly' in locals():
    strategy_returns, benchmark_returns, dynamic_weights = build_regime_strategy(returns, regimes_monthly)
    
    print("✓ Regime-aware strategy built!")
    print(f"Strategy returns calculated for {len(strategy_returns)} periods")
    
    # Show current allocation example
    print("\nExample Regime-Specific Allocations:")
    weight_map = {
        'High Inflation / Slowing Growth': {'Stocks': '10%', 'Gold': '45%', 'USD': '25%', 'Bonds': '20%'},
        'Disinflationary Expansion': {'Stocks': '55%', 'Gold': '10%', 'USD': '5%', 'Bonds': '30%'},
        'Recessionary Pressure': {'Stocks': '15%', 'Gold': '20%', 'USD': '15%', 'Bonds': '50%'},
        'Reflation / Overheating': {'Stocks': '50%', 'Gold': '25%', 'USD': '5%', 'Bonds': '20%'},
        'Neutral/Transition': {'Stocks': '35%', 'Gold': '20%', 'USD': '10%', 'Bonds': '35%'}
    }
    
    for regime, weights in weight_map.items():
        print(f"\n{regime}:")
        for asset, weight in weights.items():
            print(f"  {asset}: {weight}")

# %% [markdown]
# ## Strategy Performance Analysis

# %%
def calculate_performance_metrics(returns_series, periods_per_year=12):
    """Calculate comprehensive performance metrics"""
    returns = returns_series.dropna()
    
    if len(returns) == 0:
        return pd.Series(dtype=float)
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    cagr = (1 + returns).prod()**(periods_per_year/len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    
    # Risk metrics
    cumulative = (1 + returns.fillna(0)).cumprod()
    peak = cumulative.cummax()
    drawdowns = (cumulative / peak) - 1
    max_drawdown = drawdowns.min()
    
    # Risk-adjusted metrics
    excess_return = returns.mean() * periods_per_year - RF_MONTHLY * periods_per_year
    sharpe_ratio = excess_return / annual_vol if annual_vol != 0 else np.nan
    
    # Downside metrics
    negative_returns = returns[returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0
    sortino_ratio = excess_return / downside_vol if downside_vol != 0 else np.nan
    
    # Hit rate
    hit_rate = (returns > 0).mean()
    
    return pd.Series({
        'Total Return': total_return,
        'CAGR': cagr,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Hit Rate': hit_rate
    })

if 'strategy_returns' in locals() and 'benchmark_returns' in locals():
    # Calculate performance metrics
    strategy_metrics = calculate_performance_metrics(strategy_returns)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns)
    
    performance_comparison = pd.concat([strategy_metrics, benchmark_metrics], axis=1)
    performance_comparison.columns = ['Regime-Aware Strategy', '60/40 Benchmark']
    
    print("Performance Comparison:")
    display(performance_comparison.round(4))
    
    # Calculate equity curves
    strategy_equity = (1 + strategy_returns.fillna(0)).cumprod()
    benchmark_equity = (1 + benchmark_returns.fillna(0)).cumprod()
    
    equity_curves = pd.concat([strategy_equity, benchmark_equity], axis=1)
    equity_curves.columns = ['Regime-Aware Strategy', '60/40 Benchmark']

# %%
# Visualize strategy performance
if 'equity_curves' in locals():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Equity curves
    equity_curves.plot(ax=ax1, linewidth=2, title='Cumulative Performance')
    ax1.set_ylabel('Growth of $1')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Drawdowns
    strategy_dd = (equity_curves['Regime-Aware Strategy'] / equity_curves['Regime-Aware Strategy'].cummax()) - 1
    benchmark_dd = (equity_curves['60/40 Benchmark'] / equity_curves['60/40 Benchmark'].cummax()) - 1
    
    ax2.fill_between(strategy_dd.index, strategy_dd, 0, alpha=0.7, label='Regime-Aware Strategy')
    ax2.fill_between(benchmark_dd.index, benchmark_dd, 0, alpha=0.7, label='60/40 Benchmark')
    ax2.set_title('Drawdowns')
    ax2.set_ylabel('Drawdown')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Rolling volatility (12-month)
    strategy_vol = strategy_returns.rolling(12).std() * np.sqrt(12)
    benchmark_vol = benchmark_returns.rolling(12).std() * np.sqrt(12)
    
    strategy_vol.plot(ax=ax3, label='Regime-Aware Strategy', linewidth=2)
    benchmark_vol.plot(ax=ax3, label='60/40 Benchmark', linewidth=2)
    ax3.set_title('12-Month Rolling Volatility')
    ax3.set_ylabel('Annualized Volatility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Return distribution comparison
    strategy_returns.hist(bins=50, alpha=0.7, ax=ax4, label='Regime-Aware Strategy')
    benchmark_returns.hist(bins=50, alpha=0.7, ax=ax4, label='60/40 Benchmark')
    ax4.set_title('Monthly Return Distributions')
    ax4.set_xlabel('Monthly Return')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
# Annual performance breakdown
if 'strategy_returns' in locals() and 'benchmark_returns' in locals():
    annual_performance = pd.DataFrame({
        'Strategy': strategy_returns.groupby(strategy_returns.index.year).apply(lambda x: (1+x).prod()-1),
        'Benchmark': benchmark_returns.groupby(benchmark_returns.index.year).apply(lambda x: (1+x).prod()-1)
    }).round(4)
    
    print("Annual Performance Comparison:")
    display(annual_performance)
    
    # Visualize annual performance
    annual_performance.plot(kind='bar', figsize=(12, 6), 
                          title='Annual Performance Comparison')
    plt.ylabel('Annual Return')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Export Results

# %%
def export_all_results(output_path, **data_dict):
    """Export all analysis results to Excel file"""
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, data in data_dict.items():
                if data is not None:
                    if isinstance(data, pd.DataFrame):
                        data.to_excel(writer, sheet_name=sheet_name[:31])  # Excel sheet name limit
                    else:
                        pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name[:31])
        
        print(f"✓ All results exported to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

# Prepare data for export
if 'regimes' in locals():
    export_data = {
        'Regimes': regimes,
        'Regime_Counts': regimes['Regime'].value_counts().to_frame('Count'),
        'Monthly_Returns': returns if 'returns' in locals() else None,
        'Performance_by_Regime': regime_performance.T if 'regime_performance' in locals() else None,
        'Strategy_Performance': performance_comparison if 'performance_comparison' in locals() else None,
        'Equity_Curves': equity_curves if 'equity_curves' in locals() else None,
        'Annual_Performance': annual_performance if 'annual_performance' in locals() else None
    }
    
    # Add correlation matrices
    if 'regime_correlations' in locals():
        for regime, corr_matrix in regime_correlations.items():
            safe_name = regime.replace('/', '_').replace(' ', '_')[:25]
            export_data[f'Corr_{safe_name}'] = corr_matrix
    
    # Export to Excel
    success = export_all_results(OUTPUT_FILE, **export_data)
    
    if success:
        print(f"\n📊 Analysis Complete!")
        print(f"📁 Results saved to: {OUTPUT_FILE}")
        print(f"📈 Strategy vs Benchmark Performance:")
        if 'performance_comparison' in locals():
            key_metrics = performance_comparison.loc[['CAGR', 'Sharpe Ratio', 'Max Drawdown']].round(4)
            display(key_metrics)

# %% [markdown]
# ## Summary and Conclusions
# 
# ### Key Findings:
# 
# 1. **Regime Classification**: The analysis identifies distinct economic regimes based on inflation and growth trends
# 2. **Asset Performance**: Different assets perform better in different regimes (e.g., gold during high inflation periods)
# 3. **Dynamic Allocation**: The regime-aware strategy adjusts allocations based on economic conditions
# 4. **Risk Management**: Strategic allocation helps manage downside risk during adverse regimes
# 
# ### Strategy Benefits:
# - **Adaptive**: Responds to changing economic conditions
# - **Diversified**: Uses multiple asset classes for risk management  
# - **Data-Driven**: Based on historical regime analysis
# - **Systematic**: Removes emotional bias from allocation decisions
# 
# ### Next Steps:
# 1. **Robustness Testing**: Test with different regime definitions and parameters
# 2. **Transaction Costs**: Include realistic trading costs and constraints
# 3. **Out-of-Sample Testing**: Validate on more recent data
# 4. **Alternative Assets**: Consider adding other asset classes (REITs, commodities, etc.)
# 
# *This analysis provides a framework for regime-aware investing that can be further refined and customized based on specific investment objectives