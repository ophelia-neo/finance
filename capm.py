import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Tickers
mango_tickers = ['MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN']
market_ticker = '^GSPC'

# Download adjusted close prices
print("Downloading data...")
raw_data = yf.download(mango_tickers + [market_ticker], start='2023-07-01', end='2024-07-01')

# Handle MultiIndex structure - extract Close prices (already adjusted)
if isinstance(raw_data.columns, pd.MultiIndex):
    print("\nMultiIndex detected - extracting Close prices...")
    print(f"Available top-level columns: {raw_data.columns.get_level_values(0).unique()}")
    data = raw_data['Close']  # Close prices are already adjusted when auto_adjust=True
    print(f"Close data shape: {data.shape}")
    print(f"Close columns: {data.columns}")
else:
    # If single ticker, yfinance returns simple columns
    print("\nSimple columns detected...")
    data = raw_data

# Rename market column
data = data.rename(columns={market_ticker: 'Market'})

# Compute log returns
log_returns = np.log(data / data.shift(1)).dropna()

# Compute equally weighted MANGO portfolio return
mango_returns = log_returns[mango_tickers].mean(axis=1)
log_returns['MANGO'] = mango_returns

# Prepare regression variables
X = sm.add_constant(log_returns['Market'])  # Market returns (independent)
y = log_returns['MANGO']                    # MANGO portfolio returns (dependent)

# Run CAPM regression
print("\nRunning CAPM regression...")
capm_model = sm.OLS(y, X).fit()
print(capm_model.summary())

# Extract key statistics
alpha = capm_model.params['const']
beta = capm_model.params['Market']
r_squared = capm_model.rsquared

print(f"\nKey CAPM Statistics:")
print(f"Alpha (intercept): {alpha:.6f}")
print(f"Beta (market sensitivity): {beta:.4f}")
print(f"R-squared: {r_squared:.4f}")

# Plot CAPM regression
plt.figure(figsize=(10, 6))
sns.regplot(x='Market', y='MANGO', data=log_returns, line_kws={'color': 'red'})
plt.title(f'CAPM Regression: MANGO vs S&P 500\nBeta = {beta:.4f}, R² = {r_squared:.4f}')
plt.xlabel('Market Return (S&P 500)')
plt.ylabel('MANGO Portfolio Return')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate annualized statistics
rf = 0.03  # Assume 3% risk-free rate
mean_market_annual = log_returns['Market'].mean() * 252
mean_mango_annual = log_returns['MANGO'].mean() * 252
std_market_annual = log_returns['Market'].std() * np.sqrt(252)
std_mango_annual = log_returns['MANGO'].std() * np.sqrt(252)

# CAPM expected return
capm_expected_return = rf + beta * (mean_market_annual - rf)

print(f"\nAnnualized Statistics:")
print(f"Market (S&P 500) Return: {mean_market_annual:.4f} ({mean_market_annual*100:.2f}%)")
print(f"Market Volatility: {std_market_annual:.4f} ({std_market_annual*100:.2f}%)")
print(f"MANGO Actual Return: {mean_mango_annual:.4f} ({mean_mango_annual*100:.2f}%)")
print(f"MANGO Volatility: {std_mango_annual:.4f} ({std_mango_annual*100:.2f}%)")
print(f"CAPM Expected Return: {capm_expected_return:.4f} ({capm_expected_return*100:.2f}%)")

# Plot Security Market Line
plt.figure(figsize=(10, 6))
beta_range = np.linspace(0, 2, 100)
sml_returns = rf + beta_range * (mean_market_annual - rf)

plt.plot(beta_range, sml_returns, 'r-', label='Security Market Line', linewidth=2)
plt.scatter(1.0, mean_market_annual, color='green', s=100, label='Market (S&P 500)', zorder=5)
plt.scatter(beta, mean_mango_annual, color='blue', s=100, label='MANGO Portfolio', zorder=5)
plt.scatter(beta, capm_expected_return, color='orange', s=100, label='CAPM Expected', zorder=5)

# Add annotations
plt.annotate('Market', (1.0, mean_market_annual), xytext=(5, 5), 
             textcoords='offset points', fontsize=10)
plt.annotate('MANGO\n(Actual)', (beta, mean_mango_annual), xytext=(5, 5), 
             textcoords='offset points', fontsize=10)
plt.annotate('MANGO\n(CAPM)', (beta, capm_expected_return), xytext=(5, -20), 
             textcoords='offset points', fontsize=10)

plt.xlabel('Beta (Systematic Risk)')
plt.ylabel('Expected Annual Return')
plt.title('Security Market Line Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 2)
plt.ylim(0, max(mean_market_annual, mean_mango_annual) * 1.2)
plt.tight_layout()
plt.show()

# Performance metrics
sharpe_market = (mean_market_annual - rf) / std_market_annual
sharpe_mango = (mean_mango_annual - rf) / std_mango_annual
treynor_mango = (mean_mango_annual - rf) / beta

print(f"\nRisk-Adjusted Performance:")
print(f"Market Sharpe Ratio: {sharpe_market:.4f}")
print(f"MANGO Sharpe Ratio: {sharpe_mango:.4f}")
print(f"MANGO Treynor Ratio: {treynor_mango:.4f}")

# Jensen's Alpha (annualized)
jensen_alpha = mean_mango_annual - capm_expected_return
print(f"Jensen's Alpha: {jensen_alpha:.4f} ({jensen_alpha*100:.2f}%)")

if jensen_alpha > 0:
    print("The MANGO portfolio outperformed its CAPM expected return.")
else:
    print("The MANGO portfolio underperformed its CAPM expected return.")