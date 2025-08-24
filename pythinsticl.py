import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_forex_data(days=250, start_price=26.5):
    """Generate realistic BRL/JPY forex data"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='D')
    
    np.random.seed(42)  # For reproducible results
    
    data = []
    price = start_price
    trend = 0
    
    for date in dates:
        # Add trending behavior
        trend += np.random.normal(0, 0.01)
        trend = np.clip(trend, -0.3, 0.3)
        
        # Daily volatility
        volatility = np.random.uniform(0.2, 0.4)
        daily_change = np.random.normal(trend * 0.05, volatility)
        
        # OHLC calculation
        open_price = price
        close_price = open_price + daily_change
        
        # High and low with some randomness
        high_price = max(open_price, close_price) + np.random.exponential(0.1)
        low_price = min(open_price, close_price) - np.random.exponential(0.1)
        
        data.append({
            'Date': date,
            'Open': round(open_price, 3),
            'High': round(high_price, 3),
            'Low': round(low_price, 3),
            'Close': round(close_price, 3)
        })
        
        price = close_price
    
    return pd.DataFrame(data)

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_candlestick(ax, data, width=0.6):
    """Plot candlestick chart"""
    for idx, (i, row) in enumerate(data.iterrows()):
        date = mdates.date2num(row['Date'])
        open_price, high, low, close = row['Open'], row['High'], row['Low'], row['Close']
        
        # Determine color
        color = '#2E7D32' if close >= open_price else '#C62828'
        fill = close < open_price
        
        # Draw high-low line
        ax.plot([date, date], [low, high], color=color, linewidth=1, alpha=0.8)
        
        # Draw body rectangle
        height = abs(close - open_price)
        bottom = min(open_price, close)
        
        rect = Rectangle((date - width/2, bottom), width, height,
                        facecolor=color if fill else 'none',
                        edgecolor=color, linewidth=1, alpha=0.8)
        ax.add_patch(rect)

def create_brljpy_chart():
    """Create comprehensive BRL/JPY chart with 200DMA and RSI"""
    
    # Generate data
    print("Generating BRL/JPY forex data...")
    df = generate_forex_data()
    
    # Calculate indicators
    print("Calculating technical indicators...")
    df['SMA_200'] = calculate_sma(df, 200)
    df['RSI'] = calculate_rsi(df)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Main price chart (70% of height)
    ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=7)
    
    # Plot candlesticks
    plot_candlestick(ax1, df)
    
    # Plot 200 SMA
    ax1.plot(df['Date'], df['SMA_200'], color='#FF6F00', linewidth=2, 
             label='200 SMA', alpha=0.9)
    
    # Formatting main chart
    ax1.set_title('BRL/JPY - Brazilian Real / Japanese Yen', 
                 fontsize=20, fontweight='bold', pad=20)
    ax1.set_ylabel('Price (JPY)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Remove x-axis labels for main chart
    ax1.set_xticklabels([])
    
    # RSI subplot (30% of height)
    ax2 = plt.subplot2grid((10, 1), (7, 0), rowspan=3, sharex=ax1)
    
    # Plot RSI
    ax2.plot(df['Date'], df['RSI'], color='#7B1FA2', linewidth=2, label='RSI (14)')
    ax2.fill_between(df['Date'], df['RSI'], alpha=0.3, color='#7B1FA2')
    
    # RSI reference lines
    ax2.axhline(y=70, color='#D32F2F', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='#388E3C', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='#666666', linestyle='-', alpha=0.5, linewidth=0.5)
    
    # Format RSI chart
    ax2.set_ylabel('RSI', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    # Format x-axis for RSI chart
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add statistics box
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change_pct = ((current_price - prev_price) / prev_price) * 100
    current_rsi = df['RSI'].iloc[-1]
    current_sma200 = df['SMA_200'].iloc[-1]
    
    stats_text = f"""Current Price: {current_price:.3f}
24h Change: {change_pct:+.2f}%
200 SMA: {current_sma200:.3f}
RSI (14): {current_rsi:.1f}"""
    
    # Add text box with statistics
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Tight layout
    plt.tight_layout()
    
    # Print summary
    print("\n" + "="*60)
    print("BRL/JPY TECHNICAL ANALYSIS SUMMARY")
    print("="*60)
    print(f"Current Price: {current_price:.3f} JPY")
    print(f"24h Change: {change_pct:+.2f}%")
    print(f"200-Day SMA: {current_sma200:.3f} JPY")
    print(f"RSI (14): {current_rsi:.1f}")
    
    # Technical signals
    print("\nTechnical Signals:")
    if current_price > current_sma200:
        print("• Price is ABOVE 200 SMA (Bullish trend)")
    else:
        print("• Price is BELOW 200 SMA (Bearish trend)")
    
    if current_rsi > 70:
        print("• RSI indicates OVERBOUGHT conditions")
    elif current_rsi < 30:
        print("• RSI indicates OVERSOLD conditions")
    else:
        print("• RSI in neutral range")
    
    print("="*60)
    
    return fig, df

# Alternative version using plotly for interactive charts
def create_interactive_chart():
    """Create interactive chart using plotly"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Generate data
        df = generate_forex_data()
        df['SMA_200'] = calculate_sma(df, 200)
        df['RSI'] = calculate_rsi(df)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('BRL/JPY Price Chart', 'RSI (14)'),
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='BRL/JPY',
                increasing_line_color='#00C851',
                decreasing_line_color='#FF4444'
            ),
            row=1, col=1
        )
        
        # Add 200 SMA
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA_200'],
                mode='lines',
                name='200 SMA',
                line=dict(color='#FF6F00', width=2)
            ),
            row=1, col=1
        )
        
        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['RSI'],
                mode='lines',
                name='RSI (14)',
                line=dict(color='#7B1FA2', width=2)
            ),
            row=2, col=1
        )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="#D32F2F", 
                     annotation_text="Overbought", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#388E3C", 
                     annotation_text="Oversold", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title='BRL/JPY Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price (JPY)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        
        return fig, df
        
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        return None, None

# Main execution
if __name__ == "__main__":
    print("Creating BRL/JPY Candlestick Chart with 200DMA and RSI...")
    
    # Create matplotlib chart
    fig, df = create_brljpy_chart()
    plt.show()
    
    # Optionally create interactive plotly chart
    print("\nCreating interactive chart (requires plotly)...")
    plotly_fig, _ = create_interactive_chart()
    if plotly_fig:
        plotly_fig.show()
    
    # Save the matplotlib chart
    fig.savefig('brljpy_technical_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nChart saved as 'brljpy_technical_analysis.png'")
    
    # Save data to CSV
    df.to_csv('brljpy_data.csv', index=False)
    print(f"Data saved as 'brljpy_data.csv'")

# Required packages (install with pip):
"""
pandas
numpy
matplotlib
seaborn
plotly (optional, for interactive charts)
"""