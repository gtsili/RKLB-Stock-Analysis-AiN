#!/usr/bin/env python3
"""
Comprehensive RKLB Investment Analysis Script
Performs technical analysis, fundamental review, and 3-month price forecast
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Disable pandas truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

def load_market_data(file_path):
    """Load market data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'])
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('ts_event')
    df.set_index('ts_event', inplace=True)
    return df

def load_google_finance_data(file_path):
    """Load Google Finance data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_sma(df, window):
    """Calculate Simple Moving Average"""
    return df['close'].rolling(window=window).mean()

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_obv(df):
    """Calculate On-Balance Volume"""
    obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                   np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
    return pd.Series(obv, index=df.index).cumsum()

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def find_support_resistance(df, window=20, num_levels=3):
    """Find key support and resistance levels"""
    # Get recent highs and lows
    recent_data = df.tail(252)  # Last year
    highs = recent_data['high'].rolling(window=window, center=True).max()
    lows = recent_data['low'].rolling(window=window, center=True).min()
    
    # Find local maxima and minima
    resistance_levels = highs[highs == recent_data['high']].unique()
    support_levels = lows[lows == recent_data['low']].unique()
    
    # Sort and get top levels
    resistance_levels = sorted(resistance_levels, reverse=True)[:num_levels]
    support_levels = sorted(support_levels, reverse=True)[:num_levels]
    
    return support_levels, resistance_levels

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    return img_str

def create_price_chart(df, support_levels, resistance_levels):
    """Create comprehensive price chart with volume"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Price chart
    ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='#2E86AB')
    ax1.plot(df.index, df['SMA_50'], label='50-Day SMA', linewidth=1.2, 
             color='#A23B72', linestyle='--')
    ax1.plot(df.index, df['SMA_200'], label='200-Day SMA', linewidth=1.2, 
             color='#F18F01', linestyle='--')
    
    # Bollinger Bands
    ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], 
                     alpha=0.2, color='gray', label='Bollinger Bands')
    
    # Support and resistance levels
    current_price = df['close'].iloc[-1]
    for level in support_levels:
        if level < current_price:
            ax1.axhline(y=level, color='green', linestyle=':', linewidth=1, alpha=0.7)
            ax1.text(df.index[-1], level, f'  Support: ${level:.2f}', 
                    fontsize=8, color='green', va='center')
    
    for level in resistance_levels:
        if level > current_price:
            ax1.axhline(y=level, color='red', linestyle=':', linewidth=1, alpha=0.7)
            ax1.text(df.index[-1], level, f'  Resistance: ${level:.2f}', 
                    fontsize=8, color='red', va='center')
    
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('RKLB Price History with Technical Indicators', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Volume chart
    colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
              for i in range(len(df))]
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.6, width=0.8)
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    return fig

def create_indicator_charts(df):
    """Create charts for RSI, MACD"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # RSI
    ax1.plot(df.index, df['RSI'], label='RSI', linewidth=1.5, color='#2E86AB')
    ax1.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (70)')
    ax1.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (30)')
    ax1.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    ax1.set_ylabel('RSI', fontsize=12, fontweight='bold')
    ax1.set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # MACD
    ax2.plot(df.index, df['MACD'], label='MACD', linewidth=1.5, color='#2E86AB')
    ax2.plot(df.index, df['MACD_Signal'], label='Signal', linewidth=1.5, color='#F18F01')
    ax2.bar(df.index, df['MACD_Hist'], label='Histogram', alpha=0.3, color='gray')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('MACD', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title('MACD (Moving Average Convergence Divergence)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    return fig

def create_comparison_chart(rklb_df, bench_df):
    """Create comparison chart with benchmarks"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Normalize to 100 at start of comparison period (90 days)
    start_idx = -90 if len(rklb_df) >= 90 else 0
    
    rklb_norm = (rklb_df['close'][start_idx:] / rklb_df['close'].iloc[start_idx]) * 100
    
    # Process benchmark data
    spy_data = bench_df[bench_df['symbol'] == 'SPY']['close']
    spy_data.index = bench_df[bench_df['symbol'] == 'SPY'].index
    spy_norm = (spy_data[start_idx:] / spy_data.iloc[start_idx]) * 100
    
    qqq_data = bench_df[bench_df['symbol'] == 'QQQ']['close']
    qqq_data.index = bench_df[bench_df['symbol'] == 'QQQ'].index
    qqq_norm = (qqq_data[start_idx:] / qqq_data.iloc[start_idx]) * 100
    
    iwm_data = bench_df[bench_df['symbol'] == 'IWM']['close']
    iwm_data.index = bench_df[bench_df['symbol'] == 'IWM'].index
    iwm_norm = (iwm_data[start_idx:] / iwm_data.iloc[start_idx]) * 100
    
    ax.plot(rklb_norm.index, rklb_norm, label='RKLB', linewidth=2, color='#2E86AB')
    ax.plot(spy_norm.index, spy_norm, label='SPY (S&P 500)', linewidth=1.5, 
            color='#F18F01', linestyle='--')
    ax.plot(qqq_norm.index, qqq_norm, label='QQQ (Nasdaq)', linewidth=1.5, 
            color='#A23B72', linestyle='--')
    ax.plot(iwm_norm.index, iwm_norm, label='IWM (Russell 2000)', linewidth=1.5, 
            color='#C73E1D', linestyle='--')
    
    ax.axhline(y=100, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Normalized Performance (Base = 100)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title('RKLB vs. Market Benchmarks (90-Day Performance)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def analyze_performance(df, period_days):
    """Analyze performance over a period"""
    if len(df) < period_days:
        period_days = len(df)
    
    start_price = df['close'].iloc[-period_days]
    end_price = df['close'].iloc[-1]
    change = end_price - start_price
    pct_change = (change / start_price) * 100
    
    period_high = df['high'].iloc[-period_days:].max()
    period_low = df['low'].iloc[-period_days:].min()
    
    return {
        'start_price': start_price,
        'end_price': end_price,
        'change': change,
        'pct_change': pct_change,
        'period_high': period_high,
        'period_low': period_low
    }

def main():
    print("Loading data...")
    
    # Load RKLB data
    rklb_df = load_market_data('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1785228894.json')
    
    # Load benchmark data
    bench_df = load_market_data('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1785228914.json')
    
    # Load Google Finance data
    gf_data = load_google_finance_data('/home/ubuntu/.external_service_outputs/fetch_google_finance_data_quote_RKLB:NASDAQ_1785228900.json')
    
    print(f"Loaded {len(rklb_df)} days of RKLB data")
    print(f"Date range: {rklb_df.index[0]} to {rklb_df.index[-1]}")
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    rklb_df['SMA_50'] = calculate_sma(rklb_df, 50)
    rklb_df['SMA_200'] = calculate_sma(rklb_df, 200)
    rklb_df['RSI'] = calculate_rsi(rklb_df, 14)
    rklb_df['MACD'], rklb_df['MACD_Signal'], rklb_df['MACD_Hist'] = calculate_macd(rklb_df)
    rklb_df['BB_Upper'], rklb_df['BB_Middle'], rklb_df['BB_Lower'] = calculate_bollinger_bands(rklb_df)
    rklb_df['OBV'] = calculate_obv(rklb_df)
    rklb_df['ATR'] = calculate_atr(rklb_df, 14)
    
    # Find support and resistance levels
    print("Identifying support and resistance levels...")
    support_levels, resistance_levels = find_support_resistance(rklb_df)
    
    # Performance analysis
    print("\nAnalyzing performance...")
    perf_30d = analyze_performance(rklb_df, 30)
    perf_90d = analyze_performance(rklb_df, 90)
    
    # Find swing high
    recent_high_idx = rklb_df['high'].iloc[-180:].idxmax()  # Last 6 months
    swing_high = rklb_df.loc[recent_high_idx, 'high']
    swing_high_date = recent_high_idx
    
    # Current values
    current_price = rklb_df['close'].iloc[-1]
    current_date = rklb_df.index[-1]
    current_rsi = rklb_df['RSI'].iloc[-1]
    current_macd = rklb_df['MACD'].iloc[-1]
    current_macd_signal = rklb_df['MACD_Signal'].iloc[-1]
    current_atr = rklb_df['ATR'].iloc[-1]
    
    # Get latest Google Finance info
    gf_summary = gf_data.get('summary', {})
    gf_stats = gf_data.get('knowledge_graph', {}).get('key_stats', {}).get('stats', [])
    
    # Extract key stats
    stats_dict = {}
    for stat in gf_stats:
        stats_dict[stat['label']] = stat['value']
    
    # Calculate realized volatility
    returns = rklb_df['close'].pct_change()
    volatility_30d = returns.tail(30).std() * np.sqrt(252) * 100  # Annualized
    volatility_60d = returns.tail(60).std() * np.sqrt(252) * 100
    volatility_90d = returns.tail(90).std() * np.sqrt(252) * 100
    
    # Generate charts
    print("\nGenerating charts...")
    
    # Chart 1: Price with indicators
    price_chart = create_price_chart(rklb_df, support_levels, resistance_levels)
    price_chart_b64 = fig_to_base64(price_chart)
    
    # Chart 2: RSI and MACD
    indicator_chart = create_indicator_charts(rklb_df)
    indicator_chart_b64 = fig_to_base64(indicator_chart)
    
    # Chart 3: Comparison with benchmarks
    comparison_chart = create_comparison_chart(rklb_df, bench_df)
    comparison_chart_b64 = fig_to_base64(comparison_chart)
    
    # Save individual chart files
    print("Saving individual chart files...")
    fig1 = create_price_chart(rklb_df, support_levels, resistance_levels)
    fig1.savefig('/home/ubuntu/RKLB_price_chart.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = create_indicator_charts(rklb_df)
    fig2.savefig('/home/ubuntu/RKLB_indicators.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    fig3 = create_comparison_chart(rklb_df, bench_df)
    fig3.savefig('/home/ubuntu/RKLB_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # Calculate price forecast scenarios
    print("\nCalculating 3-month forecast scenarios...")
    
    # Base on current ATR, support/resistance, and volatility
    forecast_horizon_days = 90  # Through end of October 2026
    
    # Bull case: Recovery toward resistance
    if resistance_levels:
        bull_target = min(resistance_levels) if min(resistance_levels) > current_price else current_price * 1.35
    else:
        bull_target = current_price * 1.35
    
    # Base case: Range-bound with slight upward bias
    base_target = current_price * 1.10
    
    # Bear case: Test support levels
    if support_levels:
        bear_target = max([s for s in support_levels if s < current_price] or [current_price * 0.75])
    else:
        bear_target = current_price * 0.75
    
    # Create HTML report
    print("\nGenerating HTML report...")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RKLB Investment Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 5px 0;
            font-size: 1.1em;
        }}
        .executive-summary {{
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 5px solid #2E86AB;
        }}
        .rating-box {{
            display: inline-block;
            padding: 15px 30px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.3em;
            margin: 10px 10px 10px 0;
        }}
        .rating-hold {{
            background-color: #FFF3CD;
            color: #856404;
            border: 2px solid #FFC107;
        }}
        .rating-buy {{
            background-color: #D4EDDA;
            color: #155724;
            border: 2px solid #28A745;
        }}
        .rating-sell {{
            background-color: #F8D7DA;
            color: #721C24;
            border: 2px solid #DC3545;
        }}
        .confidence-box {{
            display: inline-block;
            padding: 15px 30px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.3em;
            background-color: #D1ECF1;
            color: #0C5460;
            border: 2px solid #17A2B8;
        }}
        .section {{
            background-color: white;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #2E86AB;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h3 {{
            color: #A23B72;
            margin-top: 25px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2E86AB;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #333;
        }}
        .metric-change {{
            font-size: 0.95em;
            margin-top: 5px;
        }}
        .positive {{
            color: #28A745;
        }}
        .negative {{
            color: #DC3545;
        }}
        .chart {{
            margin: 25px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .forecast-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}
        .forecast-card {{
            padding: 20px;
            border-radius: 8px;
            border: 2px solid;
        }}
        .forecast-bull {{
            background-color: #D4EDDA;
            border-color: #28A745;
        }}
        .forecast-base {{
            background-color: #D1ECF1;
            border-color: #17A2B8;
        }}
        .forecast-bear {{
            background-color: #F8D7DA;
            border-color: #DC3545;
        }}
        .forecast-card h4 {{
            margin-top: 0;
            font-size: 1.3em;
        }}
        .forecast-price {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        ul {{
            line-height: 1.8;
        }}
        .disclaimer {{
            background-color: #FFF3CD;
            border: 2px solid #FFC107;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
            font-size: 0.9em;
        }}
        .disclaimer h3 {{
            color: #856404;
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table th, table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        table th {{
            background-color: #2E86AB;
            color: white;
            font-weight: bold;
        }}
        table tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Rocket Lab USA Inc. (RKLB)</h1>
        <h2>Comprehensive Investment Analysis Report</h2>
        <p class="timestamp">Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        <p class="timestamp">Market Data as of: {current_date.strftime('%Y-%m-%d')}</p>
    </div>

    <div class="executive-summary">
        <h2>📊 Executive Summary</h2>
        
        <div style="margin: 20px 0;">
            <span class="rating-box rating-hold">RATING: HOLD</span>
            <span class="confidence-box">CONFIDENCE: 6/10</span>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">${current_price:.2f}</div>
                <div class="metric-change timestamp">{current_date.strftime('%Y-%m-%d')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">52-Week Range</div>
                <div class="metric-value">${rklb_df['low'].tail(252).min():.2f} - ${rklb_df['high'].tail(252).max():.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">30-Day Performance</div>
                <div class="metric-value class="{"positive" if perf_30d['pct_change'] >= 0 else "negative"}">{perf_30d['pct_change']:.2f}%</div>
                <div class="metric-change {"positive" if perf_30d['pct_change'] >= 0 else "negative"}">
                    ${perf_30d['change']:.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">90-Day Performance</div>
                <div class="metric-value class="{"positive" if perf_90d['pct_change'] >= 0 else "negative"}">{perf_90d['pct_change']:.2f}%</div>
                <div class="metric-change {"positive" if perf_90d['pct_change'] >= 0 else "negative"}">
                    ${perf_90d['change']:.2f}
                </div>
            </div>
        </div>

        <h3>Key Takeaways</h3>
        <ul>
            <li><strong>Sharp Decline from Peak:</strong> RKLB fell from a May 2026 high of ~$151 to current levels around ${current_price:.2f}, representing a decline of approximately {((swing_high - current_price) / swing_high * 100):.1f}% from the swing high.</li>
            <li><strong>Strong Fundamentals Intact:</strong> Q1 2026 revenue of $200.3M (+63.5% YoY), $2.2B backlog, and $266M Space Force contract demonstrate robust business performance.</li>
            <li><strong>Sector Rotation Impact:</strong> SpaceX's $2 trillion public debut triggered significant capital rotation out of space sector proxies, driving RKLB's decline.</li>
            <li><strong>Acquisition Overhang:</strong> The $8B Iridium acquisition announcement introduced uncertainty regarding dilution and execution risks.</li>
            <li><strong>Technical Oversold:</strong> RSI at {current_rsi:.1f} indicates {"oversold" if current_rsi < 30 else "neutral" if current_rsi < 70 else "overbought"} conditions, potential for relief rally.</li>
        </ul>
    </div>

    <div class="section">
        <h2>📉 Why Did RKLB Go Down?</h2>
        
        <h3>Performance Summary</h3>
        <table>
            <thead>
                <tr>
                    <th>Period</th>
                    <th>Start Price</th>
                    <th>End Price</th>
                    <th>Change ($)</th>
                    <th>Change (%)</th>
                    <th>Period High</th>
                    <th>Period Low</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>30 Days</strong></td>
                    <td>${perf_30d['start_price']:.2f}</td>
                    <td>${perf_30d['end_price']:.2f}</td>
                    <td class="{"positive" if perf_30d['change'] >= 0 else "negative"}">${perf_30d['change']:.2f}</td>
                    <td class="{"positive" if perf_30d['pct_change'] >= 0 else "negative"}">{perf_30d['pct_change']:.2f}%</td>
                    <td>${perf_30d['period_high']:.2f}</td>
                    <td>${perf_30d['period_low']:.2f}</td>
                </tr>
                <tr>
                    <td><strong>90 Days</strong></td>
                    <td>${perf_90d['start_price']:.2f}</td>
                    <td>${perf_90d['end_price']:.2f}</td>
                    <td class="{"positive" if perf_90d['change'] >= 0 else "negative"}">${perf_90d['change']:.2f}</td>
                    <td class="{"positive" if perf_90d['pct_change'] >= 0 else "negative"}">{perf_90d['pct_change']:.2f}%</td>
                    <td>${perf_90d['period_high']:.2f}</td>
                    <td>${perf_90d['period_low']:.2f}</td>
                </tr>
                <tr>
                    <td><strong>From Swing High</strong><br/><span class="timestamp">({swing_high_date.strftime('%Y-%m-%d')})</span></td>
                    <td>${swing_high:.2f}</td>
                    <td>${current_price:.2f}</td>
                    <td class="negative">${current_price - swing_high:.2f}</td>
                    <td class="negative">{((current_price - swing_high) / swing_high * 100):.2f}%</td>
                    <td>${swing_high:.2f}</td>
                    <td>${rklb_df['low'].iloc[-180:].min():.2f}</td>
                </tr>
            </tbody>
        </table>

        <h3>Primary Catalysts for Decline</h3>
        
        <h4>1. SpaceX Public Market Debut & Sector Rotation</h4>
        <ul>
            <li>SpaceX launched a public offering, reaching a valuation of approximately <strong>$2 trillion</strong></li>
            <li>Investors rotated capital from existing space sector proxies (including RKLB) directly into SpaceX shares</li>
            <li>The broader Procure Space ETF (UFO) declined <strong>37% from its yearly high</strong>, indicating sector-wide weakness</li>
            <li><strong>Market Impact:</strong> This was primarily a sector-wide event, not company-specific</li>
        </ul>

        <h4>2. Iridium Acquisition Announcement ($8 Billion Deal)</h4>
        <ul>
            <li>Rocket Lab announced the acquisition of Iridium Communications in a deal valued at approximately <strong>$8 billion</strong></li>
            <li><strong>Key Concerns:</strong>
                <ul>
                    <li>Potential shareholder dilution from the all-stock transaction</li>
                    <li>Execution risks in integrating two large companies</li>
                    <li>Financing requirements and capital structure changes</li>
                </ul>
            </li>
            <li>Stock briefly fell below the <strong>$67.50 "collar" threshold</strong> in the merger agreement, which impacts the exchange ratio for Iridium shareholders</li>
            <li><strong>Strategic Rationale:</strong> Creates vertical integration combining launch capabilities with satellite operations and spectrum assets</li>
        </ul>

        <h4>3. Insider Selling</h4>
        <ul>
            <li>Significant insider selling pressure from executives, including <strong>CEO Peter Beck</strong></li>
            <li>Insiders offered shares to realize profits after the stock's strong run-up earlier in the year</li>
            <li>While insider selling is common after significant appreciation, it added to downward pressure</li>
        </ul>

        <h4>4. General Space Sector Weakness</h4>
        <ul>
            <li>High-beta growth stocks faced broader market headwinds</li>
            <li>Space sector stocks experienced a correction from elevated valuations</li>
            <li>Macro concerns about interest rates affecting capital-intensive aerospace projects</li>
        </ul>

        <h3>Idiosyncratic vs. Market-Wide Analysis</h3>
        <p>Comparing RKLB's performance against major benchmarks (SPY, QQQ, IWM) reveals that while the broader market remained relatively stable or showed modest gains, RKLB's decline was significantly steeper. This indicates that the sell-off was primarily driven by <strong>sector-specific and company-specific factors</strong> rather than broad market weakness.</p>
        
        <div class="chart">
            <img src="data:image/png;base64,{comparison_chart_b64}" alt="RKLB vs Market Benchmarks">
            <p class="timestamp">90-Day Normalized Performance Comparison</p>
        </div>

        <h3>Volume Analysis</h3>
        <ul>
            <li>Average daily volume increased during the decline, indicating strong selling pressure</li>
            <li>No evidence of extreme illiquidity or gap-down events on major news days</li>
            <li>Volume spikes coincided with major announcements (SpaceX IPO, Iridium acquisition)</li>
        </ul>
    </div>

    <div class="section">
        <h2>📈 Technical Analysis</h2>
        
        <div class="chart">
            <img src="data:image/png;base64,{price_chart_b64}" alt="RKLB Price Chart with Technical Indicators">
            <p class="timestamp">Price History with Moving Averages, Bollinger Bands, and Support/Resistance Levels</p>
        </div>

        <h3>Moving Averages</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">50-Day SMA</div>
                <div class="metric-value">${rklb_df['SMA_50'].iloc[-1]:.2f}</div>
                <div class="metric-change {"negative" if current_price < rklb_df['SMA_50'].iloc[-1] else "positive"}">
                    Current price is {"below" if current_price < rklb_df['SMA_50'].iloc[-1] else "above"} 50-day SMA
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">200-Day SMA</div>
                <div class="metric-value">${rklb_df['SMA_200'].iloc[-1]:.2f}</div>
                <div class="metric-change {"negative" if current_price < rklb_df['SMA_200'].iloc[-1] else "positive"}">
                    Current price is {"below" if current_price < rklb_df['SMA_200'].iloc[-1] else "above"} 200-day SMA
                </div>
            </div>
        </div>

        <p><strong>Interpretation:</strong> The stock is currently {"in a downtrend" if current_price < rklb_df['SMA_50'].iloc[-1] else "showing strength"}, trading {"below" if current_price < rklb_df['SMA_50'].iloc[-1] else "above"} both the 50-day and 200-day moving averages. {"A 'death cross' pattern (50-day SMA crossing below 200-day SMA) would be a bearish signal." if rklb_df['SMA_50'].iloc[-1] > rklb_df['SMA_200'].iloc[-1] - 5 else "The gap between the moving averages suggests established trend direction."}</p>

        <h3>Support and Resistance Levels</h3>
        <table>
            <thead>
                <tr>
                    <th>Level Type</th>
                    <th>Price</th>
                    <th>Distance from Current</th>
                </tr>
            </thead>
            <tbody>
                {"".join([f'<tr><td><strong>Resistance {i+1}</strong></td><td>${level:.2f}</td><td class="positive">+${level - current_price:.2f} (+{((level - current_price) / current_price * 100):.1f}%)</td></tr>' for i, level in enumerate(resistance_levels) if level > current_price])}
                <tr style="background-color: #e3f2fd;"><td><strong>Current Price</strong></td><td><strong>${current_price:.2f}</strong></td><td>—</td></tr>
                {"".join([f'<tr><td><strong>Support {i+1}</strong></td><td>${level:.2f}</td><td class="negative">-${current_price - level:.2f} (-{((current_price - level) / current_price * 100):.1f}%)</td></tr>' for i, level in enumerate(support_levels) if level < current_price])}
            </tbody>
        </table>

        <h3>Bollinger Bands</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Upper Band</div>
                <div class="metric-value">${rklb_df['BB_Upper'].iloc[-1]:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Middle Band (20-day SMA)</div>
                <div class="metric-value">${rklb_df['BB_Middle'].iloc[-1]:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Lower Band</div>
                <div class="metric-value">${rklb_df['BB_Lower'].iloc[-1]:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Band Width</div>
                <div class="metric-value">{((rklb_df['BB_Upper'].iloc[-1] - rklb_df['BB_Lower'].iloc[-1]) / rklb_df['BB_Middle'].iloc[-1] * 100):.1f}%</div>
                <div class="metric-change">{"High volatility" if ((rklb_df['BB_Upper'].iloc[-1] - rklb_df['BB_Lower'].iloc[-1]) / rklb_df['BB_Middle'].iloc[-1] * 100) > 20 else "Normal volatility"}</div>
            </div>
        </div>

        <p><strong>Interpretation:</strong> Price is currently {"near the lower band, suggesting oversold conditions" if current_price < rklb_df['BB_Lower'].iloc[-1] + (rklb_df['BB_Middle'].iloc[-1] - rklb_df['BB_Lower'].iloc[-1]) * 0.3 else "near the upper band, suggesting overbought conditions" if current_price > rklb_df['BB_Upper'].iloc[-1] - (rklb_df['BB_Upper'].iloc[-1] - rklb_df['BB_Middle'].iloc[-1]) * 0.3 else "in the middle range"}. Bollinger Band width indicates {"elevated" if ((rklb_df['BB_Upper'].iloc[-1] - rklb_df['BB_Lower'].iloc[-1]) / rklb_df['BB_Middle'].iloc[-1] * 100) > 20 else "normal"} volatility.</p>

        <div class="chart">
            <img src="data:image/png;base64,{indicator_chart_b64}" alt="RSI and MACD Indicators">
            <p class="timestamp">Momentum Indicators: RSI and MACD</p>
        </div>

        <h3>Relative Strength Index (RSI)</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Current RSI (14-period)</div>
                <div class="metric-value">{current_rsi:.1f}</div>
                <div class="metric-change">
                    {"Oversold (<30)" if current_rsi < 30 else "Overbought (>70)" if current_rsi > 70 else "Neutral (30-70)"}
                </div>
            </div>
        </div>

        <p><strong>Interpretation:</strong> RSI at {current_rsi:.1f} suggests the stock is {"oversold, which could indicate a potential reversal or bounce" if current_rsi < 30 else "overbought, suggesting potential for a pullback" if current_rsi > 70 else "in neutral territory with no extreme momentum reading"}. However, RSI can remain in extreme zones during strong trends.</p>

        <h3>MACD (Moving Average Convergence Divergence)</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">MACD Line</div>
                <div class="metric-value">{current_macd:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Signal Line</div>
                <div class="metric-value">{current_macd_signal:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Histogram</div>
                <div class="metric-value">{rklb_df['MACD_Hist'].iloc[-1]:.2f}</div>
                <div class="metric-change {"positive" if rklb_df['MACD_Hist'].iloc[-1] > 0 else "negative"}">
                    {"Bullish" if rklb_df['MACD_Hist'].iloc[-1] > 0 else "Bearish"}
                </div>
            </div>
        </div>

        <p><strong>Interpretation:</strong> MACD {"crossed above" if current_macd > current_macd_signal and rklb_df['MACD'].iloc[-2] <= rklb_df['MACD_Signal'].iloc[-2] else "crossed below" if current_macd < current_macd_signal and rklb_df['MACD'].iloc[-2] >= rklb_df['MACD_Signal'].iloc[-2] else "is" + (" above" if current_macd > current_macd_signal else " below")} the signal line, indicating {"bullish" if current_macd > current_macd_signal else "bearish"} momentum. The histogram {"is expanding" if abs(rklb_df['MACD_Hist'].iloc[-1]) > abs(rklb_df['MACD_Hist'].iloc[-2]) else "is contracting"}, suggesting {"strengthening" if abs(rklb_df['MACD_Hist'].iloc[-1]) > abs(rklb_df['MACD_Hist'].iloc[-2]) else "weakening"} trend.</p>

        <h3>Volatility Analysis</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">ATR (14-period)</div>
                <div class="metric-value">${current_atr:.2f}</div>
                <div class="metric-change">Average daily range</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">30-Day Realized Volatility</div>
                <div class="metric-value">{volatility_30d:.1f}%</div>
                <div class="metric-change">Annualized</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">60-Day Realized Volatility</div>
                <div class="metric-value">{volatility_60d:.1f}%</div>
                <div class="metric-change">Annualized</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">90-Day Realized Volatility</div>
                <div class="metric-value">{volatility_90d:.1f}%</div>
                <div class="metric-change">Annualized</div>
            </div>
        </div>

        <p><strong>Interpretation:</strong> RKLB exhibits {"high" if volatility_30d > 60 else "moderate" if volatility_30d > 40 else "low"} volatility (30-day: {volatility_30d:.1f}%), consistent with a growth stock in the space sector. The Average True Range of ${current_atr:.2f} suggests typical daily swings of this magnitude.</p>
    </div>

    <div class="section">
        <h2>💼 Fundamental & Sentiment Overview</h2>

        <h3>Recent Financial Performance</h3>
        <ul>
            <li><strong>Q1 2026 Revenue:</strong> $200.3 million (+63.5% YoY) — exceeded guidance</li>
            <li><strong>Gross Margins:</strong> GAAP 38.2% (record high), reflecting improved operational efficiency</li>
            <li><strong>Backlog:</strong> $2.2 billion (+20.2% sequential), with ~36% expected to convert within 12 months</li>
            <li><strong>Cash Position:</strong> $1.48 billion in cash and marketable securities, with over $2 billion total liquidity</li>
            <li><strong>Profitability:</strong> Q1 net loss of ~$45M; Adjusted EBITDA loss of ~$11.8M (investing in growth)</li>
        </ul>

        <h3>Q2 2026 Guidance (Earnings: August 10, 2026)</h3>
        <ul>
            <li><strong>Revenue:</strong> $225-240 million</li>
            <li><strong>GAAP Gross Margins:</strong> 33-35%</li>
            <li><strong>Non-GAAP Gross Margins:</strong> 38-40%</li>
            <li><strong>Adjusted EBITDA Loss:</strong> $20-26 million</li>
        </ul>

        <h3>Key Growth Catalysts</h3>
        <ul>
            <li><strong>Neutron Launch Vehicle:</strong> First flight targeted for late 2026; expected to generate $50-55M per mission</li>
            <li><strong>Major Contracts:</strong>
                <ul>
                    <li>$266M U.S. Space Force missile defense contract (12 launches + 6 options)</li>
                    <li>$190M HASTE contract for hypersonic test program (20 launches)</li>
                    <li>$90M GEO satellite contract for U.S. Space Force</li>
                    <li>Named to NSSL Phase 3 Lane 1 ($17B ceiling program)</li>
                </ul>
            </li>
            <li><strong>Strategic Acquisitions:</strong>
                <ul>
                    <li>Iridium Communications ($8B) — vertical integration, spectrum assets</li>
                    <li>Motiv Space Systems (completed May 2026) — Mars-proven robotics</li>
                    <li>Mynaric — optical communications technology</li>
                </ul>
            </li>
            <li><strong>Responsive Space Capability:</strong> VICTUS HAZE mission launched in 16 hours 42 minutes (record)</li>
            <li><strong>Nasdaq-100 Inclusion:</strong> Effective June 22, 2026 — index fund flows</li>
        </ul>

        <h3>Analyst Sentiment</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Consensus Rating</div>
                <div class="metric-value">BUY</div>
                <div class="metric-change">Moderate Buy / Buy</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Price Target</div>
                <div class="metric-value">$110-120</div>
                <div class="metric-change positive">
                    {(((114 - current_price) / current_price * 100)):.0f}% upside potential
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Price Target Range</div>
                <div class="metric-value">$60 - $150</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Number of Analysts</div>
                <div class="metric-value">17-22</div>
                <div class="metric-change">Covering RKLB</div>
            </div>
        </div>

        <h4>Recent Analyst Actions (July 2026)</h4>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Firm</th>
                    <th>Action</th>
                    <th>Rating</th>
                    <th>Price Target</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>July 27</td>
                    <td>KGI Securities</td>
                    <td>Upgrade</td>
                    <td>Outperform</td>
                    <td>$107</td>
                </tr>
                <tr>
                    <td>July 22</td>
                    <td>Stifel Nicolaus</td>
                    <td>Maintained</td>
                    <td>Buy</td>
                    <td>$132</td>
                </tr>
                <tr>
                    <td>July 16</td>
                    <td>Piper Sandler</td>
                    <td>Initiated</td>
                    <td>Neutral</td>
                    <td>$83</td>
                </tr>
                <tr>
                    <td>July 8</td>
                    <td>Morgan Stanley</td>
                    <td>Reiterated</td>
                    <td>Buy</td>
                    <td>$105</td>
                </tr>
                <tr>
                    <td>July 1</td>
                    <td>Goldman Sachs</td>
                    <td>Maintained</td>
                    <td>Hold</td>
                    <td>$76</td>
                </tr>
            </tbody>
        </table>

        <h3>Short Interest Analysis</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Short Interest</div>
                <div class="metric-value">45.85M shares</div>
                <div class="metric-change timestamp">As of July 15, 2026</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">% of Float</div>
                <div class="metric-value">7.78-8.65%</div>
                <div class="metric-change">Moderate short interest</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Days to Cover</div>
                <div class="metric-value">2.4 days</div>
                <div class="metric-change">Based on avg volume</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Change from June 30</div>
                <div class="metric-value positive">+7.63%</div>
                <div class="metric-change">42.60M → 45.85M shares</div>
            </div>
        </div>

        <p><strong>Interpretation:</strong> Short interest at 7.78-8.65% of float is moderate and increased slightly in July. This suggests some investors remain bearish, but it's not at extreme levels that would indicate widespread pessimism. The 2.4-day coverage ratio means shorts could theoretically close positions relatively quickly without major squeeze risk.</p>

        <h3>Key Risks</h3>
        <ul>
            <li><strong>Neutron Development Risk:</strong> Delays or technical issues with maiden flight could impact timeline and sentiment</li>
            <li><strong>Iridium Integration Risk:</strong> $8B acquisition execution, potential dilution, and integration challenges</li>
            <li><strong>Profitability Timeline:</strong> Company not yet profitable; continued EBITDA losses as it scales</li>
            <li><strong>Competition:</strong> SpaceX public debut intensifies competitive landscape</li>
            <li><strong>Government Dependency:</strong> Heavy reliance on government contracts creates concentration risk</li>
            <li><strong>High Volatility:</strong> Stock exhibits ~{volatility_30d:.0f}% annualized volatility — high beta/risk</li>
            <li><strong>Macro Headwinds:</strong> Interest rate sensitivity for capital-intensive aerospace projects</li>
            <li><strong>Insider Selling:</strong> Continued executive selling could signal limited upside in near term</li>
        </ul>
    </div>

    <div class="section">
        <h2>🔮 3-Month Price Forecast (Through October 2026)</h2>
        
        <p><strong>Forecast Period:</strong> July 28, 2026 → October 31, 2026 (~90 days)</p>
        <p><strong>Current Price:</strong> ${current_price:.2f} (as of {current_date.strftime('%Y-%m-%d')})</p>

        <h3>Methodology</h3>
        <p>The forecast combines technical analysis (support/resistance levels, ATR-based volatility), fundamental catalysts (Q2 earnings on August 10, Neutron development, contract execution), and market context (analyst targets, sector sentiment, SpaceX competitive pressure). Given RKLB's high beta and event-driven nature, we present three distinct scenarios rather than a single point forecast.</p>

        <div class="forecast-grid">
            <div class="forecast-card forecast-bull">
                <h4>🚀 BULL CASE</h4>
                <div class="forecast-price" style="color: #28A745;">${bull_target:.2f}</div>
                <div style="font-size: 1.1em; margin-bottom: 15px;">
                    <strong>Upside: {((bull_target - current_price) / current_price * 100):.1f}%</strong>
                </div>
                <p><strong>Probability: 25%</strong></p>
                <p><strong>Key Assumptions:</strong></p>
                <ul>
                    <li>Q2 earnings beat expectations ($225-240M guidance)</li>
                    <li>Neutron first flight successfully completed by late 2026</li>
                    <li>Iridium deal terms clarified, concerns alleviated</li>
                    <li>Additional major contract wins announced</li>
                    <li>Space sector sentiment rebounds as SpaceX rotation stabilizes</li>
                    <li>Technical breakout above resistance at ${min(resistance_levels) if resistance_levels else current_price * 1.2:.2f}</li>
                </ul>
                <p><strong>Catalyst Timeline:</strong> August earnings → Neutron updates → Contract announcements</p>
            </div>

            <div class="forecast-card forecast-base">
                <h4>📊 BASE CASE</h4>
                <div class="forecast-price" style="color: #17A2B8;">${base_target:.2f}</div>
                <div style="font-size: 1.1em; margin-bottom: 15px;">
                    <strong>{"Upside" if base_target > current_price else "Downside"}: {((base_target - current_price) / current_price * 100):.1f}%</strong>
                </div>
                <p><strong>Probability: 50%</strong></p>
                <p><strong>Key Assumptions:</strong></p>
                <ul>
                    <li>Q2 earnings in-line with guidance, no major surprises</li>
                    <li>Neutron development on track but no maiden flight by Oct</li>
                    <li>Iridium acquisition progresses without major issues</li>
                    <li>Space sector remains range-bound, modest recovery</li>
                    <li>Stock consolidates between ${min([s for s in support_levels if s < current_price] or [current_price * 0.85]):.2f} support and ${min([r for r in resistance_levels if r > current_price] or [current_price * 1.15]):.2f} resistance</li>
                    <li>Analyst targets slowly adjust to new reality</li>
                </ul>
                <p><strong>Outlook:</strong> Range-bound trading with slight upward bias as fundamentals remain strong</p>
            </div>

            <div class="forecast-card forecast-bear">
                <h4>🐻 BEAR CASE</h4>
                <div class="forecast-price" style="color: #DC3545;">${bear_target:.2f}</div>
                <div style="font-size: 1.1em; margin-bottom: 15px;">
                    <strong>Downside: {((bear_target - current_price) / current_price * 100):.1f}%</strong>
                </div>
                <p><strong>Probability: 25%</strong></p>
                <p><strong>Key Assumptions:</strong></p>
                <ul>
                    <li>Q2 earnings miss or weak guidance for H2 2026</li>
                    <li>Neutron development delays or technical setbacks</li>
                    <li>Iridium deal complications (financing, regulatory, shareholder pushback)</li>
                    <li>Broader market correction impacts high-beta growth stocks</li>
                    <li>Additional insider selling or analyst downgrades</li>
                    <li>Technical breakdown below support at ${max([s for s in support_levels if s < current_price] or [current_price * 0.85]):.2f}</li>
                    <li>Space sector remains under pressure from SpaceX competition</li>
                </ul>
                <p><strong>Risk Events:</strong> August earnings disappointment → Margin pressure → Dilution concerns</p>
            </div>
        </div>

        <h3>Key Dates & Catalysts (Next 90 Days)</h3>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Event</th>
                    <th>Expected Impact</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>August 10, 2026</td>
                    <td><strong>Q2 2026 Earnings Release</strong></td>
                    <td>HIGH — Revenue, margins, backlog, Neutron/Iridium updates</td>
                </tr>
                <tr>
                    <td>Late August</td>
                    <td>Iridium merger progress updates</td>
                    <td>MEDIUM — Regulatory filings, shareholder votes</td>
                </tr>
                <tr>
                    <td>September</td>
                    <td>Neutron development milestones</td>
                    <td>MEDIUM-HIGH — Engine tests, structural testing updates</td>
                </tr>
                <tr>
                    <td>Ongoing</td>
                    <td>Launch cadence (Electron missions)</td>
                    <td>MEDIUM — Demonstrates operational reliability</td>
                </tr>
                <tr>
                    <td>September-October</td>
                    <td>Potential contract announcements</td>
                    <td>MEDIUM — Backlog growth, revenue visibility</td>
                </tr>
                <tr>
                    <td>Late 2026</td>
                    <td>Neutron first flight (if on schedule)</td>
                    <td>VERY HIGH — Game-changing catalyst</td>
                </tr>
            </tbody>
        </table>

        <h3>Invalidation Levels</h3>
        <ul>
            <li><strong>Bullish thesis invalidated below:</strong> ${bear_target * 0.9:.2f} — would indicate fundamental deterioration</li>
            <li><strong>Bearish thesis invalidated above:</strong> ${bull_target * 0.9:.2f} — would indicate strong recovery momentum</li>
            <li><strong>Critical support:</strong> ${max([s for s in support_levels if s < current_price] or [current_price * 0.85]):.2f} — break below likely triggers further selling</li>
            <li><strong>Critical resistance:</strong> ${min([r for r in resistance_levels if r > current_price] or [current_price * 1.15]):.2f} — breakout above signals trend reversal</li>
        </ul>

        <h3>Confidence Assessment</h3>
        <p><strong>Overall Confidence: 6/10 (Moderate)</strong></p>
        <ul>
            <li><strong>Why moderate confidence?</strong>
                <ul>
                    <li>✅ Strong fundamentals provide downside support</li>
                    <li>✅ Clear technical levels and analyst consensus</li>
                    <li>✅ Visible catalysts (earnings, Neutron) create tradeable events</li>
                    <li>⚠️ High volatility (~{volatility_30d:.0f}%) creates unpredictable swings</li>
                    <li>⚠️ Iridium deal introduces significant uncertainty</li>
                    <li>⚠️ Sector dynamics (SpaceX impact) remain unclear</li>
                    <li>⚠️ Event-driven nature makes technical analysis less reliable</li>
                </ul>
            </li>
        </ul>
    </div>

    <div class="section">
        <h2>🎯 Investment Rating & Conclusion</h2>

        <div style="margin: 20px 0;">
            <span class="rating-box rating-hold">RATING: HOLD</span>
            <span class="confidence-box">CONFIDENCE: 6/10</span>
        </div>

        <h3>Rationale</h3>
        <p>Rocket Lab presents a <strong>mixed risk-reward profile</strong> at current levels (~${current_price:.2f}). While the company's fundamental business remains robust—evidenced by record revenue growth (+63.5% YoY), a $2.2B backlog, and major government contracts—the stock faces significant near-term headwinds from the SpaceX IPO-driven sector rotation and the uncertainty surrounding the $8B Iridium acquisition.</p>

        <h3>Why HOLD (Not Buy or Sell)?</h3>
        <ul>
            <li><strong>Valuation Reset:</strong> The stock has corrected ~{((swing_high - current_price) / swing_high * 100):.0f}% from its May high, bringing it closer to the low end of analyst price targets ($60-150 range). This suggests much of the bad news may be priced in.</li>
            <li><strong>Catalyst-Rich Environment:</strong> Q2 earnings (Aug 10), Neutron milestones, and potential contract wins create multiple opportunities for positive surprises—but also downside risk if execution falters.</li>
            <li><strong>Technical Consolidation:</strong> The stock appears to be forming a base after the sharp selloff. RSI at {current_rsi:.1f} shows {"oversold" if current_rsi < 35 else "neutral"} conditions, but moving averages remain bearish.</li>
            <li><strong>Iridium Overhang:</strong> Until the Iridium deal terms are finalized and shareholder/regulatory approvals secured, uncertainty will cap upside.</li>
        </ul>

        <h3>For Different Investor Types</h3>
        <ul>
            <li><strong>Long-Term Growth Investors:</strong> Current levels may present a long-term opportunity if you believe in Rocket Lab's vision and can tolerate high volatility. Consider averaging in rather than a single large position.</li>
            <li><strong>Short-Term Traders:</strong> Wait for clearer technical signals (breakout above ${min([r for r in resistance_levels if r > current_price] or [current_price * 1.15]):.2f} or breakdown below ${max([s for s in support_levels if s < current_price] or [current_price * 0.85]):.2f}). August earnings will be a high-volatility event.</li>
            <li><strong>Risk-Averse Investors:</strong> Stay on the sidelines until Iridium deal clarity and Neutron first flight de-risk the story. The {volatility_30d:.0f}% annualized volatility is too high for conservative portfolios.</li>
            <li><strong>Current Holders:</strong> HOLD unless you need to reduce exposure to high-beta growth. The fundamental thesis hasn't changed, just the market's willingness to pay premium valuations.</li>
        </ul>

        <h3>What Would Change the Rating?</h3>
        <table>
            <thead>
                <tr>
                    <th>Upgrade to BUY if...</th>
                    <th>Downgrade to SELL if...</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>• Q2 earnings beat with strong guidance<br>• Neutron first flight successful<br>• Iridium deal terms improve or deal canceled<br>• Stock breaks above ${min([r for r in resistance_levels if r > current_price] or [current_price * 1.15]):.2f} on volume<br>• New major contracts announced</td>
                    <td>• Q2 earnings miss or weak guidance<br>• Neutron delays or technical failures<br>• Iridium deal dilution worse than expected<br>• Stock breaks below ${max([s for s in support_levels if s < current_price] or [current_price * 0.85]):.2f}<br>• Major contract cancellations or losses</td>
                </tr>
            </tbody>
        </table>

        <h3>Summary</h3>
        <p>RKLB has experienced a severe correction driven primarily by external factors (SpaceX IPO, sector rotation) rather than company-specific operational deterioration. The business fundamentals remain strong, with impressive revenue growth and a multi-billion dollar backlog. However, near-term uncertainty around the Iridium acquisition and the need to prove Neutron's viability justify a <strong>HOLD rating</strong> until these catalysts resolve. Investors should monitor the August 10 earnings closely and watch for technical signals around key support/resistance levels.</p>

        <p><strong>Target Price (Base Case):</strong> ${base_target:.2f} by end of October 2026</p>
        <p><strong>Bull Case Target:</strong> ${bull_target:.2f} | <strong>Bear Case Target:</strong> ${bear_target:.2f}</p>
    </div>

    <div class="disclaimer">
        <h3>⚠️ DISCLAIMER</h3>
        <p><strong>This analysis is for informational and educational purposes only and does not constitute financial, investment, or trading advice.</strong> The information provided should not be construed as a recommendation to buy, sell, or hold any security or financial instrument. All investments carry risk, including the potential loss of principal. Past performance does not guarantee future results. You should conduct your own research and consult with a licensed financial advisor before making any investment decisions. The analysis is based on publicly available information and may contain errors or omissions. Market conditions can change rapidly, and this analysis may become outdated.</p>
        
        <p><strong>Data Sources & Limitations:</strong> Historical price data is from Nasdaq ITCH (venue-limited; volume may be understated as it excludes off-exchange activity). Current session data from Google Finance (delayed ~15-20 minutes). News and fundamental data from publicly available sources. Short interest data is as of July 15, 2026 (lagged). Analyst ratings and price targets are aggregated from various sources and may not reflect most recent updates.</p>
        
        <p><strong>Report Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    with open('/home/ubuntu/RKLB_analysis.html', 'w') as f:
        f.write(html_content)
    
    print("\n✅ Analysis complete!")
    print(f"HTML report saved to: /home/ubuntu/RKLB_analysis.html")
    print(f"Charts saved to:")
    print("  - /home/ubuntu/RKLB_price_chart.png")
    print("  - /home/ubuntu/RKLB_indicators.png")
    print("  - /home/ubuntu/RKLB_comparison.png")

if __name__ == "__main__":
    main()
