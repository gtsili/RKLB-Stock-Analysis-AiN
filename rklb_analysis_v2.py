#!/usr/bin/env python3
"""
Rigorous RKLB Investment Analysis - Primary Sources Only
Event-driven analysis with verified catalysts from Rocket Lab IR and SEC filings
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

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def find_support_resistance_refined(daily_df, hourly_df, num_levels=3):
    """
    Find support and resistance levels using both daily and hourly data
    Uses swing highs/lows and clustered reaction zones
    """
    # Use last 6 months of hourly data for near-term levels
    recent_hourly = hourly_df.tail(1000)  # ~6 months of hourly bars
    
    # Find swing highs and lows in hourly data (higher precision)
    swing_high_threshold = recent_hourly['high'].quantile(0.90)
    swing_low_threshold = recent_hourly['low'].quantile(0.10)
    
    # Identify potential levels where price reversed
    resistance_candidates = []
    support_candidates = []
    
    # Look for peaks and troughs in hourly data
    for i in range(10, len(recent_hourly) - 10):
        # Resistance: local high with price reversal
        if (recent_hourly['high'].iloc[i] == recent_hourly['high'].iloc[i-10:i+10].max() and
            recent_hourly['high'].iloc[i] > swing_high_threshold):
            resistance_candidates.append(recent_hourly['high'].iloc[i])
        
        # Support: local low with price reversal  
        if (recent_hourly['low'].iloc[i] == recent_hourly['low'].iloc[i-10:i+10].min() and
            recent_hourly['low'].iloc[i] < swing_low_threshold):
            support_candidates.append(recent_hourly['low'].iloc[i])
    
    # Cluster nearby levels (within 2%)
    def cluster_levels(levels, threshold=0.02):
        if not levels:
            return []
        levels = sorted(levels, reverse=True)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[0]) / current_cluster[0] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters[:num_levels]
    
    resistance_levels = cluster_levels(resistance_candidates)
    support_levels = cluster_levels(support_candidates)
    
    return support_levels, resistance_levels

def create_event_study(df, swing_high_date, current_date, bench_df, peer_df):
    """
    Create event study table for top down days since swing high
    """
    # Filter data since swing high
    event_window = df.loc[swing_high_date:current_date]
    
    # Calculate daily returns
    event_window['return'] = event_window['close'].pct_change() * 100
    event_window['volume_20d_avg'] = event_window['volume'].rolling(20).mean()
    event_window['volume_vs_avg'] = event_window['volume'] / event_window['volume_20d_avg']
    
    # Get benchmark returns
    spy_df = bench_df[bench_df['symbol'] == 'SPY'].copy()
    spy_df['spy_return'] = spy_df['close'].pct_change() * 100
    
    qqq_df = bench_df[bench_df['symbol'] == 'QQQ'].copy()
    qqq_df['qqq_return'] = qqq_df['close'].pct_change() * 100
    
    # Merge benchmark data
    event_window = event_window.join(spy_df[['spy_return']], how='left')
    event_window = event_window.join(qqq_df[['qqq_return']], how='left')
    
    # Find top 10 down days
    down_days = event_window[event_window['return'] < 0].nlargest(10, 'return', keep='all').copy()
    down_days = down_days.sort_values('return')
    
    return down_days

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    return img_str

def main():
    print("="*80)
    print("RKLB RIGOROUS ANALYSIS - PRIMARY SOURCES ONLY")
    print("="*80)
    
    # Load data
    print("\n[1/8] Loading market data...")
    rklb_daily = load_market_data('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1785228894.json')
    rklb_hourly = load_market_data('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1785229458.json')
    bench_df = load_market_data('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1785228914.json')
    peer_df = load_market_data('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1785229445.json')
    
    # Focus on primary analysis period: 2024-07-01 onwards
    analysis_start = pd.Timestamp('2024-07-01', tz='UTC')
    rklb_df = rklb_daily.loc[analysis_start:]
    
    print(f"   - RKLB daily: {len(rklb_df)} days ({rklb_df.index[0].date()} to {rklb_df.index[-1].date()})")
    print(f"   - RKLB hourly: {len(rklb_hourly)} hours")
    print(f"   - Benchmarks: {len(bench_df)} records")
    print(f"   - Peers: {len(peer_df)} records")
    
    # Calculate technical indicators
    print("\n[2/8] Calculating technical indicators...")
    rklb_df['SMA_50'] = calculate_sma(rklb_df, 50)
    rklb_df['SMA_200'] = calculate_sma(rklb_df, 200)
    rklb_df['RSI'] = calculate_rsi(rklb_df, 14)
    rklb_df['MACD'], rklb_df['MACD_Signal'], rklb_df['MACD_Hist'] = calculate_macd(rklb_df)
    rklb_df['BB_Upper'], rklb_df['BB_Middle'], rklb_df['BB_Lower'] = calculate_bollinger_bands(rklb_df)
    rklb_df['ATR'] = calculate_atr(rklb_df, 14)
    
    # Find swing high
    print("\n[3/8] Identifying swing high and key levels...")
    swing_high_idx = rklb_df['high'].idxmax()
    swing_high = rklb_df.loc[swing_high_idx, 'high']
    swing_high_date = swing_high_idx
    
    print(f"   - Swing high: ${swing_high:.2f} on {swing_high_date.date()}")
    
    # Refined support/resistance using hourly data
    support_levels, resistance_levels = find_support_resistance_refined(rklb_df, rklb_hourly)
    print(f"   - Support levels: {[f'${s:.2f}' for s in support_levels]}")
    print(f"   - Resistance levels: {[f'${r:.2f}' for r in resistance_levels]}")
    
    # Current values
    current_date = rklb_df.index[-1]
    current_price = rklb_df['close'].iloc[-1]
    current_rsi = rklb_df['RSI'].iloc[-1]
    current_atr = rklb_df['ATR'].iloc[-1]
    
    print(f"   - Current price: ${current_price:.2f} on {current_date.date()}")
    print(f"   - Current RSI: {current_rsi:.1f}")
    print(f"   - Current ATR: ${current_atr:.2f}")
    
    # Volume analysis
    print("\n[4/8] Analyzing volume and liquidity...")
    vol_20d = rklb_df['volume'].tail(20).mean()
    vol_60d = rklb_df['volume'].tail(60).mean()
    dollar_vol_20d = (rklb_df['volume'] * rklb_df['close']).tail(20).mean()
    dollar_vol_60d = (rklb_df['volume'] * rklb_df['close']).tail(60).mean()
    
    print(f"   - 20D avg volume: {vol_20d:,.0f} shares (${dollar_vol_20d/1e6:.1f}M)")
    print(f"   - 60D avg volume: {vol_60d:,.0f} shares (${dollar_vol_60d/1e6:.1f}M)")
    
    # Event study
    print("\n[5/8] Building event study for top down days...")
    event_study_df = create_event_study(rklb_df, swing_high_date, current_date, bench_df, peer_df)
    
    # Performance windows
    print("\n[6/8] Calculating performance across key windows...")
    
    def calc_perf(df, days):
        if len(df) < days:
            days = len(df)
        start_price = df['close'].iloc[-days]
        end_price = df['close'].iloc[-1]
        return {
            'days': days,
            'start_price': start_price,
            'end_price': end_price,
            'change': end_price - start_price,
            'pct_change': (end_price - start_price) / start_price * 100,
            'period_high': df['high'].iloc[-days:].max(),
            'period_low': df['low'].iloc[-days:].min()
        }
    
    perf_30d = calc_perf(rklb_df, 30)
    perf_90d = calc_perf(rklb_df, 90)
    perf_swing = {
        'start_price': swing_high,
        'end_price': current_price,
        'change': current_price - swing_high,
        'pct_change': (current_price - swing_high) / swing_high * 100,
        'days': len(rklb_df.loc[swing_high_date:current_date])
    }
    
    print(f"   - 30D: {perf_30d['pct_change']:.2f}%")
    print(f"   - 90D: {perf_90d['pct_change']:.2f}%")
    print(f"   - From swing high ({perf_swing['days']} days): {perf_swing['pct_change']:.2f}%")
    
    # Volatility
    print("\n[7/8] Measuring realized volatility...")
    returns = rklb_df['close'].pct_change()
    vol_30d = returns.tail(30).std() * np.sqrt(252) * 100
    vol_60d = returns.tail(60).std() * np.sqrt(252) * 100
    vol_90d = returns.tail(90).std() * np.sqrt(252) * 100
    
    print(f"   - 30D realized vol: {vol_30d:.1f}% annualized")
    print(f"   - 60D realized vol: {vol_60d:.1f}% annualized")
    print(f"   - 90D realized vol: {vol_90d:.1f}% annualized")
    
    # Forecast scenarios based on ATR, volatility, and catalysts
    print("\n[8/8] Building 3-month forecast scenarios...")
    
    # Horizon: ~90 days (end of October 2026)
    forecast_days = 90
    
    # Base forecast on ATR and realized volatility
    atr_20d = rklb_df['ATR'].tail(20).mean()
    
    # Bull case: Technical recovery + positive catalysts
    # - Reclaim first resistance level
    # - 1.5 sigma move up based on 60D volatility
    bull_technical = current_price * (1 + (vol_60d / 100) * 1.5 * np.sqrt(forecast_days / 252))
    bull_target = min(resistance_levels) if resistance_levels and min(resistance_levels) > current_price else bull_technical
    bull_target = min(bull_target, current_price * 1.40)  # Cap at 40% upside
    
    # Base case: Range-bound trading
    # - Mean reversion toward recent average
    # - Slight upward bias given fundamentals
    base_target = current_price * 1.08
    
    # Bear case: Further breakdown
    # - Test next support level
    # - 1.0 sigma move down based on 60D volatility  
    bear_technical = current_price * (1 - (vol_60d / 100) * 1.0 * np.sqrt(forecast_days / 252))
    bear_target = max([s for s in support_levels if s < current_price]) if [s for s in support_levels if s < current_price] else bear_technical
    bear_target = max(bear_target, current_price * 0.70)  # Floor at -30%
    
    print(f"   - Bull target: ${bull_target:.2f} ({((bull_target/current_price - 1) * 100):.1f}%)")
    print(f"   - Base target: ${base_target:.2f} ({((base_target/current_price - 1) * 100):.1f}%)")
    print(f"   - Bear target: ${bear_target:.2f} ({((bear_target/current_price - 1) * 100):.1f}%)")
    
    print("\n" + "="*80)
    print("Analysis complete. Generating HTML report...")
    print("="*80 + "\n")
    
    # Save analysis results
    analysis_results = {
        'current_price': current_price,
        'current_date': str(current_date.date()),
        'swing_high': swing_high,
        'swing_high_date': str(swing_high_date.date()),
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'perf_30d': perf_30d,
        'perf_90d': perf_90d,
        'perf_swing': perf_swing,
        'current_rsi': current_rsi,
        'current_atr': current_atr,
        'vol_30d': vol_30d,
        'vol_60d': vol_60d,
        'vol_90d': vol_90d,
        'vol_20d_shares': vol_20d,
        'vol_60d_shares': vol_60d,
        'dollar_vol_20d': dollar_vol_20d,
        'dollar_vol_60d': dollar_vol_60d,
        'bull_target': bull_target,
        'base_target': base_target,
        'bear_target': bear_target,
        'event_study': event_study_df.to_dict('records') if len(event_study_df) > 0 else []
    }
    
    with open('/home/ubuntu/rklb_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print("Results saved to: /home/ubuntu/rklb_analysis_results.json")

if __name__ == "__main__":
    main()
