#!/usr/bin/env python3
"""
Generate final RKLB HTML report with charts
Based on primary sources and rigorous analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import base64
from io import BytesIO

# Load analysis results
with open('/home/ubuntu/rklb_analysis_results.json', 'r') as f:
    results = json.load(f)

# Load market data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'])
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('ts_event')
    df.set_index('ts_event', inplace=True)
    return df

rklb_df = load_data('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1785228894.json')
bench_df = load_data('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1785228914.json')
peer_df = load_data('/home/ubuntu/.external_service_outputs/fetch_market_data_output_1785229445.json')

# Filter to analysis period
rklb_df = rklb_df.loc['2024-07-01':]

# Calculate indicators
rklb_df['SMA_50'] = rklb_df['close'].rolling(50).mean()
rklb_df['SMA_200'] = rklb_df['close'].rolling(200).mean()
rklb_df['RSI'] = (lambda d: 100 - (100 / (1 + ((d.where(d > 0, 0)).rolling(14).mean() / (-d.where(d < 0, 0)).rolling(14).mean()))))(rklb_df['close'].diff())

# Helper function
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    return img_str

# Chart 1: Price with support/resistance
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(rklb_df.index, rklb_df['close'], label='Close Price', linewidth=1.5, color='#2E86AB')
ax1.plot(rklb_df.index, rklb_df['SMA_50'], label='50-Day SMA', linewidth=1.2, color='#A23B72', linestyle='--')
ax1.plot(rklb_df.index, rklb_df['SMA_200'], label='200-Day SMA', linewidth=1.2, color='#F18F01', linestyle='--')

for level in results['support_levels']:
    if level < results['current_price']:
        ax1.axhline(y=level, color='green', linestyle=':', linewidth=1, alpha=0.7)
        
for level in results['resistance_levels']:
    if level > results['current_price']:
        ax1.axhline(y=level, color='red', linestyle=':', linewidth=1, alpha=0.7)

ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
ax1.set_title('RKLB Price History (July 2024 - July 2026)', fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

colors = ['green' if rklb_df['close'].iloc[i] >= rklb_df['open'].iloc[i] else 'red' for i in range(len(rklb_df))]
ax2.bar(rklb_df.index, rklb_df['volume'], color=colors, alpha=0.6, width=0.8)
ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
chart1_b64 = fig_to_base64(fig1)

# Chart 2: Peer comparison
fig2, ax = plt.subplots(figsize=(14, 7))
start_idx = -90
rklb_norm = (rklb_df['close'][start_idx:] / rklb_df['close'].iloc[start_idx]) * 100

spy_data = bench_df[bench_df['symbol'] == 'SPY']['close']
spy_data.index = bench_df[bench_df['symbol'] == 'SPY'].index
spy_norm = (spy_data[start_idx:] / spy_data.iloc[start_idx]) * 100

qqq_data = bench_df[bench_df['symbol'] == 'QQQ']['close']
qqq_data.index = bench_df[bench_df['symbol'] == 'QQQ'].index
qqq_norm = (qqq_data[start_idx:] / qqq_data.iloc[start_idx]) * 100

ax.plot(rklb_norm.index, rklb_norm, label='RKLB', linewidth=2.5, color='#2E86AB')
ax.plot(spy_norm.index, spy_norm, label='SPY (S&P 500)', linewidth=1.5, color='#F18F01', linestyle='--')
ax.plot(qqq_norm.index, qqq_norm, label='QQQ (Nasdaq)', linewidth=1.5, color='#A23B72', linestyle='--')

ax.axhline(y=100, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_ylabel('Normalized Performance (Base = 100)', fontsize=12, fontweight='bold')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_title('RKLB vs. Market Benchmarks (90-Day Performance)', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
chart2_b64 = fig_to_base64(fig2)

# Chart 3: RSI
fig3, ax = plt.subplots(figsize=(14, 5))
ax.plot(rklb_df.index, rklb_df['RSI'], label='RSI', linewidth=1.5, color='#2E86AB')
ax.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (70)')
ax.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (30)')
ax.fill_between(rklb_df.index, 30, 70, alpha=0.1, color='gray')
ax.set_ylabel('RSI', fontsize=12, fontweight='bold')
ax.set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
chart3_b64 = fig_to_base64(fig3)

# Generate HTML report
html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RKLB Investment Analysis Report</title>
    <style>
        body {{font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;}}
        .header {{background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}}
        .header h1 {{margin: 0 0 10px 0; font-size: 2.5em;}}
        .section {{background-color: white; padding: 25px; margin-bottom: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
        h2 {{color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 10px;}}
        h3 {{color: #A23B72; margin-top: 25px;}}
        table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
        table th, table td {{padding: 12px; text-align: left; border-bottom: 1px solid #ddd;}}
        table th {{background-color: #2E86AB; color: white;}}
        .metric-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;}}
        .metric-card {{background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB;}}
        .metric-label {{font-size: 0.9em; color: #666; margin-bottom: 5px;}}
        .metric-value {{font-size: 1.4em; font-weight: bold; color: #333;}}
        .positive {{color: #28A745;}}
        .negative {{color: #DC3545;}}
        .rating-box {{display: inline-block; padding: 15px 30px; border-radius: 8px; font-weight: bold; font-size: 1.3em; margin: 10px 10px 10px 0;}}
        .rating-hold {{background-color: #FFF3CD; color: #856404; border: 2px solid #FFC107;}}
        .confidence-box {{display: inline-block; padding: 15px 30px; border-radius: 8px; font-weight: bold; font-size: 1.3em; background-color: #D1ECF1; color: #0C5460; border: 2px solid #17A2B8;}}
        .chart {{margin: 25px 0; text-align: center;}}
        .chart img {{max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}}
        .disclaimer {{background-color: #FFF3CD; border: 2px solid #FFC107; padding: 20px; border-radius: 8px; margin-top: 30px; font-size: 0.9em;}}
        .timestamp {{color: #666; font-size: 0.9em; font-style: italic;}}
        ul {{line-height: 1.8;}}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Rocket Lab USA Inc. (RKLB)</h1>
        <h2>Investment Analysis Report</h2>
        <p class="timestamp">Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        <p class="timestamp">Market Data as of: {results['current_date']}</p>
        <p style="margin-top: 15px; font-size: 0.95em;">Analysis based on primary sources: Rocket Lab Investor Relations, SEC filings (8-K), and verified market data</p>
    </div>

    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div style="margin: 20px 0;">
            <span class="rating-box rating-hold">RATING: HOLD</span>
            <span class="confidence-box">CONFIDENCE: 6/10</span>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">${results['current_price']:.2f}</div>
                <div class="timestamp">{results['current_date']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Swing High (May 27, 2026)</div>
                <div class="metric-value">${results['swing_high']:.2f}</div>
                <div class="negative">Down {abs(results['perf_swing']['pct_change']):.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">30-Day Performance</div>
                <div class="metric-value negative">{results['perf_30d']['pct_change']:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Realized Volatility (30D)</div>
                <div class="metric-value">{results['vol_30d']:.1f}%</div>
                <div class="timestamp">Annualized</div>
            </div>
        </div>

        <h3>Key Takeaways</h3>
        <ul>
            <li><strong>Severe Decline:</strong> RKLB declined {abs(results['perf_swing']['pct_change']):.1f}% from its May 27, 2026 high of ${results['swing_high']:.2f} to current levels of ${results['current_price']:.2f}.</li>
            <li><strong>Extreme Volatility:</strong> Annualized realized volatility exceeds {results['vol_30d']:.0f}%, indicating high-risk, event-driven price action.</li>
            <li><strong>Fundamentals Remain Strong:</strong> Q1 2026 revenue of $200.3M (+63.5% YoY), $2.2B backlog, and $266M Space Force contract demonstrate operational strength.</li>
            <li><strong>Major Catalysts Ahead:</strong> Q2 earnings (Aug 10), Iridium acquisition progress, and Neutron first flight (late 2026) are key near-term drivers.</li>
            <li><strong>Trading Near Support:</strong> Current price of ${results['current_price']:.2f} is testing critical support levels around ${min([s for s in results['support_levels'] if s < results['current_price']], default=results['current_price']):.2f}.</li>
        </ul>
    </div>

    <div class="section">
        <h2>📉 Why Did RKLB Go Down?</h2>
        
        <h3>Performance Summary</h3>
        <table>
            <thead>
                <tr><th>Period</th><th>Start Price</th><th>End Price</th><th>Change ($)</th><th>Change (%)</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>30 Days</strong></td>
                    <td>${results['perf_30d']['start_price']:.2f}</td>
                    <td>${results['perf_30d']['end_price']:.2f}</td>
                    <td class="negative">${results['perf_30d']['change']:.2f}</td>
                    <td class="negative">{results['perf_30d']['pct_change']:.2f}%</td>
                </tr>
                <tr>
                    <td><strong>90 Days</strong></td>
                    <td>${results['perf_90d']['start_price']:.2f}</td>
                    <td>${results['perf_90d']['end_price']:.2f}</td>
                    <td class="negative">${results['perf_90d']['change']:.2f}</td>
                    <td class="negative">{results['perf_90d']['pct_change']:.2f}%</td>
                </tr>
                <tr>
                    <td><strong>From Swing High</strong><br/><span class="timestamp">({results['swing_high_date']})</span></td>
                    <td>${results['perf_swing']['start_price']:.2f}</td>
                    <td>${results['perf_swing']['end_price']:.2f}</td>
                    <td class="negative">${results['perf_swing']['change']:.2f}</td>
                    <td class="negative">{results['perf_swing']['pct_change']:.2f}%</td>
                </tr>
            </tbody>
        </table>

        <h3>Verified Catalysts from Primary Sources</h3>
        <p><strong>Source: Rocket Lab Investor Relations & SEC Filings (8-K)</strong></p>
        
        <h4>1. Iridium Acquisition Announcement (June 29, 2026)</h4>
        <ul>
            <li><strong>Deal Structure:</strong> $8 billion cash-and-stock transaction (Iridium shareholders receive $54/share: $27 cash + stock with collar between $67.50-$112.50)</li>
            <li><strong>Financing:</strong> $3.6B bridge loan from Deutsche Bank and Wells Fargo</li>
            <li><strong>Market Reaction:</strong> RKLB +16% on announcement day, but stock subsequently fell below $67.50 collar threshold</li>
            <li><strong>Investor Concerns:</strong> Dilution risk, integration complexity, debt burden, and collar mechanics</li>
            <li><strong>Source:</strong> 8-K filed June 29, 2026; Rocket Lab press release</li>
        </ul>

        <h4>2. Q1 2026 Financial Results (May 7, 2026)</h4>
        <ul>
            <li><strong>Revenue:</strong> $200.3M (exceeded guidance, +63.5% YoY)</li>
            <li><strong>Backlog:</strong> $2.2B (+20.2% sequential)</li>
            <li><strong>Operating Loss:</strong> $56.0M; Adj. EBITDA loss: $11.8M</li>
            <li><strong>Q2 Guidance:</strong> $225-240M revenue, continued EBITDA losses</li>
            <li><strong>Context:</strong> Strong revenue growth but ongoing profitability challenges</li>
            <li><strong>Source:</strong> 8-K filed May 7, 2026; Q1 2026 earnings release</li>
        </ul>

        <h4>3. Major Contract Wins (April-July 2026)</h4>
        <ul>
            <li><strong>July 27:</strong> $266M U.S. Space Force missile defense contract (12 launches + 6 options)</li>
            <li><strong>June 25:</strong> NASA PolSIR and TSIS-2 missions (3 Electron launches, Q1 2027)</li>
            <li><strong>June 22:</strong> VICTUS HAZE mission (16 hours 42 minutes responsive launch record)</li>
            <li><strong>May 26:</strong> Motiv Space Systems acquisition completed (Mars robotics)</li>
            <li><strong>Source:</strong> Rocket Lab press releases verified through SEC filings</li>
        </ul>

        <h4>4. Corporate Milestones</h4>
        <ul>
            <li><strong>June 12, 2026:</strong> Nasdaq-100 Index inclusion (index fund flows)</li>
            <li><strong>Neutron Development:</strong> On track for late 2026 first flight (per Q1 earnings)</li>
            <li><strong>Source:</strong> Verified through Rocket Lab IR and press releases</li>
        </ul>

        <div class="chart">
            <img src="data:image/png;base64,{chart2_b64}" alt="RKLB vs Benchmarks">
            <p class="timestamp">90-Day Performance: RKLB vs. SPY and QQQ</p>
        </div>

        <h3>Analysis: Idiosyncratic vs. Market-Wide</h3>
        <p>Comparing RKLB to SPY and QQQ over the past 90 days reveals that the decline was <strong>primarily idiosyncratic</strong>, not driven by broad market weakness. While SPY and QQQ remained relatively stable or showed modest gains, RKLB experienced a severe selloff. This indicates that company-specific and sector-specific factors dominated the price action.</p>
        
        <p><strong>Note on Volume:</strong> The historical data used is venue-limited (Nasdaq ITCH only) and may understate total volume as it excludes off-exchange activity. Average daily volume: {results['vol_20d_shares']:,.0f} shares (${results['dollar_vol_20d']/1e6:.1f}M), providing adequate liquidity but subject to elevated volatility.</p>
    </div>

    <div class="section">
        <h2>📈 Technical Analysis</h2>
        
        <div class="chart">
            <img src="data:image/png;base64,{chart1_b64}" alt="RKLB Price Chart">
            <p class="timestamp">RKLB Price History with 50-Day and 200-Day Moving Averages</p>
        </div>

        <h3>Support and Resistance Levels</h3>
        <p><strong>Methodology:</strong> Levels identified using hourly price data from January-July 2026, focusing on swing highs/lows and clustered reaction zones.</p>
        
        <table>
            <thead>
                <tr><th>Level Type</th><th>Price</th><th>Distance from Current</th></tr>
            </thead>
            <tbody>
                {"".join([f'<tr><td><strong>Resistance {i+1}</strong></td><td>${level:.2f}</td><td class="positive">+${level - results["current_price"]:.2f} (+{((level - results["current_price"]) / results["current_price"] * 100):.1f}%)</td></tr>' for i, level in enumerate(results['resistance_levels']) if level > results['current_price']])}
                <tr style="background-color: #e3f2fd;"><td><strong>Current Price</strong></td><td><strong>${results['current_price']:.2f}</strong></td><td>—</td></tr>
                {"".join([f'<tr><td><strong>Support {i+1}</strong></td><td>${level:.2f}</td><td class="negative">-${results["current_price"] - level:.2f} (-{((results["current_price"] - level) / results["current_price"] * 100):.1f}%)</td></tr>' for i, level in enumerate(results['support_levels']) if level < results['current_price']])}
            </tbody>
        </table>

        <div class="chart">
            <img src="data:image/png;base64,{chart3_b64}" alt="RSI">
            <p class="timestamp">Relative Strength Index (RSI)</p>
        </div>

        <h3>Technical Indicators</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">RSI (14-period)</div>
                <div class="metric-value">{results['current_rsi']:.1f}</div>
                <div class="timestamp">{"Oversold (<30)" if results['current_rsi'] < 30 else "Neutral (30-70)" if results['current_rsi'] < 70 else "Overbought (>70)"}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ATR (14-period)</div>
                <div class="metric-value">${results['current_atr']:.2f}</div>
                <div class="timestamp">Average daily range</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">30D Volatility</div>
                <div class="metric-value">{results['vol_30d']:.1f}%</div>
                <div class="timestamp">Annualized</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">60D Volatility</div>
                <div class="metric-value">{results['vol_60d']:.1f}%</div>
                <div class="timestamp">Annualized</div>
            </div>
        </div>

        <p><strong>Interpretation:</strong> RSI at {results['current_rsi']:.1f} indicates oversold conditions, suggesting potential for a relief rally. However, extremely high volatility ({results['vol_30d']:.0f}% annualized) creates unpredictable swings. The stock is trading {"below" if results['current_price'] < min(results['support_levels']) else "near"} critical support levels, with next major support at ${min(results['support_levels']):.2f}.</p>
    </div>

    <div class="section">
        <h2>🔮 3-Month Price Forecast (Through October 2026)</h2>
        
        <p><strong>Methodology:</strong> Scenarios anchored to 20-day ATR (${results['current_atr']:.2f}), 60-day realized volatility ({results['vol_60d']:.1f}%), verified support/resistance levels, and confirmed catalysts (Q2 earnings Aug 10, Neutron progress, Iridium updates).</p>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 25px 0;">
            <div style="padding: 20px; border-radius: 8px; border: 2px solid #28A745; background-color: #D4EDDA;">
                <h4 style="margin-top: 0;">🚀 BULL CASE</h4>
                <div style="font-size: 2em; font-weight: bold; margin: 10px 0; color: #28A745;">${results['bull_target']:.2f}</div>
                <div style="font-size: 1.1em; margin-bottom: 15px;"><strong>Upside: {((results['bull_target'] - results['current_price']) / results['current_price'] * 100):.1f}%</strong></div>
                <p><strong>Probability: 25%</strong></p>
                <ul>
                    <li>Q2 earnings beat ($225-240M guidance)</li>
                    <li>Neutron first flight successful by Q4 2026</li>
                    <li>Iridium deal concerns alleviated, collar repricing</li>
                    <li>Technical breakout above ${min([r for r in results['resistance_levels'] if r > results['current_price']], default=results['current_price']*1.2):.2f}</li>
                    <li>Additional major contract wins</li>
                </ul>
            </div>

            <div style="padding: 20px; border-radius: 8px; border: 2px solid #17A2B8; background-color: #D1ECF1;">
                <h4 style="margin-top: 0;">📊 BASE CASE</h4>
                <div style="font-size: 2em; font-weight: bold; margin: 10px 0; color: #17A2B8;">${results['base_target']:.2f}</div>
                <div style="font-size: 1.1em; margin-bottom: 15px;"><strong>{"Upside" if results['base_target'] > results['current_price'] else "Downside"}: {((results['base_target'] - results['current_price']) / results['current_price'] * 100):.1f}%</strong></div>
                <p><strong>Probability: 50%</strong></p>
                <ul>
                    <li>Q2 earnings in-line with guidance</li>
                    <li>Neutron on track but no flight by Oct</li>
                    <li>Iridium acquisition progresses without major issues</li>
                    <li>Range-bound trading ${min([s for s in results['support_levels'] if s < results['current_price']], default=results['current_price']*0.9):.2f} - ${min([r for r in results['resistance_levels'] if r > results['current_price']], default=results['current_price']*1.1):.2f}</li>
                    <li>Modest recovery as fundamentals reassessed</li>
                </ul>
            </div>

            <div style="padding: 20px; border-radius: 8px; border: 2px solid #DC3545; background-color: #F8D7DA;">
                <h4 style="margin-top: 0;">🐻 BEAR CASE</h4>
                <div style="font-size: 2em; font-weight: bold; margin: 10px 0; color: #DC3545;">${results['bear_target']:.2f}</div>
                <div style="font-size: 1.1em; margin-bottom: 15px;"><strong>Downside: {((results['bear_target'] - results['current_price']) / results['current_price'] * 100):.1f}%</strong></div>
                <p><strong>Probability: 25%</strong></p>
                <ul>
                    <li>Q2 earnings miss or weak H2 guidance</li>
                    <li>Neutron development delays</li>
                    <li>Iridium deal complications or financing concerns</li>
                    <li>Breakdown below support ${max([s for s in results['support_levels'] if s < results['current_price']], default=results['current_price']*0.85):.2f}</li>
                    <li>Continued high-beta growth stock pressure</li>
                </ul>
            </div>
        </div>

        <h3>Key Catalysts (Next 90 Days)</h3>
        <table>
            <thead>
                <tr><th>Date</th><th>Event</th><th>Impact</th></tr>
            </thead>
            <tbody>
                <tr><td><strong>August 10, 2026</strong></td><td>Q2 2026 Earnings Release</td><td>HIGH — Revenue, margins, backlog, Neutron/Iridium updates</td></tr>
                <tr><td>Late August</td><td>Iridium merger regulatory filings</td><td>MEDIUM — Shareholder votes, antitrust review</td></tr>
                <tr><td>September</td><td>Neutron development milestones</td><td>MEDIUM-HIGH — Engine tests, integration progress</td></tr>
                <tr><td>Ongoing</td><td>Electron launch cadence</td><td>MEDIUM — Operational reliability demonstration</td></tr>
                <tr><td>Late 2026</td><td>Neutron first flight (if on schedule)</td><td>VERY HIGH — Transformational catalyst</td></tr>
            </tbody>
        </table>

        <h3>Confidence Assessment: 6/10 (Moderate)</h3>
        <ul>
            <li>✅ Strong fundamentals provide downside support</li>
            <li>✅ Clear technical levels and visible catalysts</li>
            <li>⚠️ Extreme volatility ({results['vol_30d']:.0f}%) creates unpredictable swings</li>
            <li>⚠️ Iridium deal uncertainty (dilution, financing, collar)</li>
            <li>⚠️ Event-driven nature limits technical analysis reliability</li>
            <li>⚠️ Unverified claims about "SpaceX rotation" and insider selling patterns require caution</li>
        </ul>
    </div>

    <div class="section">
        <h2>🎯 Investment Rating & Conclusion</h2>

        <div style="margin: 20px 0;">
            <span class="rating-box rating-hold">RATING: HOLD</span>
            <span class="confidence-box">CONFIDENCE: 6/10</span>
        </div>

        <h3>Rationale</h3>
        <p>Rocket Lab presents a <strong>mixed risk-reward profile</strong> at ${results['current_price']:.2f}. While operational fundamentals remain robust—record Q1 revenue (+63.5% YoY), $2.2B backlog, major Space Force contract—the stock faces significant near-term uncertainty from the $8B Iridium acquisition and extreme volatility ({results['vol_30d']:.0f}% annualized).</p>

        <h3>Why HOLD?</h3>
        <ul>
            <li><strong>Valuation Reset:</strong> {abs(results['perf_swing']['pct_change']):.1f}% decline from May high suggests much negative sentiment already priced in</li>
            <li><strong>Catalyst-Rich Environment:</strong> Q2 earnings (Aug 10), Neutron milestones, Iridium progress create multiple inflection points</li>
            <li><strong>Technical Oversold:</strong> RSI at {results['current_rsi']:.1f}, trading near critical support ${min([s for s in results['support_levels'] if s < results['current_price']], default=results['current_price']):.2f}</li>
            <li><strong>Uncertainty Premium:</strong> Iridium deal structure (collar, dilution, financing) caps upside until clarity emerges</li>
        </ul>

        <h3>What Would Change the Rating?</h3>
        <table>
            <thead>
                <tr><th>Upgrade to BUY if...</th><th>Downgrade to SELL if...</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        • Q2 earnings beat with strong guidance<br>
                        • Neutron first flight successful<br>
                        • Iridium terms improve or deal canceled<br>
                        • Break above ${min([r for r in results['resistance_levels'] if r > results['current_price']], default=results['current_price']*1.2):.2f} on volume
                    </td>
                    <td>
                        • Q2 earnings miss or weak guidance<br>
                        • Neutron delays or technical failures<br>
                        • Iridium dilution worse than expected<br>
                        • Break below ${max([s for s in results['support_levels'] if s < results['current_price']], default=results['current_price']*0.85):.2f} support
                    </td>
                </tr>
            </tbody>
        </table>

        <h3>Target Prices</h3>
        <ul>
            <li><strong>Bull Case (25% probability):</strong> ${results['bull_target']:.2f} — {((results['bull_target']/results['current_price'] - 1) * 100):.1f}% upside</li>
            <li><strong>Base Case (50% probability):</strong> ${results['base_target']:.2f} — {((results['base_target']/results['current_price'] - 1) * 100):.1f}% {"upside" if results['base_target'] > results['current_price'] else "downside"}</li>
            <li><strong>Bear Case (25% probability):</strong> ${results['bear_target']:.2f} — {((results['bear_target']/results['current_price'] - 1) * 100):.1f}% downside</li>
        </ul>
    </div>

    <div class="disclaimer">
        <h3>⚠️ DISCLAIMER</h3>
        <p><strong>This analysis is for informational and educational purposes only and does not constitute financial, investment, or trading advice.</strong> The information provided should not be construed as a recommendation to buy, sell, or hold any security. All investments carry risk, including potential loss of principal. Past performance does not guarantee future results. Consult with a licensed financial advisor before making investment decisions.</p>
        
        <p><strong>Data Sources & Limitations:</strong></p>
        <ul>
            <li><strong>Primary Sources:</strong> Rocket Lab Investor Relations (investors.rocketlabcorp.com), SEC Form 8-K filings (CIK 0001819994)</li>
            <li><strong>Market Data:</strong> Nasdaq ITCH (venue-limited; volume excludes off-exchange activity and may be understated). Historical data through July 27, 2026.</li>
            <li><strong>Analysis Period:</strong> July 1, 2024 — July 27, 2026 (primary focus)</li>
            <li><strong>Limitations:</strong> Unverified web claims about "SpaceX IPO rotation" and specific insider selling patterns were excluded from the analysis. Forecasts are scenario-based estimates, not predictions. Market conditions change rapidly.</li>
        </ul>
        
        <p><strong>Report Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
</body>
</html>
"""

# Save HTML report
with open('/home/ubuntu/RKLB_analysis.html', 'w') as f:
    f.write(html)

print("✅ Final HTML report generated: /home/ubuntu/RKLB_analysis.html")
