# RKLB-Stock-Analysis-AiNative

RKLB (Rocket Lab USA, Inc.) Stock Analysis — an AI-native exploration of the stock's
historical price action, technical indicators, and peer comparison, packaged as a
self-contained HTML report with supporting charts and data.

## Overview

This project pulls RKLB market data, computes a set of technical indicators, generates
visualizations, and produces a shareable HTML report summarizing the findings.

## Contents

| File | Description |
| --- | --- |
| `rklb_analysis.py` | Core analysis script — fetches data, computes metrics/indicators, and builds charts. |
| `rklb_analysis_v2.py` | Updated/refined version of the analysis workflow. |
| `generate_report.py` | Assembles the analysis outputs into the `RKLB_analysis.html` report. |
| `rklb_analysis_results.json` | Structured results/metrics produced by the analysis. |
| `RKLB_analysis.html` | Standalone HTML report with the full write-up and embedded charts. |
| `RKLB_price_chart.png` | RKLB historical price chart. |
| `RKLB_indicators.png` | Technical indicators (e.g. moving averages, RSI, MACD). |
| `RKLB_comparison.png` | Peer / benchmark comparison chart. |

## Usage

```bash
# Install dependencies (pandas, numpy, matplotlib, yfinance, etc. as required)
pip install -r requirements.txt   # if a requirements file is provided

# Run the analysis
python rklb_analysis_v2.py

# Generate the HTML report
python generate_report.py
```

Then open `RKLB_analysis.html` in any web browser to view the report.

## Disclaimer

This project is for educational and informational purposes only and does not constitute
financial advice. Always do your own research before making investment decisions.
