# Trader Performance vs Bitcoin Market Sentiment

**Primetrade.ai Internship Assignment — Data Science**  
Analyzing the relationship between Bitcoin Fear & Greed sentiment and trader performance on Hyperliquid.

---

## Project Overview

This project explores whether Bitcoin market sentiment (Fear, Greed, Neutral, etc.) has a meaningful impact on trader outcomes. Using ~104K closed trades from Hyperliquid and the Bitcoin Fear & Greed Index, we compute average PnL, win rates, trade volume, and side behavior across all five sentiment classes.

---

## Dataset Description

| Dataset | Source | Rows | Key Columns |
|---|---|---|---|
| `fear_greed_index.csv` | Bitcoin Fear & Greed Index | 2,644 | `date`, `classification` |
| `historical_data.csv` | Hyperliquid Historical Trades | 211,224 | `Account`, `Closed PnL`, `Side`, `Timestamp IST`, `Fee`, etc. |

After filtering to trades with non-zero Closed PnL and merging on trade date, the analysis covers **104,402 trades** across **419 unique trading days**.

---

## Setup

### Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn

### Install dependencies

```bash
pip install -r requirements.txt
```

### File structure

```
trading_sentiment/
├── analysis.py              # Main analysis script
├── fear_greed_index.csv     # Sentiment dataset
├── historical_data.csv      # Trade dataset
├── requirements.txt
├── README.md
└── outputs/
    ├── 01_avg_pnl_by_sentiment.png
    ├── 02_win_rate_by_sentiment.png
    ├── 03_trade_count_by_sentiment.png
    ├── 04_pnl_distribution_boxplot.png
    ├── 05_side_winrate_by_sentiment.png
    └── merged_preview.csv
```

---

## How to Run

```bash
# Place both CSV files in the same directory as analysis.py, then:
python3 analysis.py
```

All charts are saved to `outputs/`. A 200-row preview of the merged dataset is saved as `outputs/merged_preview.csv`.

---

## Key Insights

| # | Insight |
|---|---|
| 1 | **Extreme Greed → Best PnL**: Average trade PnL of **$130.21** — highest of all sentiment classes |
| 2 | **Extreme Fear → Lowest PnL**: Average trade PnL of **$71.03** — weakest returns per trade |
| 3 | **Extreme Greed → Best Win Rate**: **89.2%** of trades were profitable during Extreme Greed periods |
| 4 | **Greed beats Fear on win rate**: Greed-regime avg win rate (83.0%) > Fear-regime avg (81.8%) |
| 5 | **Greed beats Fear on PnL**: Greed-regime avg PnL ($107.80) > Fear-regime avg ($91.83) |
| 6 | **Fear → Highest activity**: Most trades (29,808) occurred during Fear periods — peak exposure risk |
| 7 | **PnL extremes exist everywhere**: Outcomes range from -$117,990 to +$135,329 across all sentiments; risk management is always essential |
| 8 | Note:
The dataset shows unusually high win rates across all sentiment categories, which may indicate dataset bias (e.g., incomplete loss data or filtered trades).|
| 9 | Traders appear to follow momentum-based strategies, with peak performance during Extreme Greed. This suggests traders capitalize on strong bullish trends rather than acting contrarian during Fear phases.


---

## Assumptions

1. **Only closed trades analyzed**: Rows with `Closed PnL == 0` are excluded as they represent open or unclosed positions, not realized outcomes.
2. **Date alignment**: Trade timestamps use the `Timestamp IST` column (format `DD-MM-YYYY HH:MM`). Sentiment is matched by calendar date (no timezone conversion applied beyond IST→date normalization).
3. **Net PnL** is computed as `Closed PnL - Fee` for internal reference; all primary charts use `Closed PnL`.
4. **Outlier handling in boxplot**: The PnL distribution plot clips to the 2nd–98th percentile for readability — extreme outliers exist but are not lost from the summary stats.
5. **No leverage data**: The raw dataset does not include a leverage column; leverage-based analysis was omitted.
6. **Inner join on date**: ~6 rows were dropped where no matching sentiment date was found (all at date boundaries).

---

## Tech Stack

- **pandas** — data loading, cleaning, merging, aggregation
- **matplotlib** — base charting engine
- **seaborn** — styled plots (boxplot, theme)
- **numpy** — numerical utilities
