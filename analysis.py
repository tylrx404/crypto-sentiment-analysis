"""
Trader Performance vs Bitcoin Market Sentiment Analysis
=======================================================
Analyzes how Fear/Greed sentiment correlates with trader performance
on Hyperliquid using historical trade data and the Fear & Greed Index.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
SENTIMENT_PATH = "fear_greed_index.csv"
TRADES_PATH    = "historical_data.csv"
OUTPUT_DIR     = "outputs"

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
PALETTE = {
    "Extreme Fear": "#d62728",
    "Fear":         "#ff7f0e",
    "Neutral":      "#7f7f7f",
    "Greed":        "#2ca02c",
    "Extreme Greed":"#1f77b4",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
def load_sentiment(path: str) -> pd.DataFrame:
    """Load and parse the Fear & Greed Index CSV."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[["date", "classification"]].rename(columns={"classification": "sentiment"})
    df["sentiment"] = df["sentiment"].str.strip()
    return df


def load_trades(path: str) -> pd.DataFrame:
    """Load and parse the Hyperliquid historical trade CSV."""
    df = pd.read_csv(path, dtype={"Closed PnL": float})

    # Parse timestamp: format is DD-MM-YYYY HH:MM
    df["trade_date"] = pd.to_datetime(
        df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    ).dt.normalize()

    # Keep only rows with valid PnL-bearing events (closed trades)
    df = df[df["Closed PnL"] != 0].copy()

    # Rename for convenience
    df = df.rename(columns={
        "Account":         "account",
        "Coin":            "coin",
        "Execution Price": "exec_price",
        "Size USD":        "size_usd",
        "Side":            "side",
        "Direction":       "direction",
        "Closed PnL":      "pnl",
        "Fee":             "fee",
        "Crossed":         "is_cross_margin",
    })

    df["is_win"] = df["pnl"] > 0
    df["net_pnl"] = df["pnl"] - df["fee"].fillna(0)

    return df[[
        "account", "coin", "exec_price", "size_usd",
        "side", "direction", "pnl", "fee", "net_pnl",
        "is_win", "is_cross_margin", "trade_date"
    ]]


# ── 2. Merge ─────────────────────────────────────────────────────────────────
def merge_datasets(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Merge trade data with daily sentiment on trade_date == date."""
    merged = trades.merge(sentiment, left_on="trade_date", right_on="date", how="inner")
    merged.drop(columns=["date"], inplace=True)
    # Enforce ordered categorical for correct chart ordering
    merged["sentiment"] = pd.Categorical(
        merged["sentiment"], categories=SENTIMENT_ORDER, ordered=True
    )
    print(f"[INFO] Merged dataset: {len(merged):,} rows across "
          f"{merged['trade_date'].nunique()} unique trading days.")
    return merged


# ── 3. Compute Summary Stats ─────────────────────────────────────────────────
def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate key metrics by sentiment class."""
    summary = (
        df.groupby("sentiment", observed=True)
        .agg(
            trade_count   = ("pnl", "count"),
            avg_pnl       = ("pnl", "mean"),
            median_pnl    = ("pnl", "median"),
            total_pnl     = ("pnl", "sum"),
            win_rate      = ("is_win", "mean"),
            avg_size_usd  = ("size_usd", "mean"),
            avg_net_pnl   = ("net_pnl", "mean"),
        )
        .reset_index()
    )
    summary["win_rate_pct"] = summary["win_rate"] * 100
    return summary


def compute_side_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate split by sentiment × trade side (BUY/SELL)."""
    return (
        df.groupby(["sentiment", "side"], observed=True)
        .agg(win_rate=("is_win", "mean"), count=("pnl", "count"))
        .reset_index()
    )


# ── 4. Visualisations ────────────────────────────────────────────────────────
def _bar_colors(labels):
    return [PALETTE.get(l, "#999999") for l in labels]


def plot_avg_pnl(summary: pd.DataFrame, out_dir: str):
    """Bar chart: Average PnL per trade by sentiment."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        summary["sentiment"], summary["avg_pnl"],
        color=_bar_colors(summary["sentiment"]), edgecolor="white", linewidth=0.6
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.bar_label(bars, fmt="$%.1f", padding=4, fontsize=9)
    ax.set_title("Average PnL per Trade by Market Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment", fontsize=11)
    ax.set_ylabel("Avg PnL (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    sns.despine()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/01_avg_pnl_by_sentiment.png", dpi=150)
    plt.close(fig)
    print("[SAVED] 01_avg_pnl_by_sentiment.png")


def plot_win_rate(summary: pd.DataFrame, out_dir: str):
    """Bar chart: Win rate % by sentiment."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        summary["sentiment"], summary["win_rate_pct"],
        color=_bar_colors(summary["sentiment"]), edgecolor="white", linewidth=0.6
    )
    ax.axhline(50, color="black", linewidth=0.8, linestyle="--", label="50% baseline")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_title("Win Rate by Market Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment", fontsize=11)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/02_win_rate_by_sentiment.png", dpi=150)
    plt.close(fig)
    print("[SAVED] 02_win_rate_by_sentiment.png")


def plot_trade_count(summary: pd.DataFrame, out_dir: str):
    """Bar chart: Trade volume by sentiment."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        summary["sentiment"], summary["trade_count"],
        color=_bar_colors(summary["sentiment"]), edgecolor="white", linewidth=0.6
    )
    ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=9)
    ax.set_title("Trade Frequency by Market Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment", fontsize=11)
    ax.set_ylabel("Number of Trades", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    sns.despine()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/03_trade_count_by_sentiment.png", dpi=150)
    plt.close(fig)
    print("[SAVED] 03_trade_count_by_sentiment.png")


def plot_pnl_boxplot(df: pd.DataFrame, out_dir: str):
    """Boxplot: PnL distribution by sentiment (capped outliers for readability)."""
    q_low  = df["pnl"].quantile(0.02)
    q_high = df["pnl"].quantile(0.98)
    clipped = df[(df["pnl"] >= q_low) & (df["pnl"] <= q_high)].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=clipped, x="sentiment", y="pnl", order=SENTIMENT_ORDER,
        palette=PALETTE, linewidth=0.8, fliersize=2, ax=ax
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("PnL Distribution by Market Sentiment (2nd–98th Pct)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment", fontsize=11)
    ax.set_ylabel("Closed PnL (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    sns.despine()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/04_pnl_distribution_boxplot.png", dpi=150)
    plt.close(fig)
    print("[SAVED] 04_pnl_distribution_boxplot.png")


def plot_side_winrate(side_df: pd.DataFrame, out_dir: str):
    """Grouped bar: Win rate by sentiment × side (BUY/SELL)."""
    pivot = side_df.pivot(index="sentiment", columns="side", values="win_rate") * 100
    pivot = pivot.reindex(SENTIMENT_ORDER).dropna(how="all")

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="white", linewidth=0.6)
    ax.axhline(50, color="black", linewidth=0.8, linestyle="--", label="50% baseline")
    ax.set_title("Win Rate by Sentiment & Trade Side", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment", fontsize=11)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(title="Side", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    sns.despine()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/05_side_winrate_by_sentiment.png", dpi=150)
    plt.close(fig)
    print("[SAVED] 05_side_winrate_by_sentiment.png")


# ── 5. Insights ───────────────────────────────────────────────────────────────
def print_insights(summary: pd.DataFrame, df: pd.DataFrame):
    best  = summary.loc[summary["avg_pnl"].idxmax(), "sentiment"]
    worst = summary.loc[summary["avg_pnl"].idxmin(), "sentiment"]
    highest_wr = summary.loc[summary["win_rate_pct"].idxmax(), "sentiment"]
    most_active = summary.loc[summary["trade_count"].idxmax(), "sentiment"]
    best_pnl    = summary.loc[summary["avg_pnl"].idxmax(), "avg_pnl"]
    worst_pnl   = summary.loc[summary["avg_pnl"].idxmin(), "avg_pnl"]

    fear_wr   = summary.loc[summary["sentiment"].isin(["Fear","Extreme Fear"]),"win_rate_pct"].mean()
    greed_wr  = summary.loc[summary["sentiment"].isin(["Greed","Extreme Greed"]),"win_rate_pct"].mean()
    fear_pnl  = summary.loc[summary["sentiment"].isin(["Fear","Extreme Fear"]),"avg_pnl"].mean()
    greed_pnl = summary.loc[summary["sentiment"].isin(["Greed","Extreme Greed"]),"avg_pnl"].mean()

    print("\n" + "="*60)
    print("  KEY INSIGHTS: TRADER PERFORMANCE vs MARKET SENTIMENT")
    print("="*60)

    insights = [
        (1, f"Best returns in '{best}' periods",
            f"Traders earned an average of ${best_pnl:,.2f} per trade during "
            f"'{best}' sentiment — the highest across all sentiment classes."),

        (2, f"Worst returns in '{worst}' periods",
            f"'{worst}' sentiment generated the weakest average PnL "
            f"(${worst_pnl:,.2f} per trade), suggesting heightened risk during "
            f"extreme sentiment swings."),

        (3, f"'{highest_wr}' yields the highest win rate",
            f"Trades placed during '{highest_wr}' had the best win-rate "
            f"({summary.loc[summary['sentiment']==highest_wr,'win_rate_pct'].values[0]:.1f}%), "
            f"making it the most reliable environment for profitable entries."),

        (4, "Fear vs Greed: win-rate comparison",
            f"Fear-regime (Fear + Extreme Fear) average win rate: {fear_wr:.1f}%. "
            f"Greed-regime (Greed + Extreme Greed) average win rate: {greed_wr:.1f}%. "
            + ("Greed periods see more consistent wins." if greed_wr > fear_wr
               else "Fear periods surprisingly produce more consistent wins.")),

        (5, "Fear vs Greed: PnL comparison",
            f"Average trade PnL — Fear regimes: ${fear_pnl:,.2f} | "
            f"Greed regimes: ${greed_pnl:,.2f}. "
            + ("Greed markets deliver higher per-trade profits." if greed_pnl > fear_pnl
               else "Fear markets deliver higher per-trade profits on average.")),

        (6, f"Highest trading activity in '{most_active}' periods",
            f"'{most_active}' days saw {summary.loc[summary['sentiment']==most_active,'trade_count'].values[0]:,} "
            f"closed trades — the most of any sentiment. This indicates traders are most "
            f"active (and therefore most exposed) during that sentiment regime."),

        (7, "Wide PnL dispersion across all sentiments",
            f"With trade PnL ranging from ${df['pnl'].min():,.0f} to ${df['pnl'].max():,.0f}, "
            f"extreme outcomes occur in every sentiment class. Risk management remains "
            f"essential regardless of market mood."),
    ]

    for num, title, body in insights:
        print(f"\n  [{num}] {title}")
        print(f"      {body}")

    print("\n" + "="*60 + "\n")


# ── 6. Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n[INFO] Loading datasets...")
    sentiment = load_sentiment(SENTIMENT_PATH)
    trades    = load_trades(TRADES_PATH)
    print(f"[INFO] Sentiment rows: {len(sentiment):,} | Trade rows (non-zero PnL): {len(trades):,}")

    print("[INFO] Merging on trade date...")
    df = merge_datasets(trades, sentiment)

    print("[INFO] Computing summary statistics...")
    summary    = compute_summary(df)
    side_break = compute_side_breakdown(df)

    print("\n── Summary Table ──────────────────────────────────────────")
    display_cols = ["sentiment","trade_count","avg_pnl","win_rate_pct","total_pnl","avg_size_usd"]
    print(summary[display_cols].to_string(index=False, float_format="%.2f"))

    print("\n[INFO] Generating visualisations...")
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plot_avg_pnl(summary, OUTPUT_DIR)
    plot_win_rate(summary, OUTPUT_DIR)
    plot_trade_count(summary, OUTPUT_DIR)
    plot_pnl_boxplot(df, OUTPUT_DIR)
    plot_side_winrate(side_break, OUTPUT_DIR)

    print_insights(summary, df)

    # Save cleaned merged dataset preview
    preview_path = f"{OUTPUT_DIR}/merged_preview.csv"
    df.head(200).to_csv(preview_path, index=False)
    print(f"[INFO] Cleaned dataset preview saved → {preview_path}")

    return df, summary


if __name__ == "__main__":
    df, summary = main()
