import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta


# -----------------------------
# Configuration (easy to change)
# -----------------------------
# If you already have a CSV file, put its path here (e.g., "data/sector_prices.csv").
# If left as None, the script will create a small sample dataset for you and save it
# as "sample_sector_prices.csv" in the current folder.
CSV_PATH = None  # change to a string path if you have your own CSV

# Optional: Map of sector name to a Yahoo Finance CSV file path (you download locally).
# If this is non-empty, the script will load these files instead of CSV_PATH or sample.
# Each CSV should have at least columns: Date, Adj Close (or Close)
YAHOO_CSV_PATHS: dict[str, str] = {
    # Use the downloaded data from download_yahoo_data.py
    "IT": "data/IT.csv",
    "Pharma": "data/Pharma.csv",
    "Banking": "data/Banking.csv",
    "FMCG": "data/FMCG.csv",
}

# Yahoo Finance symbols for each sector (you can change these)
# These are popular US sector ETFs - replace with your preferred symbols
YAHOO_SYMBOLS = {
    "IT": "XLK",        # Technology Select Sector SPDR Fund
    "Pharma": "XLV",    # Health Care Select Sector SPDR Fund  
    "Banking": "XLF",   # Financial Select Sector SPDR Fund
    "FMCG": "XLP",      # Consumer Staples Select Sector SPDR Fund
}

# Investment distribution - will be calculated from real market data
# This will be updated automatically based on market cap or performance
INVESTMENT_DISTRIBUTION = {}


# -------------------------------------------------
# Helper function: create a small sample test dataset
# -------------------------------------------------
# This function creates simple synthetic daily closing prices for a few sectors.
# It uses a random walk so that each sector has a slightly different trend/volatility.
# It returns a pandas DataFrame and can optionally save it to a CSV file.

def create_sample_dataset(output_csv_path: str | None = None) -> pd.DataFrame:
    # Set a random seed so results are the same every time you run the script
    np.random.seed(42)

    # Define sectors you want to simulate
    sectors = ["IT", "Pharma", "Banking", "FMCG"]

    # Create a date range for our sample data (100 business days)
    dates = pd.bdate_range(start="2024-01-01", periods=100)

    # We'll build a list of DataFrames (one per sector), then combine them
    sector_frames = []

    # Define a simple drift (average daily growth) and volatility (randomness) per sector
    # These are small numbers because daily stock moves are usually small
    drift_by_sector = {
        "IT": 0.0008,      # slightly higher growth
        "Pharma": 0.0005,  # moderate growth
        "Banking": 0.0006, # moderate growth
        "FMCG": 0.0004,    # slightly lower growth
    }
    vol_by_sector = {
        "IT": 0.012,       # a bit more volatile
        "Pharma": 0.008,   # less volatile
        "Banking": 0.010,  # medium volatility
        "FMCG": 0.007,     # least volatile among these
    }

    # Generate prices for each sector using a basic geometric random walk
    for sector in sectors:
        # Start each sector at a price of 100
        start_price = 100.0

        # Simulate daily returns: drift + random noise
        # np.random.normal(0, vol, size) generates random noise
        daily_noise = np.random.normal(loc=0.0, scale=vol_by_sector[sector], size=len(dates))
        daily_returns = drift_by_sector[sector] + daily_noise

        # Convert daily returns into a price series using cumulative product
        # Price_t = Price_0 * product(1 + daily_return)
        price_series = start_price * np.cumprod(1.0 + daily_returns)

        # Build a DataFrame for this sector
        df_sector = pd.DataFrame({
            "Date": dates,
            "Sector": sector,
            "Close": price_series,
        })
        sector_frames.append(df_sector)

    # Combine all sectors into a single DataFrame
    df_all = pd.concat(sector_frames, ignore_index=True)

    # If a path is provided, save the CSV for you to reuse later
    if output_csv_path is not None:
        df_all.to_csv(output_csv_path, index=False)

    return df_all


# --------------------------------------------
# Yahoo Finance CSV loader (multiple sector files)
# --------------------------------------------
# Reads per-sector CSVs exported from Yahoo Finance. We use Adj Close if available.


def _read_single_yahoo_csv(path: str, sector_name: str) -> pd.DataFrame:
    # Read CSV and parse the Date column
    df = pd.read_csv(path, parse_dates=["Date"])  # Yahoo uses 'Date'

    # Prefer 'Adj Close' when available; otherwise fall back to 'Close'
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    # Build a normalized DataFrame with our expected columns
    df_norm = pd.DataFrame({
        "Date": df["Date"],
        "Sector": sector_name,
        "Close": df[price_col].astype(float),
    }).dropna(subset=["Close"])  # drop rows where price is missing

    return df_norm


def load_yahoo_csvs(paths_by_sector: dict[str, str]) -> pd.DataFrame:
    # Load each CSV, add the sector name, and concatenate into one DataFrame
    frames = []
    for sector, path in paths_by_sector.items():
        try:
            frames.append(_read_single_yahoo_csv(path, sector))
        except FileNotFoundError:
            print(f"Warning: File for sector '{sector}' not found at '{path}'. Skipping.")
    if not frames:
        raise FileNotFoundError("No valid Yahoo CSV files were loaded. Please check paths.")
    df_all = pd.concat(frames, ignore_index=True)
    return df_all


# ------------------------------------
# Load data from Yahoo, CSV, or sample
# ------------------------------------
# This function tries Yahoo CSVs first (if provided), then your single CSV, else sample.

def load_data(csv_path: str | None) -> pd.DataFrame:
    # First priority: Yahoo per-sector CSVs if provided
    if YAHOO_CSV_PATHS:
        try:
            df = load_yahoo_csvs(YAHOO_CSV_PATHS)
            return df
        except Exception as exc:
            print(f"Yahoo CSV load failed: {exc}. Falling back to single CSV or sample...")

    # Second: single CSV path
    if csv_path:
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            return df
        except FileNotFoundError:
            print(f"CSV not found at '{csv_path}'. Creating a sample dataset instead...")

    # Last resort: create a sample dataset
    sample_path = "sample_sector_prices.csv"
    df_sample = create_sample_dataset(output_csv_path=sample_path)
    print(f"Sample dataset created and saved to '{sample_path}'.")
    return df_sample


# ------------------------
# Analytics helper methods
# ------------------------
# These functions compute the required metrics in a clear, step-by-step way.


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    # Sort by date to ensure correct order for percentage change
    df_sorted = df.sort_values(["Sector", "Date"])  # sort within each sector

    # Group by sector, then compute pct_change on the Close column
    df_sorted["DailyReturn"] = df_sorted.groupby("Sector")["Close"].pct_change()

    # Drop the first day per sector (it has NaN return because there is no previous day)
    df_returns = df_sorted.dropna(subset=["DailyReturn"]).copy()
    return df_returns


def compute_volatility(df_returns: pd.DataFrame) -> pd.Series:
    # Standard deviation of daily returns per sector is a simple measure of volatility
    volatility_by_sector = df_returns.groupby("Sector")["DailyReturn"].std()
    return volatility_by_sector


def compute_cagr(df: pd.DataFrame) -> pd.Series:
    # CAGR = (EndingValue / StartingValue)^(1/years) - 1
    # We'll compute this for each sector using its first and last Close values and the date range
    cagr_values = {}

    # Ensure data is sorted properly
    df_sorted = df.sort_values(["Sector", "Date"])  # sort within each sector

    # Loop through each sector to compute its CAGR
    for sector, df_sector in df_sorted.groupby("Sector"):
        first_close = df_sector["Close"].iloc[0]
        last_close = df_sector["Close"].iloc[-1]

        # Calculate the number of years between first and last date
        num_days = (df_sector["Date"].iloc[-1] - df_sector["Date"].iloc[0]).days
        years = max(num_days / 365.25, 1e-9)  # avoid division by zero

        cagr = (last_close / first_close) ** (1.0 / years) - 1.0
        cagr_values[sector] = cagr

    # Convert dictionary to a pandas Series for convenience
    return pd.Series(cagr_values)


def compute_cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    # Compute cumulative return index per sector, rebased to 1.0 on the first available day
    df_sorted = df.sort_values(["Sector", "Date"]).copy()
    df_sorted["CumReturnIndex"] = df_sorted.groupby("Sector")["Close"].transform(
        lambda s: s / s.iloc[0]
    )
    return df_sorted


def compute_average_daily_returns(df_returns: pd.DataFrame) -> pd.Series:
    # Mean of daily returns per sector across the whole period
    avg_returns = df_returns.groupby("Sector")["DailyReturn"].mean()
    return avg_returns


def calculate_real_investment_distribution(df: pd.DataFrame, method: str = "market_cap") -> dict:
    """
    Calculate investment distribution based on real market data.
    
    Args:
        df: DataFrame with Date, Sector, Close columns
        method: "market_cap" (based on latest prices), "performance" (based on returns), or "equal_weight"
    
    Returns:
        Dictionary with sector names as keys and weights as values
    """
    if method == "equal_weight":
        # Equal weight distribution
        sectors = df["Sector"].unique()
        weight = 1.0 / len(sectors)
        return {sector: weight for sector in sectors}
    
    elif method == "market_cap":
        # Weight by latest market price (proxy for market cap)
        latest_prices = df.groupby("Sector")["Close"].last()
        total_value = latest_prices.sum()
        weights = latest_prices / total_value
        return weights.to_dict()
    
    elif method == "performance":
        # Weight by cumulative performance (higher performing sectors get more weight)
        df_sorted = df.sort_values(["Sector", "Date"])
        df_sorted["CumReturnIndex"] = df_sorted.groupby("Sector")["Close"].transform(
            lambda s: s / s.iloc[0]
        )
        latest_performance = df_sorted.groupby("Sector")["CumReturnIndex"].last()
        # Normalize so weights sum to 1
        total_performance = latest_performance.sum()
        weights = latest_performance / total_performance
        return weights.to_dict()
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'market_cap', 'performance', or 'equal_weight'")


# -----------------
# Plotting functions
# -----------------
# Simple visualizations using matplotlib.


def plot_sector_performance_over_time(df_cum: pd.DataFrame) -> None:
    # Pivot the data so that each sector becomes a column, indexed by Date
    pivot = df_cum.pivot(index="Date", columns="Sector", values="CumReturnIndex")

    # Create a line plot where each sector is a separate line
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each sector with different colors and styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional color palette
    styles = ['-', '--', '-.', ':']  # Different line styles
    
    for i, (sector, series) in enumerate(pivot.items()):
        ax.plot(series.index, series.values, 
                label=f'{sector} (Final: {series.iloc[-1]:.2f}x)', 
                linewidth=2.5, 
                color=colors[i % len(colors)],
                linestyle=styles[i % len(styles)])
    
    # Add horizontal line at 1.0 (starting point)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Starting Point')
    
    # Formatting
    ax.set_title("Real Sector Performance Over Time (1-Year Period)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return Index", fontsize=12)
    ax.legend(title="Sector Performance", loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.tick_params(axis='x', rotation=45)
    
    # Add performance summary text
    best_sector = pivot.iloc[-1].idxmax()
    worst_sector = pivot.iloc[-1].idxmin()
    best_return = pivot.iloc[-1].max()
    worst_return = pivot.iloc[-1].min()
    
    summary_text = f"Best: {best_sector} ({best_return:.1%})\nWorst: {worst_sector} ({worst_return:.1%})"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("plot_sector_performance_over_time.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_investment_distribution_bar(distribution: dict) -> None:
    # Bar chart for investment per sector (weights) - now with real market data
    sectors = list(distribution.keys())
    weights = list(distribution.values())

    total = sum(weights)
    if not np.isclose(total, 1.0):
        weights = [w / total for w in weights]

    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort sectors by weight for better visualization
    sorted_data = sorted(zip(sectors, weights), key=lambda x: x[1], reverse=True)
    sorted_sectors, sorted_weights = zip(*sorted_data)
    
    # Use gradient colors based on weight
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_sectors)))
    
    bars = ax.bar(sorted_sectors, sorted_weights, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    for i, (bar, weight) in enumerate(zip(bars, sorted_weights)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{weight:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Formatting
    ax.set_title("Real Market-Based Investment Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Sector", fontsize=12)
    ax.set_ylabel("Portfolio Weight", fontsize=12)
    ax.set_ylim(0, max(sorted_weights) * 1.15)
    
    # Add grid and formatting
    ax.grid(axis="y", alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add total value text
    total_text = f"Total Allocation: {sum(sorted_weights):.1%}"
    ax.text(0.02, 0.98, total_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("plot_investment_distribution_bar.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_volatility_vs_growth_heatmap(volatility: pd.Series, cagr: pd.Series) -> None:
    # Align indices (sectors) to ensure same order
    sectors = sorted(set(volatility.index).intersection(set(cagr.index)))
    vol_values = volatility.reindex(sectors).values
    cagr_values = cagr.reindex(sectors).values

    # Build a 2 x N matrix: first row = volatility, second row = CAGR
    data = np.vstack([vol_values, cagr_values])

    # Create enhanced heatmap with better styling
    fig, ax = plt.subplots(figsize=(max(8, len(sectors) * 1.5), 6))
    
    # Use a better colormap and normalize data for better visualization
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", interpolation='nearest')
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Value", fontsize=12)

    # Set ticks and labels with better formatting
    ax.set_xticks(np.arange(len(sectors)))
    ax.set_xticklabels(sectors, rotation=45, ha="right", fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Volatility (Daily Std)", "Growth (CAGR)"], fontsize=11)

    # Overlay numeric values with better contrast
    for y in range(2):
        for x in range(len(sectors)):
            value = data[y, x]
            # Format based on row type
            if y == 1:  # CAGR row
                text = f"{value:.1%}"
            else:  # Volatility row
                text = f"{value:.2%}"
            
            # Choose text color based on background
            text_color = "white" if value < data.max()/2 else "black"
            
            ax.text(x, y, text, ha="center", va="center", 
                   color=text_color, fontweight='bold', fontsize=10)

    # Enhanced title and formatting
    ax.set_title("Real Market Data: Volatility vs Growth Analysis", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(sectors)) - 0.5, minor=True)
    ax.set_yticks(np.arange(2) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    # Add summary statistics
    max_vol_sector = sectors[np.argmax(vol_values)]
    max_growth_sector = sectors[np.argmax(cagr_values)]
    
    summary_text = f"Highest Volatility: {max_vol_sector}\nHighest Growth: {max_growth_sector}"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("plot_volatility_vs_growth_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()


# ------------------
# Insight generation
# ------------------
# Produce simple, plain-English statements from the metrics we computed.


def print_plain_english_insights(volatility: pd.Series,
                                 cagr: pd.Series,
                                 last_cum_returns: pd.Series,
                                 avg_daily_returns: pd.Series) -> None:
    # Identify sectors based on metrics
    most_volatile = volatility.idxmax()
    least_volatile = volatility.idxmin()
    best_cagr_sector = cagr.idxmax()
    best_total_return_sector = last_cum_returns.idxmax()
    best_avg_return_sector = avg_daily_returns.idxmax()

    # Print clear, short insights
    print("\nInsights:")
    print(f"- Most volatile sector: {most_volatile} (std of daily returns = {volatility[most_volatile]:.4f})")
    print(f"- Least volatile sector: {least_volatile} (std of daily returns = {volatility[least_volatile]:.4f})")
    print(f"- Highest CAGR: {best_cagr_sector} (CAGR = {cagr[best_cagr_sector]:.2%})")
    print(f"- Best overall performer (total period): {best_total_return_sector} (Cumulative index = {last_cum_returns[best_total_return_sector]:.2f}x)")
    print(f"- Highest average daily return: {best_avg_return_sector} ({avg_daily_returns[best_avg_return_sector]:.4%} per day)")


# -----
# Main
# -----
# Orchestrate the steps: load/generate data, run analytics, plot, and print insights.


def main() -> None:
    # 1) Load data (Yahoo per-sector CSVs if provided, else single CSV, else sample)
    df = load_data(CSV_PATH)

    # Expect the DataFrame to have these columns: Date, Sector, Close
    # If you bring your own CSV, ensure it matches these column names.

    # 2) Calculate real investment distribution based on market data
    # You can change the method: "market_cap", "performance", or "equal_weight"
    global INVESTMENT_DISTRIBUTION
    INVESTMENT_DISTRIBUTION = calculate_real_investment_distribution(df, method="market_cap")
    
    print("Real Investment Distribution (based on market cap):")
    for sector, weight in INVESTMENT_DISTRIBUTION.items():
        print(f"  {sector}: {weight:.1%}")
    print()

    # 3) Compute analytics
    df_returns = calculate_daily_returns(df)
    volatility = compute_volatility(df_returns)
    cagr = compute_cagr(df)

    # Cumulative return index over time
    df_cum = compute_cumulative_returns(df)

    # Last cumulative index per sector tells us which sector did best over the full period
    last_cum = df_cum.sort_values("Date").groupby("Sector")["CumReturnIndex"].last()

    # Average daily return across the period
    avg_daily_returns = compute_average_daily_returns(df_returns)

    # 4) Visualizations per new spec
    # Line graph: Sector trends over time
    plot_sector_performance_over_time(df_cum)

    # Bar chart: Investment per sector (now using real data)
    plot_investment_distribution_bar(INVESTMENT_DISTRIBUTION)

    # Heatmap: Volatility vs Growth (CAGR)
    plot_volatility_vs_growth_heatmap(volatility, cagr)

    # 5) Plain-English insights
    print_plain_english_insights(
        volatility=volatility,
        cagr=cagr,
        last_cum_returns=last_cum,
        avg_daily_returns=avg_daily_returns,
    )


if __name__ == "__main__":
    main()
