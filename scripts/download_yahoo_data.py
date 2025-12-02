"""
Yahoo Finance Data Downloader
============================

This script downloads 1-year historical data from Yahoo Finance for different sectors
and saves them as CSV files in a 'data' folder.

Requirements:
- pip install yfinance pandas

Usage:
    python download_yahoo_data.py

The script will:
1. Create a 'data' folder if it doesn't exist
2. Download 1-year data for each sector symbol
3. Save each as a separate CSV file
4. Print progress and any errors
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta


# Configuration: Yahoo Finance symbols for each sector
# You can change these to any symbols you want to track
SECTOR_SYMBOLS = {
    "IT": "XLK",        # Technology Select Sector SPDR Fund
    "Pharma": "XLV",    # Health Care Select Sector SPDR Fund  
    "Banking": "XLF",   # Financial Select Sector SPDR Fund
    "FMCG": "XLP",      # Consumer Staples Select Sector SPDR Fund
}

# Data folder name
DATA_FOLDER = "data"


def create_data_folder(folder_name: str) -> None:
    """Create the data folder if it doesn't exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    else:
        print(f"Folder already exists: {folder_name}")


def download_sector_data(symbol: str, sector_name: str, data_folder: str) -> bool:
    """
    Download 1-year historical data for a given symbol and save as CSV.
    
    Args:
        symbol: Yahoo Finance symbol (e.g., 'XLK')
        sector_name: Name of the sector (e.g., 'IT')
        data_folder: Folder to save the CSV file
    
    Returns:
        True if successful, False if failed
    """
    try:
        print(f"Downloading data for {sector_name} ({symbol})...")
        
        # Calculate date range (1 year ago to today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Download data using yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"  ❌ No data found for {symbol}")
            return False
        
        # Prepare the data for our analytics script
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Rename columns to match our expected format
        data = data.rename(columns={
            'Date': 'Date',
            'Adj Close': 'Close'  # Use Adj Close as our main price column
        })
        
        # Select only the columns we need
        data = data[['Date', 'Close']].copy()
        
        # Add sector name as a column
        data['Sector'] = sector_name
        
        # Reorder columns to match our expected format
        data = data[['Date', 'Sector', 'Close']]
        
        # Save to CSV
        filename = f"{sector_name}.csv"
        filepath = os.path.join(data_folder, filename)
        data.to_csv(filepath, index=False)
        
        print(f"  ✅ Saved {len(data)} records to {filepath}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error downloading {symbol}: {str(e)}")
        return False


def download_all_sectors() -> None:
    """Download data for all sectors defined in SECTOR_SYMBOLS."""
    print("Starting Yahoo Finance data download...")
    print("=" * 50)
    
    # Create data folder
    create_data_folder(DATA_FOLDER)
    
    # Track success/failure
    successful_downloads = 0
    total_downloads = len(SECTOR_SYMBOLS)
    
    # Download data for each sector
    for sector_name, symbol in SECTOR_SYMBOLS.items():
        success = download_sector_data(symbol, sector_name, DATA_FOLDER)
        if success:
            successful_downloads += 1
        print()  # Add blank line for readability
    
    # Summary
    print("=" * 50)
    print(f"Download complete: {successful_downloads}/{total_downloads} sectors successful")
    
    if successful_downloads == total_downloads:
        print("🎉 All downloads successful!")
        print(f"\nNext steps:")
        print(f"1. Use these files with main.py or data_collection.py:")
        print(f"   YAHOO_CSV_PATHS = {{")
        for sector_name in SECTOR_SYMBOLS.keys():
            print(f'       "{sector_name}": "data/{sector_name}.csv",')
        print(f"   }}")
        print(f"2. Run: python main.py")
    else:
        print("⚠️  Some downloads failed. Check the error messages above.")


def main():
    """Main function to run the download process."""
    print("Yahoo Finance Data Downloader")
    print("Downloading 1-year historical data for sector analysis")
    print()
    
    # Check if yfinance is available
    try:
        import yfinance
        print(f"Using yfinance version: {yfinance.__version__}")
    except ImportError:
        print("❌ yfinance not found. Please install it with:")
        print("   pip install yfinance")
        return
    
    # Start download process
    download_all_sectors()


if __name__ == "__main__":
    main()
