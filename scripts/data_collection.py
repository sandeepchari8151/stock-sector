"""
Data Collection Module
=====================

This module handles all data collection activities:
- Downloading data from Yahoo Finance
- Storing data in MongoDB
- Creating sample datasets for testing

Dependencies: yfinance, pandas, numpy, pymongo
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from .mongodb_utils import MongoDBManager, MongoDBContext


# Configuration for Yahoo Finance symbols
YAHOO_SYMBOLS = {
    "IT": "XLK",        # Technology Select Sector SPDR Fund
    "Pharma": "XLV",    # Health Care Select Sector SPDR Fund  
    "Banking": "XLF",   # Financial Select Sector SPDR Fund
    "FMCG": "XLP",      # Consumer Staples Select Sector SPDR Fund
}

# Data folder configuration
DATA_FOLDER = "data"


class DataCollector:
    """Handles all data collection operations."""
    
    def __init__(self, mongo_manager: Optional[MongoDBManager] = None):
        self.mongo_manager = mongo_manager or MongoDBManager()
        self.symbols = YAHOO_SYMBOLS
    
    def create_data_folder(self) -> None:
        """Create the data folder if it doesn't exist."""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"Created folder: {self.data_folder}")
        else:
            print(f"Folder already exists: {self.data_folder}")
    
    def download_sector_data(self, symbol: str, sector_name: str) -> bool:
        """
        Download 1-year historical data for a given symbol and store in MongoDB.
        
        Args:
            symbol: Yahoo Finance symbol (e.g., 'XLK')
            sector_name: Name of the sector (e.g., 'IT')
        
        Returns:
            True if successful, False otherwise
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
            
            # Prepare the data for MongoDB storage
            data = data.reset_index()
            data = data.rename(columns={'Adj Close': 'Close'})
            
            # Convert to list of dictionaries for MongoDB
            prices_data = []
            for _, row in data.iterrows():
                price_doc = {
                    "sector_code": sector_name,
                    "trade_date": row['Date'].to_pydatetime(),
                    "open_price": float(row.get('Open', 0)) if pd.notna(row.get('Open')) else None,
                    "high_price": float(row.get('High', 0)) if pd.notna(row.get('High')) else None,
                    "low_price": float(row.get('Low', 0)) if pd.notna(row.get('Low')) else None,
                    "close_price": float(row['Close']) if pd.notna(row['Close']) else None,
                    "adjusted_close": float(row['Close']) if pd.notna(row['Close']) else None,
                    "volume": int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None
                }
                prices_data.append(price_doc)
            
            # Store in MongoDB
            inserted_count = self.mongo_manager.insert_sector_etf_prices(prices_data)
            
            print(f"  ✅ Downloaded and stored {inserted_count} records for {sector_name}")
            return inserted_count > 0
            
        except Exception as e:
            print(f"  ❌ Error downloading {symbol}: {str(e)}")
            return False
    
    def download_all_sectors(self) -> bool:
        """
        Download data for all sectors defined in YAHOO_SYMBOLS and store in MongoDB.
        
        Returns:
            True if at least one sector was successfully downloaded, False otherwise
        """
        print("Starting Yahoo Finance data download...")
        print("=" * 50)
        
        # Ensure sectors exist in MongoDB
        self._ensure_sectors_exist()
        
        # Download data for each sector
        successful_downloads = 0
        
        for sector_name, symbol in self.symbols.items():
            success = self.download_sector_data(symbol, sector_name)
            if success:
                successful_downloads += 1
            print()  # Add blank line for readability
        
        print("=" * 50)
        print(f"Download complete: {successful_downloads}/{len(self.symbols)} sectors successful")
        
        # Log data quality
        self.mongo_manager.log_data_quality(
            process_name="data_collection",
            status="success" if successful_downloads > 0 else "error",
            records_processed=successful_downloads * 250,  # Approximate records per sector
            error_count=len(self.symbols) - successful_downloads
        )
        
        return successful_downloads > 0
    
    def _ensure_sectors_exist(self) -> None:
        """Ensure all sectors exist in MongoDB."""
        for sector_name, symbol in self.symbols.items():
            sector_data = {
                "code": sector_name,
                "name": self._get_sector_display_name(sector_name),
                "etf_symbol": symbol,
                "is_active": True
            }
            self.mongo_manager.upsert_sector(sector_data)
    
    def _get_sector_display_name(self, sector_code: str) -> str:
        """Get display name for sector code."""
        display_names = {
            "IT": "Information Technology",
            "Pharma": "Pharmaceuticals & Healthcare", 
            "Banking": "Financials & Banking",
            "FMCG": "Consumer Staples (FMCG)"
        }
        return display_names.get(sector_code, sector_code)
    
    def get_data_from_mongodb(self, start_date: Optional[datetime] = None, 
                             end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get sector data from MongoDB and return as DataFrame.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with Date, Sector, Close columns
        """
        try:
            all_data = []
            
            for sector_name in self.symbols.keys():
                sector_data = self.mongo_manager.get_sector_etf_prices(
                    sector_name, start_date, end_date
                )
                
                if not sector_data.empty:
                    # Convert to expected format
                    sector_df = pd.DataFrame({
                        'Date': sector_data['trade_date'],
                        'Sector': sector_name,
                        'Close': sector_data['close_price']
                    })
                    all_data.append(sector_df)
            
            if not all_data:
                return pd.DataFrame()
            
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data.sort_values(['Sector', 'Date']).reset_index(drop=True)
            
        except Exception as e:
            print(f"Error getting data from MongoDB: {e}")
            return pd.DataFrame()
    
    def save_data_to_csv(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Optional filename (defaults to timestamp)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sector_data_{timestamp}.csv"
        
        filepath = os.path.join(self.data_folder, filename)
        data.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        return filepath
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            DataFrame with loaded data
        """
        try:
            data = pd.read_csv(filepath, parse_dates=['Date'])
            print(f"Loaded {len(data)} records from {filepath}")
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return pd.DataFrame()
    
    def load_individual_sector_csvs(self, csv_paths: dict) -> pd.DataFrame:
        """
        Load data from individual sector CSV files.
        
        Args:
            csv_paths: Dictionary mapping sector names to file paths
        
        Returns:
            Combined DataFrame with all sector data
        """
        all_data = []
        
        for sector_name, filepath in csv_paths.items():
            try:
                data = pd.read_csv(filepath, parse_dates=['Date'])
                # Ensure sector column exists
                if 'Sector' not in data.columns:
                    data['Sector'] = sector_name
                all_data.append(data)
                print(f"Loaded {sector_name} data from {filepath}")
            except Exception as e:
                print(f"Error loading {sector_name} from {filepath}: {str(e)}")
        
        if not all_data:
            raise Exception("No data was successfully loaded")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    
    def create_sample_dataset(self) -> bool:
        """
        Create a synthetic sample dataset for testing and store in MongoDB.
        
        Returns:
            True if successful, False otherwise
        """
        print("Creating sample dataset...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        sectors = list(self.symbols.keys())
        dates = pd.bdate_range(start="2024-01-01", periods=100)
        
        # Define sector characteristics
        drift_by_sector = {
            "IT": 0.0008, "Pharma": 0.0005, 
            "Banking": 0.0006, "FMCG": 0.0004
        }
        vol_by_sector = {
            "IT": 0.012, "Pharma": 0.008, 
            "Banking": 0.010, "FMCG": 0.007
        }
        
        # Ensure sectors exist
        self._ensure_sectors_exist()
        
        total_inserted = 0
        
        for sector in sectors:
            start_price = 100.0
            daily_noise = np.random.normal(0, vol_by_sector[sector], len(dates))
            daily_returns = drift_by_sector[sector] + daily_noise
            price_series = start_price * np.cumprod(1.0 + daily_returns)
            
            # Create price data for MongoDB
            prices_data = []
            for i, date in enumerate(dates):
                price_doc = {
                    "sector_code": sector,
                    "trade_date": date.to_pydatetime(),
                    "open_price": float(price_series[i] * (1 + np.random.normal(0, 0.001))),
                    "high_price": float(price_series[i] * (1 + abs(np.random.normal(0, 0.002)))),
                    "low_price": float(price_series[i] * (1 - abs(np.random.normal(0, 0.002)))),
                    "close_price": float(price_series[i]),
                    "adjusted_close": float(price_series[i]),
                    "volume": int(np.random.normal(50000000, 10000000))
                }
                prices_data.append(price_doc)
            
            # Insert into MongoDB
            inserted_count = self.mongo_manager.insert_sector_etf_prices(prices_data)
            total_inserted += inserted_count
        
        print(f"Sample dataset created with {total_inserted} records")
        
        # Log data quality
        self.mongo_manager.log_data_quality(
            process_name="sample_data_creation",
            status="success",
            records_processed=total_inserted
        )
        
        return total_inserted > 0


def main():
    """Example usage of the DataCollector class."""
    try:
        with MongoDBContext() as mongo_manager:
            collector = DataCollector(mongo_manager)
            
            # Download real data
            success = collector.download_all_sectors()
            
            if success:
                # Get data from MongoDB to verify
                data = collector.get_data_from_mongodb()
                print(f"\nData shape: {data.shape}")
                print(f"Sectors: {data['Sector'].unique()}")
                print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
            else:
                print("Download failed, creating sample data...")
                collector.create_sample_dataset()
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
