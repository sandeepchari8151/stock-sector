"""
Data Preprocessing Module
========================

This module handles data cleaning, filtering, and preparation:
- Data validation and quality checks
- Date range filtering
- Missing data handling
- Data normalization and standardization
- MongoDB integration for data storage and retrieval

Dependencies: pandas, numpy, pymongo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from .mongodb_utils import MongoDBManager


class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self, mongo_manager: Optional[MongoDBManager] = None):
        self.mongo_manager = mongo_manager or MongoDBManager()
        self.required_columns = ['Date', 'Sector', 'Close']
        self.data_quality_report = {}
    
    def get_data_from_mongodb(self, start_date: Optional[datetime] = None, 
                             end_date: Optional[datetime] = None,
                             sectors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get data from MongoDB and return as DataFrame.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            sectors: Optional list of sectors to include
            
        Returns:
            DataFrame with Date, Sector, Close columns
        """
        try:
            all_data = []
            
            # Get sectors to process
            if sectors is None:
                sectors_data = self.mongo_manager.get_sectors()
                sectors = [s['code'] for s in sectors_data if s.get('is_active', True)]
            
            for sector_code in sectors:
                sector_data = self.mongo_manager.get_sector_etf_prices(
                    sector_code, start_date, end_date
                )
                
                if not sector_data.empty:
                    # Convert to expected format
                    sector_df = pd.DataFrame({
                        'Date': sector_data['trade_date'],
                        'Sector': sector_code,
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
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data structure and quality.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for empty DataFrame
        if data.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check data types
        if 'Date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['Date']):
            issues.append("Date column is not datetime type")
        
        if 'Close' in data.columns and not pd.api.types.is_numeric_dtype(data['Close']):
            issues.append("Close column is not numeric type")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.any():
            issues.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check for negative prices
        if 'Close' in data.columns:
            negative_prices = (data['Close'] <= 0).sum()
            if negative_prices > 0:
                issues.append(f"Found {negative_prices} non-positive prices")
        
        # Check for duplicate rows
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling common issues.
        
        Args:
            data: Raw DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        original_shape = data.shape
        
        # Make a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Remove duplicate rows
        cleaned_data = cleaned_data.drop_duplicates()
        
        # Remove rows with missing critical data
        cleaned_data = cleaned_data.dropna(subset=['Date', 'Sector', 'Close'])
        
        # Remove rows with non-positive prices
        cleaned_data = cleaned_data[cleaned_data['Close'] > 0]
        
        # Sort by Date and Sector
        cleaned_data = cleaned_data.sort_values(['Sector', 'Date']).reset_index(drop=True)
        
        # Ensure Date is datetime (handle timezone-aware dates)
        if not pd.api.types.is_datetime64_any_dtype(cleaned_data['Date']):
            cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'], utc=True)
        
        # Convert timezone-aware dates to timezone-naive
        if hasattr(cleaned_data['Date'].dtype, 'tz') and cleaned_data['Date'].dtype.tz is not None:
            cleaned_data['Date'] = cleaned_data['Date'].dt.tz_localize(None)
        
        # Ensure Close is numeric
        cleaned_data['Close'] = pd.to_numeric(cleaned_data['Close'], errors='coerce')
        
        # Remove any rows that became NaN after conversion
        cleaned_data = cleaned_data.dropna(subset=['Close'])
        
        final_shape = cleaned_data.shape
        removed_rows = original_shape[0] - final_shape[0]
        
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows during cleaning")
        
        print(f"Data shape: {original_shape} -> {final_shape}")
        return cleaned_data
    
    def filter_date_range(self, data: pd.DataFrame, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            data: DataFrame with Date column
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
        
        Returns:
            Filtered DataFrame
        """
        filtered_data = data.copy()
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data['Date'] >= start_dt]
            print(f"Filtered data from {start_date}")
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data['Date'] <= end_dt]
            print(f"Filtered data until {end_date}")
        
        return filtered_data
    
    def filter_sectors(self, data: pd.DataFrame, sectors: List[str]) -> pd.DataFrame:
        """
        Filter data to include only specified sectors.
        
        Args:
            data: DataFrame with Sector column
            sectors: List of sector names to include
        
        Returns:
            Filtered DataFrame
        """
        available_sectors = data['Sector'].unique()
        valid_sectors = [s for s in sectors if s in available_sectors]
        
        if not valid_sectors:
            print(f"Warning: No valid sectors found. Available: {available_sectors}")
            return data
        
        filtered_data = data[data['Sector'].isin(valid_sectors)].copy()
        print(f"Filtered to sectors: {valid_sectors}")
        
        return filtered_data
    
    def handle_missing_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing dates by forward-filling prices.
        
        Args:
            data: DataFrame with Date, Sector, Close columns
        
        Returns:
            DataFrame with missing dates filled
        """
        filled_data = data.copy()
        
        # Create complete date range for each sector
        all_dates = pd.date_range(
            start=data['Date'].min(), 
            end=data['Date'].max(), 
            freq='D'
        )
        
        # Filter to business days only
        business_dates = all_dates[all_dates.weekday < 5]
        
        sectors = data['Sector'].unique()
        complete_data = []
        
        for sector in sectors:
            sector_data = data[data['Sector'] == sector].copy()
            
            # Create complete date range for this sector
            sector_dates = pd.DataFrame({
                'Date': business_dates,
                'Sector': sector
            })
            
            # Merge with actual data
            merged = sector_dates.merge(sector_data, on=['Date', 'Sector'], how='left')
            
            # Forward fill missing prices
            merged['Close'] = merged['Close'].ffill()
            
            # Remove rows where we couldn't fill (before first actual data point)
            merged = merged.dropna(subset=['Close'])
            
            complete_data.append(merged)
        
        result = pd.concat(complete_data, ignore_index=True)
        print(f"Handled missing dates. Shape: {data.shape} -> {result.shape}")
        
        return result
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect and optionally remove outliers.
        
        Args:
            data: DataFrame with Close column
            method: 'iqr' (Interquartile Range) or 'zscore'
        
        Returns:
            DataFrame with outlier information
        """
        outlier_data = data.copy()
        
        if method == 'iqr':
            # IQR method
            for sector in data['Sector'].unique():
                sector_mask = outlier_data['Sector'] == sector
                prices = outlier_data.loc[sector_mask, 'Close']
                
                Q1 = prices.quantile(0.25)
                Q3 = prices.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (prices < lower_bound) | (prices > upper_bound)
                outlier_data.loc[sector_mask, 'is_outlier'] = outlier_mask
        
        elif method == 'zscore':
            # Z-score method
            for sector in data['Sector'].unique():
                sector_mask = outlier_data['Sector'] == sector
                prices = outlier_data.loc[sector_mask, 'Close']
                
                z_scores = np.abs((prices - prices.mean()) / prices.std())
                outlier_data.loc[sector_mask, 'is_outlier'] = z_scores > 3
        
        outlier_count = outlier_data['is_outlier'].sum()
        print(f"Detected {outlier_count} outliers using {method} method")
        
        return outlier_data
    
    def remove_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the data.
        
        Args:
            data: DataFrame with Close column
            method: 'iqr' or 'zscore'
        
        Returns:
            DataFrame with outliers removed
        """
        data_with_outliers = self.detect_outliers(data, method)
        cleaned_data = data_with_outliers[~data_with_outliers['is_outlier']].copy()
        cleaned_data = cleaned_data.drop('is_outlier', axis=1)
        
        removed_count = len(data) - len(cleaned_data)
        print(f"Removed {removed_count} outliers")
        
        return cleaned_data
    
    def generate_data_quality_report(self, data: pd.DataFrame) -> dict:
        """
        Generate a comprehensive data quality report.
        
        Args:
            data: DataFrame to analyze
        
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_records': len(data),
            'date_range': {
                'start': data['Date'].min().strftime('%Y-%m-%d'),
                'end': data['Date'].max().strftime('%Y-%m-%d'),
                'days': (data['Date'].max() - data['Date'].min()).days
            },
            'sectors': {
                'count': data['Sector'].nunique(),
                'names': data['Sector'].unique().tolist(),
                'records_per_sector': data['Sector'].value_counts().to_dict()
            },
            'missing_values': data.isnull().sum().to_dict(),
            'price_statistics': {
                'min': data['Close'].min(),
                'max': data['Close'].max(),
                'mean': data['Close'].mean(),
                'median': data['Close'].median(),
                'std': data['Close'].std()
            },
            'duplicates': data.duplicated().sum()
        }
        
        self.data_quality_report = report
        return report
    
    def print_quality_report(self, report: dict = None) -> None:
        """Print a formatted data quality report."""
        if report is None:
            report = self.data_quality_report
        
        print("\n" + "="*50)
        print("DATA QUALITY REPORT")
        print("="*50)
        print(f"Total Records: {report['total_records']:,}")
        print(f"Date Range: {report['date_range']['start']} to {report['date_range']['end']} ({report['date_range']['days']} days)")
        print(f"Sectors: {report['sectors']['count']} ({', '.join(report['sectors']['names'])})")
        print(f"Records per Sector: {report['sectors']['records_per_sector']}")
        print(f"Missing Values: {report['missing_values']}")
        print(f"Duplicates: {report['duplicates']}")
        print(f"Price Range: ${report['price_statistics']['min']:.2f} - ${report['price_statistics']['max']:.2f}")
        print(f"Average Price: ${report['price_statistics']['mean']:.2f}")
        print("="*50)
    
    def preprocess_pipeline(self, data: Optional[pd.DataFrame] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          sectors: Optional[List[str]] = None,
                          remove_outliers: bool = False,
                          handle_missing_dates: bool = True,
                          use_mongodb: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Raw DataFrame (if None, will fetch from MongoDB)
            start_date: Optional start date filter
            end_date: Optional end date filter
            sectors: Optional list of sectors to include
            remove_outliers: Whether to remove outliers
            handle_missing_dates: Whether to handle missing dates
            use_mongodb: Whether to use MongoDB for data source and storage
        
        Returns:
            Preprocessed DataFrame
        """
        print("Starting data preprocessing pipeline...")
        
        # Get data from MongoDB if not provided
        if data is None and use_mongodb:
            start_dt = pd.to_datetime(start_date) if start_date else None
            end_dt = pd.to_datetime(end_date) if end_date else None
            data = self.get_data_from_mongodb(start_dt, end_dt, sectors)
        
        if data is None or data.empty:
            print("No data available for preprocessing")
            return pd.DataFrame()
        
        # Validate data
        is_valid, issues = self.validate_data(data)
        if not is_valid:
            print("Data validation issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Clean data
        processed_data = self.clean_data(data)
        
        # Filter by date range
        if start_date or end_date:
            processed_data = self.filter_date_range(processed_data, start_date, end_date)
        
        # Filter by sectors
        if sectors:
            processed_data = self.filter_sectors(processed_data, sectors)
        
        # Handle missing dates
        if handle_missing_dates:
            processed_data = self.handle_missing_dates(processed_data)
        
        # Remove outliers
        if remove_outliers:
            processed_data = self.remove_outliers(processed_data)
        
        # Generate quality report
        quality_report = self.generate_data_quality_report(processed_data)
        self.print_quality_report(quality_report)
        
        # Store processed data back to MongoDB if using MongoDB
        if use_mongodb and not processed_data.empty:
            self._store_processed_data_to_mongodb(processed_data)
        
        # Log data quality
        self.mongo_manager.log_data_quality(
            process_name="data_preprocessing",
            status="success" if is_valid else "warning",
            records_processed=len(processed_data),
            records_skipped=len(data) - len(processed_data),
            warning_count=len(issues) if not is_valid else 0,
            error_details={"validation_issues": issues} if not is_valid else {}
        )
        
        print("Data preprocessing completed!")
        return processed_data
    
    def _store_processed_data_to_mongodb(self, data: pd.DataFrame) -> None:
        """
        Store processed data back to MongoDB as daily returns.
        
        Args:
            data: Processed DataFrame with Date, Sector, Close columns
        """
        try:
            # Calculate daily returns
            data_sorted = data.sort_values(['Sector', 'Date']).copy()
            data_sorted['DailyReturn'] = data_sorted.groupby('Sector')['Close'].pct_change()
            
            # Prepare returns data for MongoDB
            returns_data = []
            for _, row in data_sorted.iterrows():
                if pd.notna(row['DailyReturn']):
                    return_doc = {
                        "sector_code": row['Sector'],
                        "trade_date": row['Date'].to_pydatetime() if hasattr(row['Date'], 'to_pydatetime') else row['Date'],
                        "daily_return": float(row['DailyReturn']),
                        "cumulative_return": 0.0,  # Will be calculated separately
                        "return_type": "sector"
                    }
                    returns_data.append(return_doc)
            
            # Insert daily returns
            if returns_data:
                self.mongo_manager.insert_daily_returns(returns_data)
                print(f"Stored {len(returns_data)} daily return records to MongoDB")
                
        except Exception as e:
            print(f"Error storing processed data to MongoDB: {e}")


def main():
    """Example usage of the DataPreprocessor class."""
    try:
        from .mongodb_utils import MongoDBContext
        
        with MongoDBContext() as mongo_manager:
            preprocessor = DataPreprocessor(mongo_manager)
            
            # Run preprocessing pipeline using MongoDB
            cleaned_data = preprocessor.preprocess_pipeline(
                start_date='2024-01-15',
                remove_outliers=True,
                use_mongodb=True
            )
            
            print("\nCleaned data from MongoDB:")
            print(cleaned_data.head())
            print(f"Data shape: {cleaned_data.shape}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
