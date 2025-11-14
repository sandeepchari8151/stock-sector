"""
MongoDB Database Utilities
=========================

This module provides MongoDB connection and utility functions for the stock analytics project.
Handles all database operations including connection, data insertion, queries, and indexing.

Dependencies: pymongo, pandas, numpy
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import logging

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBManager:
    """Manages MongoDB connections and operations for the stock analytics project."""
    
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        """
        Initialize MongoDB connection.
        
        Args:
            uri: MongoDB connection string (defaults to MONGODB_URI env var)
            db_name: Database name (defaults to MONGODB_DB env var or 'sectorscope')
        """
        self.uri = uri or os.environ.get("MONGODB_URI", "").strip()
        self.db_name = db_name or os.environ.get("MONGODB_DB", "sectorscope").strip()
        
        if not self.uri:
            raise ValueError("MongoDB URI is required. Set MONGODB_URI environment variable.")
        
        self.client: Optional[MongoClient] = None
        self.db = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB: {self.db_name}")
            self._create_indexes()
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self) -> None:
        """Create necessary indexes for optimal performance."""
        try:
            # Sectors collection indexes
            self.db.sectors.create_index("code", unique=True)
            
            # Stocks collection indexes
            self.db.stocks.create_index("symbol", unique=True)
            self.db.stocks.create_index("sector_code")
            
            # Stock prices collection indexes
            self.db.stock_prices.create_index([("symbol", ASCENDING), ("trade_date", ASCENDING)], unique=True)
            self.db.stock_prices.create_index("trade_date")
            self.db.stock_prices.create_index("symbol")
            
            # Sector ETF prices collection indexes
            self.db.sector_etf_prices.create_index([("sector_code", ASCENDING), ("trade_date", ASCENDING)], unique=True)
            self.db.sector_etf_prices.create_index("trade_date")
            self.db.sector_etf_prices.create_index("sector_code")
            
            # Daily returns collection indexes
            self.db.daily_returns.create_index([("symbol", ASCENDING), ("trade_date", ASCENDING)], unique=True, sparse=True)
            self.db.daily_returns.create_index([("sector_code", ASCENDING), ("trade_date", ASCENDING)], unique=True, sparse=True)
            self.db.daily_returns.create_index("trade_date")
            self.db.daily_returns.create_index("return_type")
            
            # Sector metrics collection indexes
            self.db.sector_metrics.create_index([("sector_code", ASCENDING), ("calculation_date", ASCENDING), ("period_type", ASCENDING)], unique=True)
            self.db.sector_metrics.create_index("calculation_date")
            self.db.sector_metrics.create_index("sector_code")
            
            # Portfolio analyses collection indexes
            self.db.portfolio_analyses.create_index("start_date")
            self.db.portfolio_analyses.create_index("analysis_type")
            
            # Correlation matrix collection indexes
            self.db.correlation_matrix.create_index([("calculation_date", ASCENDING), ("sector_1", ASCENDING), ("sector_2", ASCENDING), ("period_days", ASCENDING)], unique=True)
            self.db.correlation_matrix.create_index("calculation_date")
            
            # Data quality logs collection indexes
            self.db.data_quality_logs.create_index("process_date")
            self.db.data_quality_logs.create_index("status")
            
            # User preferences collection indexes
            self.db.user_preferences.create_index([("user_session_id", ASCENDING), ("preference_type", ASCENDING), ("preference_key", ASCENDING)], unique=True)
            self.db.user_preferences.create_index("user_session_id")
            
            # Alerts collection indexes
            self.db.alerts.create_index("created_at")
            self.db.alerts.create_index("is_active")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Some indexes may not have been created: {e}")
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def upsert_sector(self, sector_data: Dict[str, Any]) -> bool:
        """
        Insert or update sector data.
        
        Args:
            sector_data: Dictionary containing sector information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            sector_data["updated_at"] = datetime.utcnow()
            if "created_at" not in sector_data:
                sector_data["created_at"] = datetime.utcnow()
            
            result = self.db.sectors.update_one(
                {"code": sector_data["code"]},
                {"$set": sector_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error upserting sector {sector_data.get('code', 'unknown')}: {e}")
            return False
    
    def upsert_stock(self, stock_data: Dict[str, Any]) -> bool:
        """
        Insert or update stock data.
        
        Args:
            stock_data: Dictionary containing stock information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            stock_data["updated_at"] = datetime.utcnow()
            if "created_at" not in stock_data:
                stock_data["created_at"] = datetime.utcnow()
            
            result = self.db.stocks.update_one(
                {"symbol": stock_data["symbol"]},
                {"$set": stock_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error upserting stock {stock_data.get('symbol', 'unknown')}: {e}")
            return False
    
    def insert_stock_prices(self, prices_data: List[Dict[str, Any]]) -> int:
        """
        Insert stock price data.
        
        Args:
            prices_data: List of dictionaries containing price information
            
        Returns:
            Number of documents inserted
        """
        try:
            # Add created_at timestamp
            for price in prices_data:
                price["created_at"] = datetime.utcnow()
            
            result = self.db.stock_prices.insert_many(prices_data, ordered=False)
            logger.info(f"Inserted {len(result.inserted_ids)} stock price records")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Error inserting stock prices: {e}")
            return 0
    
    def insert_sector_etf_prices(self, prices_data: List[Dict[str, Any]]) -> int:
        """
        Insert sector ETF price data.
        
        Args:
            prices_data: List of dictionaries containing ETF price information
            
        Returns:
            Number of documents inserted
        """
        try:
            # Add created_at timestamp
            for price in prices_data:
                price["created_at"] = datetime.utcnow()
            
            result = self.db.sector_etf_prices.insert_many(prices_data, ordered=False)
            logger.info(f"Inserted {len(result.inserted_ids)} sector ETF price records")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Error inserting sector ETF prices: {e}")
            return 0
    
    def insert_daily_returns(self, returns_data: List[Dict[str, Any]]) -> int:
        """
        Insert daily returns data.
        
        Args:
            returns_data: List of dictionaries containing return information
            
        Returns:
            Number of documents inserted
        """
        try:
            # Add created_at timestamp
            for return_data in returns_data:
                return_data["created_at"] = datetime.utcnow()
            
            result = self.db.daily_returns.insert_many(returns_data, ordered=False)
            logger.info(f"Inserted {len(result.inserted_ids)} daily return records")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Error inserting daily returns: {e}")
            return 0
    
    def insert_sector_metrics(self, metrics_data: List[Dict[str, Any]]) -> int:
        """
        Insert sector metrics data.
        
        Args:
            metrics_data: List of dictionaries containing metrics information
            
        Returns:
            Number of documents inserted
        """
        try:
            # Add created_at timestamp
            for metric in metrics_data:
                metric["created_at"] = datetime.utcnow()
            
            result = self.db.sector_metrics.insert_many(metrics_data, ordered=False)
            logger.info(f"Inserted {len(result.inserted_ids)} sector metric records")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Error inserting sector metrics: {e}")
            return 0
    
    def get_sectors(self) -> List[Dict[str, Any]]:
        """Get all sectors."""
        try:
            return list(self.db.sectors.find({}, {"_id": 0}))
        except Exception as e:
            logger.error(f"Error getting sectors: {e}")
            return []
    
    def get_stocks(self, sector_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get stocks, optionally filtered by sector.
        
        Args:
            sector_code: Optional sector code to filter by
            
        Returns:
            List of stock dictionaries
        """
        try:
            query = {"is_active": True}
            if sector_code:
                query["sector_code"] = sector_code
            
            return list(self.db.stocks.find(query, {"_id": 0}))
        except Exception as e:
            logger.error(f"Error getting stocks: {e}")
            return []
    
    def get_stock_prices(self, symbol: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get stock prices for a specific symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with stock prices
        """
        try:
            query = {"symbol": symbol}
            if start_date:
                query["trade_date"] = {"$gte": start_date}
            if end_date:
                if "trade_date" in query:
                    query["trade_date"]["$lte"] = end_date
                else:
                    query["trade_date"] = {"$lte": end_date}
            
            cursor = self.db.stock_prices.find(query).sort("trade_date", ASCENDING)
            data = list(cursor)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock prices for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_sector_etf_prices(self, sector_code: str, start_date: Optional[datetime] = None, 
                             end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get sector ETF prices for a specific sector.
        
        Args:
            sector_code: Sector code
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with sector ETF prices
        """
        try:
            query = {"sector_code": sector_code}
            if start_date:
                query["trade_date"] = {"$gte": start_date}
            if end_date:
                if "trade_date" in query:
                    query["trade_date"]["$lte"] = end_date
                else:
                    query["trade_date"] = {"$lte": end_date}
            
            cursor = self.db.sector_etf_prices.find(query).sort("trade_date", ASCENDING)
            data = list(cursor)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
            
        except Exception as e:
            logger.error(f"Error getting sector ETF prices for {sector_code}: {e}")
            return pd.DataFrame()
    
    def get_daily_returns(self, symbol: Optional[str] = None, sector_code: Optional[str] = None,
                         start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get daily returns data.
        
        Args:
            symbol: Optional stock symbol filter
            sector_code: Optional sector code filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with daily returns
        """
        try:
            query = {}
            if symbol:
                query["symbol"] = symbol
            if sector_code:
                query["sector_code"] = sector_code
            if start_date:
                query["trade_date"] = {"$gte": start_date}
            if end_date:
                if "trade_date" in query:
                    query["trade_date"]["$lte"] = end_date
                else:
                    query["trade_date"] = {"$lte": end_date}
            
            cursor = self.db.daily_returns.find(query).sort("trade_date", ASCENDING)
            data = list(cursor)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
            
        except Exception as e:
            logger.error(f"Error getting daily returns: {e}")
            return pd.DataFrame()
    
    def get_sector_metrics(self, sector_code: Optional[str] = None, 
                          calculation_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get sector metrics data.
        
        Args:
            sector_code: Optional sector code filter
            calculation_date: Optional calculation date filter
            
        Returns:
            DataFrame with sector metrics
        """
        try:
            query = {}
            if sector_code:
                query["sector_code"] = sector_code
            if calculation_date:
                query["calculation_date"] = calculation_date
            
            cursor = self.db.sector_metrics.find(query).sort("calculation_date", DESCENDING)
            data = list(cursor)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df["calculation_date"] = pd.to_datetime(df["calculation_date"])
            return df
            
        except Exception as e:
            logger.error(f"Error getting sector metrics: {e}")
            return pd.DataFrame()
    
    def log_data_quality(self, process_name: str, status: str, records_processed: int = 0,
                        records_skipped: int = 0, error_count: int = 0, warning_count: int = 0,
                        error_details: Optional[Dict] = None, processing_time_seconds: int = 0) -> bool:
        """
        Log data quality information.
        
        Args:
            process_name: Name of the process
            status: Process status (success, warning, error)
            records_processed: Number of records processed
            records_skipped: Number of records skipped
            error_count: Number of errors
            warning_count: Number of warnings
            error_details: Detailed error information
            processing_time_seconds: Processing time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            log_entry = {
                "process_name": process_name,
                "process_date": datetime.utcnow().date(),
                "status": status,
                "records_processed": records_processed,
                "records_skipped": records_skipped,
                "error_count": error_count,
                "warning_count": warning_count,
                "error_details": error_details or {},
                "processing_time_seconds": processing_time_seconds,
                "created_at": datetime.utcnow()
            }
            
            self.db.data_quality_logs.insert_one(log_entry)
            return True
        except Exception as e:
            logger.error(f"Error logging data quality: {e}")
            return False
    
    def get_latest_data_date(self, collection: str, date_field: str = "trade_date") -> Optional[datetime]:
        """
        Get the latest date from a collection.
        
        Args:
            collection: Collection name
            date_field: Date field name
            
        Returns:
            Latest date or None if not found
        """
        try:
            result = self.db[collection].find().sort(date_field, DESCENDING).limit(1)
            data = list(result)
            if data:
                return data[0][date_field]
            return None
        except Exception as e:
            logger.error(f"Error getting latest date from {collection}: {e}")
            return None


def get_mongodb_manager() -> MongoDBManager:
    """Get a MongoDB manager instance."""
    return MongoDBManager()


# Context manager for MongoDB operations
class MongoDBContext:
    """Context manager for MongoDB operations."""
    
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.manager = None
        self.uri = uri
        self.db_name = db_name
    
    def __enter__(self) -> MongoDBManager:
        self.manager = MongoDBManager(self.uri, self.db_name)
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.manager:
            self.manager.close()


if __name__ == "__main__":
    # Test MongoDB connection
    try:
        with MongoDBContext() as mongo:
            print("MongoDB connection successful!")
            print(f"Database: {mongo.db_name}")
            
            # Test basic operations
            sectors = mongo.get_sectors()
            print(f"Found {len(sectors)} sectors")
            
            stocks = mongo.get_stocks()
            print(f"Found {len(stocks)} stocks")
            
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
