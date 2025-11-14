#!/usr/bin/env python3
"""
MongoDB Atlas Setup Script
==========================

This script sets up the complete database schema in MongoDB Atlas and migrates existing data.

Usage:
    python scripts/setup_mongodb_atlas.py

Requirements:
    - MONGODB_URI environment variable must be set
    - Existing CSV files in data/raw/ directory (optional)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .mongodb_utils import MongoDBContext, MongoDBManager


class MongoDBAtlasSetup:
    """Handles complete MongoDB Atlas setup and data migration."""
    
    def __init__(self, mongo_manager: MongoDBManager):
        self.mongo_manager = mongo_manager
        self.setup_log = []
    
    def setup_database_schema(self) -> bool:
        """Set up the complete database schema with collections and indexes."""
        print("Setting up MongoDB Atlas database schema...")
        print("=" * 50)
        
        try:
            # 1. Create sectors collection and seed data
            print("1. Setting up sectors collection...")
            self._setup_sectors()
            
            # 2. Create stocks collection and seed data
            print("2. Setting up stocks collection...")
            self._setup_stocks()
            
            # 3. Create price collections
            print("3. Setting up price collections...")
            self._setup_price_collections()
            
            # 4. Create analytics collections
            print("4. Setting up analytics collections...")
            self._setup_analytics_collections()
            
            # 5. Create utility collections
            print("5. Setting up utility collections...")
            self._setup_utility_collections()
            
            # 6. Create indexes for performance
            print("6. Creating performance indexes...")
            self._create_performance_indexes()
            
            print("✅ Database schema setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up database schema: {e}")
            return False
    
    def _setup_sectors(self) -> None:
        """Set up sectors collection with master data."""
        sectors_data = [
            {
                "code": "IT",
                "name": "Information Technology",
                "description": "Technology companies including software, hardware, and IT services",
                "etf_symbol": "XLK",
                "market_cap_weight": 0.30,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "code": "Pharma",
                "name": "Pharmaceuticals & Healthcare",
                "description": "Pharmaceutical companies, biotech, and healthcare services",
                "etf_symbol": "XLV",
                "market_cap_weight": 0.25,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "code": "Banking",
                "name": "Financials & Banking",
                "description": "Banks, financial services, and insurance companies",
                "etf_symbol": "XLF",
                "market_cap_weight": 0.25,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "code": "FMCG",
                "name": "Consumer Staples (FMCG)",
                "description": "Fast-moving consumer goods and essential products",
                "etf_symbol": "XLP",
                "market_cap_weight": 0.20,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]
        
        for sector in sectors_data:
            success = self.mongo_manager.upsert_sector(sector)
            if success:
                print(f"  ✅ Sector: {sector['code']} - {sector['name']}")
            else:
                print(f"  ❌ Failed to create sector: {sector['code']}")
    
    def _setup_stocks(self) -> None:
        """Set up stocks collection with sample data."""
        stocks_data = [
            # IT Sector
            {"sector_code": "IT", "symbol": "AAPL", "company_name": "Apple Inc.", "market_cap": 3000000000000, "industry": "Consumer Electronics", "is_active": True},
            {"sector_code": "IT", "symbol": "MSFT", "company_name": "Microsoft Corporation", "market_cap": 2800000000000, "industry": "Software", "is_active": True},
            {"sector_code": "IT", "symbol": "NVDA", "company_name": "NVIDIA Corporation", "market_cap": 1200000000000, "industry": "Semiconductors", "is_active": True},
            {"sector_code": "IT", "symbol": "GOOGL", "company_name": "Alphabet Inc.", "market_cap": 1800000000000, "industry": "Internet Services", "is_active": True},
            {"sector_code": "IT", "symbol": "META", "company_name": "Meta Platforms Inc.", "market_cap": 800000000000, "industry": "Social Media", "is_active": True},
            
            # Pharma Sector
            {"sector_code": "Pharma", "symbol": "PFE", "company_name": "Pfizer Inc.", "market_cap": 200000000000, "industry": "Pharmaceuticals", "is_active": True},
            {"sector_code": "Pharma", "symbol": "JNJ", "company_name": "Johnson & Johnson", "market_cap": 450000000000, "industry": "Healthcare", "is_active": True},
            {"sector_code": "Pharma", "symbol": "MRK", "company_name": "Merck & Co., Inc.", "market_cap": 300000000000, "industry": "Pharmaceuticals", "is_active": True},
            {"sector_code": "Pharma", "symbol": "ABBV", "company_name": "AbbVie Inc.", "market_cap": 250000000000, "industry": "Biotechnology", "is_active": True},
            {"sector_code": "Pharma", "symbol": "UNH", "company_name": "UnitedHealth Group Inc.", "market_cap": 500000000000, "industry": "Health Insurance", "is_active": True},
            
            # Banking Sector
            {"sector_code": "Banking", "symbol": "JPM", "company_name": "JPMorgan Chase & Co.", "market_cap": 400000000000, "industry": "Banking", "is_active": True},
            {"sector_code": "Banking", "symbol": "BAC", "company_name": "Bank of America Corporation", "market_cap": 250000000000, "industry": "Banking", "is_active": True},
            {"sector_code": "Banking", "symbol": "WFC", "company_name": "Wells Fargo & Company", "market_cap": 150000000000, "industry": "Banking", "is_active": True},
            {"sector_code": "Banking", "symbol": "GS", "company_name": "Goldman Sachs Group Inc.", "market_cap": 120000000000, "industry": "Investment Banking", "is_active": True},
            {"sector_code": "Banking", "symbol": "C", "company_name": "Citigroup Inc.", "market_cap": 100000000000, "industry": "Banking", "is_active": True},
            
            # FMCG Sector
            {"sector_code": "FMCG", "symbol": "PG", "company_name": "Procter & Gamble Company", "market_cap": 350000000000, "industry": "Consumer Goods", "is_active": True},
            {"sector_code": "FMCG", "symbol": "KO", "company_name": "The Coca-Cola Company", "market_cap": 250000000000, "industry": "Beverages", "is_active": True},
            {"sector_code": "FMCG", "symbol": "PEP", "company_name": "PepsiCo, Inc.", "market_cap": 220000000000, "industry": "Food & Beverages", "is_active": True},
            {"sector_code": "FMCG", "symbol": "WMT", "company_name": "Walmart Inc.", "market_cap": 400000000000, "industry": "Retail", "is_active": True},
            {"sector_code": "FMCG", "symbol": "CL", "company_name": "Colgate-Palmolive Company", "market_cap": 60000000000, "industry": "Personal Care", "is_active": True}
        ]
        
        for stock in stocks_data:
            stock["created_at"] = datetime.utcnow()
            stock["updated_at"] = datetime.utcnow()
            success = self.mongo_manager.upsert_stock(stock)
            if success:
                print(f"  ✅ Stock: {stock['symbol']} - {stock['company_name']}")
            else:
                print(f"  ❌ Failed to create stock: {stock['symbol']}")
    
    def _setup_price_collections(self) -> None:
        """Set up price-related collections."""
        # These collections will be populated by data collection and migration
        print("  ✅ Price collections ready for data population")
    
    def _setup_analytics_collections(self) -> None:
        """Set up analytics collections."""
        # These collections will be populated by calculations
        print("  ✅ Analytics collections ready for calculation results")
    
    def _setup_utility_collections(self) -> None:
        """Set up utility collections."""
        # Create sample user preferences
        sample_preferences = [
            {
                "user_session_id": "default_user",
                "preference_type": "default_sectors",
                "preference_key": "selected_sectors",
                "preference_value": ["IT", "Pharma", "Banking", "FMCG"],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "user_session_id": "default_user",
                "preference_type": "chart_settings",
                "preference_key": "default_chart_type",
                "preference_value": "line",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]
        
        for pref in sample_preferences:
            try:
                self.mongo_manager.db.user_preferences.insert_one(pref)
                print(f"  ✅ User preference: {pref['preference_type']}")
            except Exception as e:
                print(f"  ❌ Failed to create preference: {e}")
    
    def _create_performance_indexes(self) -> None:
        """Create performance indexes for optimal query speed."""
        try:
            # Sectors indexes
            self.mongo_manager.db.sectors.create_index("code", unique=True)
            self.mongo_manager.db.sectors.create_index("etf_symbol", unique=True)
            
            # Stocks indexes
            self.mongo_manager.db.stocks.create_index("symbol", unique=True)
            self.mongo_manager.db.stocks.create_index("sector_code")
            self.mongo_manager.db.stocks.create_index("is_active")
            
            # Price indexes
            self.mongo_manager.db.sector_etf_prices.create_index([("sector_code", 1), ("trade_date", 1)], unique=True)
            self.mongo_manager.db.sector_etf_prices.create_index("trade_date")
            self.mongo_manager.db.sector_etf_prices.create_index("sector_code")
            
            self.mongo_manager.db.stock_prices.create_index([("symbol", 1), ("trade_date", 1)], unique=True)
            self.mongo_manager.db.stock_prices.create_index("trade_date")
            self.mongo_manager.db.stock_prices.create_index("symbol")
            
            # Returns indexes
            self.mongo_manager.db.daily_returns.create_index([("sector_code", 1), ("trade_date", 1)], unique=True, sparse=True)
            self.mongo_manager.db.daily_returns.create_index([("symbol", 1), ("trade_date", 1)], unique=True, sparse=True)
            self.mongo_manager.db.daily_returns.create_index("trade_date")
            self.mongo_manager.db.daily_returns.create_index("return_type")
            
            # Metrics indexes
            self.mongo_manager.db.sector_metrics.create_index([("sector_code", 1), ("calculation_date", 1), ("period_type", 1)], unique=True)
            self.mongo_manager.db.sector_metrics.create_index("calculation_date")
            self.mongo_manager.db.sector_metrics.create_index("sector_code")
            
            # Portfolio analysis indexes
            self.mongo_manager.db.portfolio_analyses.create_index("start_date")
            self.mongo_manager.db.portfolio_analyses.create_index("analysis_type")
            self.mongo_manager.db.portfolio_analyses.create_index("created_at")
            
            # Correlation matrix indexes
            self.mongo_manager.db.correlation_matrix.create_index([("calculation_date", 1), ("sector_1", 1), ("sector_2", 1), ("period_days", 1)], unique=True)
            self.mongo_manager.db.correlation_matrix.create_index("calculation_date")
            
            # Data quality logs indexes
            self.mongo_manager.db.data_quality_logs.create_index("process_date")
            self.mongo_manager.db.data_quality_logs.create_index("status")
            self.mongo_manager.db.data_quality_logs.create_index("created_at")
            
            # User preferences indexes
            self.mongo_manager.db.user_preferences.create_index([("user_session_id", 1), ("preference_type", 1), ("preference_key", 1)], unique=True)
            self.mongo_manager.db.user_preferences.create_index("user_session_id")
            
            # Alerts indexes
            self.mongo_manager.db.alerts.create_index("created_at")
            self.mongo_manager.db.alerts.create_index("is_active")
            self.mongo_manager.db.alerts.create_index("alert_type")
            
            print("  ✅ All performance indexes created successfully")
            
        except Exception as e:
            print(f"  ⚠️  Some indexes may not have been created: {e}")
    
    def migrate_existing_data(self) -> bool:
        """Migrate existing CSV data to MongoDB Atlas."""
        print("\nMigrating existing CSV data to MongoDB Atlas...")
        print("=" * 50)
        
        try:
            # Import and run the migration tool
            from .migrate_csv_to_mongodb import CSVMigrationTool
            
            migration_tool = CSVMigrationTool(self.mongo_manager)
            success = migration_tool.run_migration()
            
            if success:
                print("✅ Data migration completed successfully!")
            else:
                print("⚠️  Data migration completed with some errors")
            
            return success
            
        except Exception as e:
            print(f"❌ Error during data migration: {e}")
            return False
    
    def download_fresh_data(self) -> bool:
        """Download fresh data from Yahoo Finance."""
        print("\nDownloading fresh data from Yahoo Finance...")
        print("=" * 50)
        
        try:
            from .data_collection import DataCollector
            
            collector = DataCollector(self.mongo_manager)
            success = collector.download_all_sectors()
            
            if success:
                print("✅ Fresh data downloaded and stored successfully!")
            else:
                print("⚠️  Some data download issues occurred")
            
            return success
            
        except Exception as e:
            print(f"❌ Error downloading fresh data: {e}")
            return False
    
    def run_complete_setup(self, migrate_csv: bool = True, download_fresh: bool = False) -> bool:
        """Run the complete MongoDB Atlas setup process."""
        print("MongoDB Atlas Complete Setup")
        print("=" * 50)
        print(f"Migrate CSV data: {migrate_csv}")
        print(f"Download fresh data: {download_fresh}")
        print()
        
        # Step 1: Setup database schema
        if not self.setup_database_schema():
            return False
        
        # Step 2: Migrate existing CSV data
        if migrate_csv:
            if not self.migrate_existing_data():
                print("⚠️  CSV migration had issues, but continuing...")
        
        # Step 3: Download fresh data
        if download_fresh:
            if not self.download_fresh_data():
                print("⚠️  Fresh data download had issues, but continuing...")
        
        # Step 4: Log setup completion
        self.mongo_manager.log_data_quality(
            process_name="mongodb_atlas_setup",
            status="success",
            records_processed=0,
            error_count=0,
            processing_time_seconds=0
        )
        
        print("\n🎉 MongoDB Atlas setup completed successfully!")
        print("Your database is ready for the stock market analytics application.")
        
        return True


def main():
    """Main function to run MongoDB Atlas setup."""
    print("MongoDB Atlas Setup")
    print("=" * 25)
    
    try:
        with MongoDBContext() as mongo_manager:
            setup_tool = MongoDBAtlasSetup(mongo_manager)
            
            print("Setting up MongoDB Atlas database...")
            print("This will create schema, migrate data, and download fresh data.")
            print()
            
            # Run complete setup
            success = setup_tool.run_complete_setup(
                migrate_csv=True,
                download_fresh=True
            )
            
            if success:
                print("\n✅ Setup completed successfully!")
                print("You can now run: python main.py")
            else:
                print("\n❌ Setup failed. Please check the error messages above.")
                
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("Make sure MongoDB is properly configured with MONGODB_URI environment variable")


if __name__ == "__main__":
    main()
