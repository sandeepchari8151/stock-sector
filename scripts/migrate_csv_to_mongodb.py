"""
CSV to MongoDB Migration Script
==============================

This script migrates existing CSV data to MongoDB collections.
It reads data from the existing CSV files and stores them in the appropriate MongoDB collections.

Usage:
    python scripts/migrate_csv_to_mongodb.py

Requirements:
    - MONGODB_URI environment variable must be set
    - Existing CSV files in data/raw/ directory
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .mongodb_utils import MongoDBContext, MongoDBManager


class CSVMigrationTool:
    """Handles migration of CSV data to MongoDB."""
    
    def __init__(self, mongo_manager: MongoDBManager):
        self.mongo_manager = mongo_manager
        self.migrated_count = 0
        self.errors = []
    
    def migrate_sectors(self) -> bool:
        """Migrate sector master data."""
        print("Migrating sectors...")
        
        try:
            # Define sectors based on existing CSV files
            sectors_data = [
                {
                    "code": "IT",
                    "name": "Information Technology",
                    "etf_symbol": "XLK",
                    "is_active": True
                },
                {
                    "code": "Pharma", 
                    "name": "Pharmaceuticals & Healthcare",
                    "etf_symbol": "XLV",
                    "is_active": True
                },
                {
                    "code": "Banking",
                    "name": "Financials & Banking", 
                    "etf_symbol": "XLF",
                    "is_active": True
                },
                {
                    "code": "FMCG",
                    "name": "Consumer Staples (FMCG)",
                    "etf_symbol": "XLP", 
                    "is_active": True
                }
            ]
            
            for sector in sectors_data:
                success = self.mongo_manager.upsert_sector(sector)
                if success:
                    self.migrated_count += 1
                    print(f"  ✅ Migrated sector: {sector['code']}")
                else:
                    self.errors.append(f"Failed to migrate sector: {sector['code']}")
            
            return len(self.errors) == 0
            
        except Exception as e:
            self.errors.append(f"Error migrating sectors: {e}")
            return False
    
    def migrate_sector_prices(self) -> bool:
        """Migrate sector price data from CSV files."""
        print("Migrating sector price data...")
        
        csv_files = {
            "IT": "data/raw/IT.csv",
            "Pharma": "data/raw/Pharma.csv", 
            "Banking": "data/raw/Banking.csv",
            "FMCG": "data/raw/FMCG.csv"
        }
        
        total_migrated = 0
        
        for sector_code, csv_path in csv_files.items():
            try:
                if not os.path.exists(csv_path):
                    print(f"  ⚠️  CSV file not found: {csv_path}")
                    continue
                
                # Read CSV file
                df = pd.read_csv(csv_path, parse_dates=['Date'])
                
                if df.empty:
                    print(f"  ⚠️  Empty CSV file: {csv_path}")
                    continue
                
                # Convert to MongoDB format
                prices_data = []
                for _, row in df.iterrows():
                    price_doc = {
                        "sector_code": sector_code,
                        "trade_date": row['Date'].to_pydatetime() if hasattr(row['Date'], 'to_pydatetime') else row['Date'],
                        "close_price": float(row['Close']) if pd.notna(row['Close']) else None,
                        "adjusted_close": float(row['Close']) if pd.notna(row['Close']) else None,
                        "volume": None  # Not available in current CSV format
                    }
                    prices_data.append(price_doc)
                
                # Insert into MongoDB
                inserted_count = self.mongo_manager.insert_sector_etf_prices(prices_data)
                total_migrated += inserted_count
                print(f"  ✅ Migrated {inserted_count} price records for {sector_code}")
                
            except Exception as e:
                error_msg = f"Error migrating {sector_code}: {e}"
                self.errors.append(error_msg)
                print(f"  ❌ {error_msg}")
        
        self.migrated_count += total_migrated
        return len(self.errors) == 0
    
    def migrate_sample_data(self) -> bool:
        """Migrate sample data if available."""
        print("Checking for sample data...")
        
        sample_file = "data/raw/sample_sector_prices.csv"
        if not os.path.exists(sample_file):
            print("  ℹ️  No sample data file found")
            return True
        
        try:
            # Read sample data
            df = pd.read_csv(sample_file, parse_dates=['date'])
            
            if df.empty:
                print("  ⚠️  Empty sample data file")
                return True
            
            # Group by symbol and migrate as sector data
            symbol_to_sector = {
                "XLK": "IT",
                "XLV": "Pharma", 
                "XLF": "Banking",
                "XLP": "FMCG"
            }
            
            total_migrated = 0
            
            for symbol, sector_code in symbol_to_sector.items():
                symbol_data = df[df['symbol'].str.upper() == symbol.upper()]
                
                if symbol_data.empty:
                    continue
                
                # Convert to MongoDB format
                prices_data = []
                for _, row in symbol_data.iterrows():
                    price_doc = {
                        "sector_code": sector_code,
                        "trade_date": row['date'].to_pydatetime() if hasattr(row['date'], 'to_pydatetime') else row['date'],
                        "open_price": float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                        "high_price": float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                        "low_price": float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                        "close_price": float(row.get('close', 0)) if pd.notna(row.get('close')) else None,
                        "adjusted_close": float(row.get('close', 0)) if pd.notna(row.get('close')) else None,
                        "volume": int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None
                    }
                    prices_data.append(price_doc)
                
                # Insert into MongoDB
                inserted_count = self.mongo_manager.insert_sector_etf_prices(prices_data)
                total_migrated += inserted_count
                print(f"  ✅ Migrated {inserted_count} sample records for {sector_code}")
            
            self.migrated_count += total_migrated
            return True
            
        except Exception as e:
            error_msg = f"Error migrating sample data: {e}"
            self.errors.append(error_msg)
            print(f"  ❌ {error_msg}")
            return False
    
    def migrate_processed_data(self) -> bool:
        """Migrate processed data files if available."""
        print("Checking for processed data...")
        
        processed_dir = "data/processed"
        if not os.path.exists(processed_dir):
            print("  ℹ️  No processed data directory found")
            return True
        
        try:
            # Find the most recent processed data file
            csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv') and f.startswith('processed_data_')]
            
            if not csv_files:
                print("  ℹ️  No processed data files found")
                return True
            
            # Sort by filename (which includes timestamp)
            csv_files.sort(reverse=True)
            latest_file = csv_files[0]
            file_path = os.path.join(processed_dir, latest_file)
            
            print(f"  📁 Migrating latest processed data: {latest_file}")
            
            # Read processed data
            df = pd.read_csv(file_path, parse_dates=['Date'])
            
            if df.empty:
                print("  ⚠️  Empty processed data file")
                return True
            
            # Convert to MongoDB format
            prices_data = []
            for _, row in df.iterrows():
                price_doc = {
                    "sector_code": row['Sector'],
                    "trade_date": row['Date'].to_pydatetime() if hasattr(row['Date'], 'to_pydatetime') else row['Date'],
                    "close_price": float(row['Close']) if pd.notna(row['Close']) else None,
                    "adjusted_close": float(row['Close']) if pd.notna(row['Close']) else None,
                    "volume": None
                }
                prices_data.append(price_doc)
            
            # Insert into MongoDB
            inserted_count = self.mongo_manager.insert_sector_etf_prices(prices_data)
            self.migrated_count += inserted_count
            print(f"  ✅ Migrated {inserted_count} processed records")
            
            return True
            
        except Exception as e:
            error_msg = f"Error migrating processed data: {e}"
            self.errors.append(error_msg)
            print(f"  ❌ {error_msg}")
            return False
    
    def run_migration(self) -> bool:
        """Run the complete migration process."""
        print("Starting CSV to MongoDB migration...")
        print("=" * 50)
        
        success = True
        
        # Migrate sectors first
        if not self.migrate_sectors():
            success = False
        
        # Migrate sector prices
        if not self.migrate_sector_prices():
            success = False
        
        # Migrate sample data
        if not self.migrate_sample_data():
            success = False
        
        # Migrate processed data
        if not self.migrate_processed_data():
            success = False
        
        # Print summary
        print("\n" + "=" * 50)
        print("MIGRATION SUMMARY")
        print("=" * 50)
        print(f"Total records migrated: {self.migrated_count}")
        print(f"Errors encountered: {len(self.errors)}")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        
        if success:
            print("\n✅ Migration completed successfully!")
        else:
            print("\n⚠️  Migration completed with errors")
        
        return success


def main():
    """Main function to run the migration."""
    print("CSV to MongoDB Migration Tool")
    print("=============================")
    
    try:
        with MongoDBContext() as mongo_manager:
            migration_tool = CSVMigrationTool(mongo_manager)
            success = migration_tool.run_migration()
            
            if success:
                print("\n🎉 All data has been successfully migrated to MongoDB!")
                print("You can now run the main application with MongoDB support.")
            else:
                print("\n❌ Migration completed with errors. Please check the error messages above.")
                
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("Make sure MongoDB is properly configured with MONGODB_URI environment variable")


if __name__ == "__main__":
    main()
