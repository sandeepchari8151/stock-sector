#!/usr/bin/env python3
"""
MongoDB Atlas Dashboard
======================

This script provides a dashboard view of your MongoDB Atlas database status,
collections, and data statistics.

Usage:
    python scripts/mongodb_dashboard.py
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .mongodb_utils import MongoDBContext, MongoDBManager


class MongoDBDashboard:
    """MongoDB Atlas dashboard for monitoring and statistics."""
    
    def __init__(self, mongo_manager: MongoDBManager):
        self.mongo_manager = mongo_manager
        self.stats = {}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            # Get database stats
            db_stats = self.mongo_manager.db.command("dbStats")
            
            # Get collection stats
            collections = self.mongo_manager.db.list_collection_names()
            collection_stats = {}
            
            for collection_name in collections:
                try:
                    collection = self.mongo_manager.db[collection_name]
                    count = collection.count_documents({})
                    collection_stats[collection_name] = {
                        "count": count,
                        "size_bytes": db_stats.get("collections", {}).get(collection_name, {}).get("size", 0)
                    }
                except Exception as e:
                    collection_stats[collection_name] = {
                        "count": 0,
                        "size_bytes": 0,
                        "error": str(e)
                    }
            
            return {
                "database_name": self.mongo_manager.db_name,
                "total_collections": len(collections),
                "total_documents": sum(stats["count"] for stats in collection_stats.values()),
                "database_size_bytes": db_stats.get("dataSize", 0),
                "storage_size_bytes": db_stats.get("storageSize", 0),
                "indexes": db_stats.get("indexes", 0),
                "collections": collection_stats
            }
            
        except Exception as e:
            return {"error": f"Failed to get database stats: {e}"}
    
    def get_sector_data_summary(self) -> Dict[str, Any]:
        """Get summary of sector data."""
        try:
            sectors = self.mongo_manager.get_sectors()
            sector_summary = {}
            
            for sector in sectors:
                sector_code = sector["code"]
                
                # Get price data count
                price_count = self.mongo_manager.db.sector_etf_prices.count_documents(
                    {"sector_code": sector_code}
                )
                
                # Get latest price
                latest_price = self.mongo_manager.db.sector_etf_prices.find_one(
                    {"sector_code": sector_code},
                    sort=[("trade_date", -1)]
                )
                
                # Get returns data count
                returns_count = self.mongo_manager.db.daily_returns.count_documents(
                    {"sector_code": sector_code}
                )
                
                # Get metrics count
                metrics_count = self.mongo_manager.db.sector_metrics.count_documents(
                    {"sector_code": sector_code}
                )
                
                sector_summary[sector_code] = {
                    "name": sector["name"],
                    "etf_symbol": sector["etf_symbol"],
                    "price_records": price_count,
                    "returns_records": returns_count,
                    "metrics_records": metrics_count,
                    "latest_price": latest_price["close_price"] if latest_price else None,
                    "latest_date": latest_price["trade_date"] if latest_price else None
                }
            
            return sector_summary
            
        except Exception as e:
            return {"error": f"Failed to get sector summary: {e}"}
    
    def get_stock_data_summary(self) -> Dict[str, Any]:
        """Get summary of stock data."""
        try:
            stocks = self.mongo_manager.get_stocks()
            stock_summary = {}
            
            for stock in stocks:
                symbol = stock["symbol"]
                
                # Get price data count
                price_count = self.mongo_manager.db.stock_prices.count_documents(
                    {"symbol": symbol}
                )
                
                # Get latest price
                latest_price = self.mongo_manager.db.stock_prices.find_one(
                    {"symbol": symbol},
                    sort=[("trade_date", -1)]
                )
                
                stock_summary[symbol] = {
                    "company_name": stock["company_name"],
                    "sector_code": stock["sector_code"],
                    "price_records": price_count,
                    "latest_price": latest_price["close_price"] if latest_price else None,
                    "latest_date": latest_price["trade_date"] if latest_price else None
                }
            
            return stock_summary
            
        except Exception as e:
            return {"error": f"Failed to get stock summary: {e}"}
    
    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get data quality and processing summary."""
        try:
            # Get recent data quality logs
            recent_logs = list(self.mongo_manager.db.data_quality_logs.find(
                {"created_at": {"$gte": datetime.utcnow() - timedelta(days=7)}},
                sort=[("created_at", -1)]
            ))
            
            # Get process statistics
            process_stats = {}
            for log in recent_logs:
                process_name = log["process_name"]
                if process_name not in process_stats:
                    process_stats[process_name] = {
                        "total_runs": 0,
                        "successful_runs": 0,
                        "failed_runs": 0,
                        "total_records_processed": 0,
                        "total_errors": 0,
                        "last_run": None
                    }
                
                process_stats[process_name]["total_runs"] += 1
                if log["status"] == "success":
                    process_stats[process_name]["successful_runs"] += 1
                else:
                    process_stats[process_name]["failed_runs"] += 1
                
                process_stats[process_name]["total_records_processed"] += log.get("records_processed", 0)
                process_stats[process_name]["total_errors"] += log.get("error_count", 0)
                process_stats[process_name]["last_run"] = log["created_at"]
            
            return {
                "recent_logs_count": len(recent_logs),
                "process_statistics": process_stats
            }
            
        except Exception as e:
            return {"error": f"Failed to get data quality summary: {e}"}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        try:
            # Get latest sector metrics
            latest_metrics = {}
            sectors = self.mongo_manager.get_sectors()
            
            for sector in sectors:
                sector_code = sector["code"]
                latest_metric = self.mongo_manager.db.sector_metrics.find_one(
                    {"sector_code": sector_code},
                    sort=[("calculation_date", -1)]
                )
                
                if latest_metric:
                    latest_metrics[sector_code] = {
                        "cagr": latest_metric.get("cagr", 0),
                        "volatility": latest_metric.get("volatility", 0),
                        "sharpe_ratio": latest_metric.get("sharpe_ratio", 0),
                        "max_drawdown": latest_metric.get("max_drawdown", 0),
                        "calculation_date": latest_metric.get("calculation_date")
                    }
            
            # Get portfolio analyses count
            portfolio_count = self.mongo_manager.db.portfolio_analyses.count_documents({})
            
            # Get correlation matrix count
            correlation_count = self.mongo_manager.db.correlation_matrix.count_documents({})
            
            return {
                "sector_metrics": latest_metrics,
                "portfolio_analyses_count": portfolio_count,
                "correlation_entries_count": correlation_count
            }
            
        except Exception as e:
            return {"error": f"Failed to get performance metrics: {e}"}
    
    def print_dashboard(self) -> None:
        """Print the complete dashboard."""
        print("📊 MongoDB Atlas Dashboard")
        print("=" * 50)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Database stats
        print("🗄️  DATABASE OVERVIEW")
        print("-" * 30)
        db_stats = self.get_database_stats()
        if "error" in db_stats:
            print(f"❌ {db_stats['error']}")
        else:
            print(f"Database: {db_stats['database_name']}")
            print(f"Collections: {db_stats['total_collections']}")
            print(f"Total Documents: {db_stats['total_documents']:,}")
            print(f"Database Size: {db_stats['database_size_bytes'] / 1024 / 1024:.2f} MB")
            print(f"Storage Size: {db_stats['storage_size_bytes'] / 1024 / 1024:.2f} MB")
            print(f"Indexes: {db_stats['indexes']}")
        print()
        
        # Collection details
        print("📁 COLLECTION DETAILS")
        print("-" * 30)
        if "collections" in db_stats:
            for collection_name, stats in db_stats["collections"].items():
                if "error" in stats:
                    print(f"  {collection_name}: ❌ {stats['error']}")
                else:
                    print(f"  {collection_name}: {stats['count']:,} documents")
        print()
        
        # Sector data summary
        print("🏢 SECTOR DATA SUMMARY")
        print("-" * 30)
        sector_summary = self.get_sector_data_summary()
        if "error" in sector_summary:
            print(f"❌ {sector_summary['error']}")
        else:
            for sector_code, data in sector_summary.items():
                print(f"  {sector_code} ({data['name']}):")
                print(f"    ETF Symbol: {data['etf_symbol']}")
                print(f"    Price Records: {data['price_records']:,}")
                print(f"    Returns Records: {data['returns_records']:,}")
                print(f"    Metrics Records: {data['metrics_records']:,}")
                if data['latest_price']:
                    print(f"    Latest Price: ${data['latest_price']:.2f} ({data['latest_date']})")
                print()
        
        # Stock data summary
        print("📈 STOCK DATA SUMMARY")
        print("-" * 30)
        stock_summary = self.get_stock_data_summary()
        if "error" in stock_summary:
            print(f"❌ {stock_summary['error']}")
        else:
            print(f"Total Stocks: {len(stock_summary)}")
            for symbol, data in list(stock_summary.items())[:5]:  # Show first 5
                print(f"  {symbol} ({data['company_name']}):")
                print(f"    Sector: {data['sector_code']}")
                print(f"    Price Records: {data['price_records']:,}")
                if data['latest_price']:
                    print(f"    Latest Price: ${data['latest_price']:.2f} ({data['latest_date']})")
            if len(stock_summary) > 5:
                print(f"  ... and {len(stock_summary) - 5} more stocks")
        print()
        
        # Data quality summary
        print("🔍 DATA QUALITY SUMMARY")
        print("-" * 30)
        quality_summary = self.get_data_quality_summary()
        if "error" in quality_summary:
            print(f"❌ {quality_summary['error']}")
        else:
            print(f"Recent Logs (7 days): {quality_summary['recent_logs_count']}")
            for process_name, stats in quality_summary["process_statistics"].items():
                success_rate = (stats["successful_runs"] / stats["total_runs"] * 100) if stats["total_runs"] > 0 else 0
                print(f"  {process_name}:")
                print(f"    Runs: {stats['total_runs']} (Success: {success_rate:.1f}%)")
                print(f"    Records Processed: {stats['total_records_processed']:,}")
                print(f"    Errors: {stats['total_errors']}")
                if stats["last_run"]:
                    print(f"    Last Run: {stats['last_run']}")
        print()
        
        # Performance metrics
        print("📊 PERFORMANCE METRICS")
        print("-" * 30)
        perf_metrics = self.get_performance_metrics()
        if "error" in perf_metrics:
            print(f"❌ {perf_metrics['error']}")
        else:
            print(f"Portfolio Analyses: {perf_metrics['portfolio_analyses_count']}")
            print(f"Correlation Entries: {perf_metrics['correlation_entries_count']}")
            print()
            print("Latest Sector Metrics:")
            for sector_code, metrics in perf_metrics["sector_metrics"].items():
                print(f"  {sector_code}:")
                print(f"    CAGR: {metrics['cagr']:.2%}")
                print(f"    Volatility: {metrics['volatility']:.2%}")
                print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")
        print()
        
        print("✅ Dashboard complete!")


def main():
    """Main function to run the dashboard."""
    try:
        with MongoDBContext() as mongo_manager:
            dashboard = MongoDBDashboard(mongo_manager)
            dashboard.print_dashboard()
            
    except Exception as e:
        print(f"❌ Dashboard failed: {e}")
        print("Make sure MongoDB is properly configured with MONGODB_URI environment variable")


if __name__ == "__main__":
    main()
