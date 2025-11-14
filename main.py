"""
Main Application - Stock Market Sector Performance Analytics
===========================================================

This is the main orchestrator that coordinates all modules:
- Data Collection (Yahoo Finance) -> MongoDB
- Data Preprocessing (cleaning, filtering) -> MongoDB
- Financial Calculations (returns, volatility, CAGR) -> MongoDB
- Visualization (charts and dashboard)
- Insights & Reporting (analysis and recommendations)

Usage:
    python main.py

The application will:
1. Download or load data from MongoDB
2. Clean and preprocess data
3. Calculate financial metrics and store in MongoDB
4. Generate visualizations
5. Create insights and reports
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import our custom modules
from scripts.data_collection import DataCollector
from scripts.data_preprocessing import DataPreprocessor
from scripts.calculations import FinancialCalculator
from scripts.visualization import ChartGenerator
from scripts.insights_reporting import InsightsGenerator
from scripts.mongodb_utils import MongoDBManager, MongoDBContext


class SectorAnalyticsApp:
    """Main application class that orchestrates all modules."""
    
    def __init__(self, config: Optional[Dict] = None, mongo_manager: Optional[MongoDBManager] = None):
        """
        Initialize the application with configuration.
        
        Args:
            config: Optional configuration dictionary
            mongo_manager: Optional MongoDB manager instance
        """
        self.config = config or self.get_default_config()
        self.mongo_manager = mongo_manager or MongoDBManager()
        self.data = None
        self.results = {}
        self.analysis_results = {}
        
        # Initialize modules with MongoDB manager
        self.data_collector = DataCollector(self.mongo_manager)
        self.preprocessor = DataPreprocessor(self.mongo_manager)
        self.calculator = FinancialCalculator(self.mongo_manager)
        self.chart_generator = ChartGenerator()
        self.insights_generator = InsightsGenerator()
    
    def get_default_config(self) -> Dict:
        """Get default configuration settings."""
        return {
            'use_real_data': True,
            'download_fresh_data': False,
            'data_folder': 'data',
            'start_date': None,  # None means use all available data
            'end_date': None,
            'sectors': None,  # None means use all available sectors
            'remove_outliers': False,
            'handle_missing_dates': True,
            'investment_method': 'market_cap',  # 'market_cap', 'performance', 'equal_weight'
            'generate_charts': True,
            'generate_reports': True,
            'save_results': True
        }
    
    def collect_data(self) -> None:
        """Collect data from various sources."""
        print("\n" + "="*60)
        print("STEP 1: DATA COLLECTION")
        print("="*60)
        
        if self.config['use_real_data']:
            if self.config['download_fresh_data']:
                print("Downloading fresh data from Yahoo Finance...")
                success = self.data_collector.download_all_sectors()
                if success:
                    print("Data downloaded and stored in MongoDB")
                else:
                    print("Failed to download data")
            else:
                # Try to load existing data from MongoDB
                try:
                    print("Loading existing data from MongoDB...")
                    self.data = self.data_collector.get_data_from_mongodb()
                    if self.data.empty:
                        print("No data found in MongoDB, downloading fresh data...")
                        success = self.data_collector.download_all_sectors()
                        if success:
                            self.data = self.data_collector.get_data_from_mongodb()
                except Exception as e:
                    print(f"Could not load existing data: {e}")
                    print("Downloading fresh data instead...")
                    success = self.data_collector.download_all_sectors()
                    if success:
                        self.data = self.data_collector.get_data_from_mongodb()
        else:
            print("Creating sample data for testing...")
            success = self.data_collector.create_sample_dataset()
            if success:
                self.data = self.data_collector.get_data_from_mongodb()
        
        if self.data is not None and not self.data.empty:
            print(f"Data collection completed. Shape: {self.data.shape}")
            print(f"Sectors: {self.data['Sector'].unique()}")
            print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        else:
            print("No data available for analysis")
    
    def preprocess_data(self) -> None:
        """Preprocess and clean the data."""
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        self.data = self.preprocessor.preprocess_pipeline(
            data=self.data,
            start_date=self.config['start_date'],
            end_date=self.config['end_date'],
            sectors=self.config['sectors'],
            remove_outliers=self.config['remove_outliers'],
            handle_missing_dates=self.config['handle_missing_dates'],
            use_mongodb=True
        )
        
        if self.data is not None and not self.data.empty:
            print(f"Data preprocessing completed. Final shape: {self.data.shape}")
        else:
            print("No data available after preprocessing")
    
    def calculate_metrics(self) -> None:
        """Calculate all financial metrics."""
        print("\n" + "="*60)
        print("STEP 3: FINANCIAL CALCULATIONS")
        print("="*60)
        
        if self.data is None or self.data.empty:
            print("No data available for calculations")
            return
        
        # Calculate investment distribution
        investment_dist = self.calculator.calculate_investment_distribution(
            self.data, method=self.config['investment_method']
        )
        
        # Run all calculations with MongoDB integration
        self.results = self.calculator.run_all_calculations(
            data=self.data,
            weights=investment_dist,
            use_mongodb=True,
            start_date=pd.to_datetime(self.config['start_date']) if self.config['start_date'] else None,
            end_date=pd.to_datetime(self.config['end_date']) if self.config['end_date'] else None,
            sectors=self.config['sectors']
        )
        
        # Print summary
        if self.results:
            self.calculator.print_summary(self.results)
        else:
            print("No calculation results available")
    
    def generate_visualizations(self) -> None:
        """Generate all charts and visualizations."""
        if not self.config['generate_charts']:
            print("\nSkipping chart generation (disabled in config)")
            return
        
        print("\n" + "="*60)
        print("STEP 4: VISUALIZATION")
        print("="*60)
        
        # Get cumulative returns data
        cumulative_data = self.results.get('cumulative_returns', self.data)
        
        # Generate all charts
        saved_charts = self.chart_generator.save_all_charts(self.results, cumulative_data)
        
        print(f"Generated {len(saved_charts)} charts:")
        for chart in saved_charts:
            print(f"  - {chart}")
    
    def generate_insights(self) -> None:
        """Generate insights and reports."""
        if not self.config['generate_reports']:
            print("\nSkipping insights generation (disabled in config)")
            return
        
        print("\n" + "="*60)
        print("STEP 5: INSIGHTS & REPORTING")
        print("="*60)
        
        # Run complete analysis
        self.analysis_results = self.insights_generator.run_complete_analysis(self.results)
        
        # Print executive summary
        print("\nEXECUTIVE SUMMARY:")
        print("-" * 40)
        print(self.analysis_results['executive_summary'])
    
    def save_results(self) -> None:
        """Save all results to files."""
        if not self.config['save_results']:
            print("\nSkipping results saving (disabled in config)")
            return
        
        print("\n" + "="*60)
        print("STEP 6: SAVING RESULTS")
        print("="*60)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_collector.save_data_to_csv(self.data, f"processed_data_{timestamp}.csv")
        
        # Save calculation results as JSON
        import json
        results_to_save = {}
        for key, value in self.results.items():
            if isinstance(value, pd.Series):
                results_to_save[key] = value.to_dict()
            elif isinstance(value, pd.DataFrame):
                results_to_save[key] = value.to_dict()
            else:
                results_to_save[key] = value
        
        with open(f"results/calculation_results_{timestamp}.json", 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        print(f"Results saved with timestamp: {timestamp}")
    
    def run_complete_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        print("STOCK MARKET SECTOR PERFORMANCE ANALYTICS")
        print("=" * 60)
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration: {self.config}")
        
        try:
            # Run all steps
            self.collect_data()
            self.preprocess_data()
            self.calculate_metrics()
            self.generate_visualizations()
            self.generate_insights()
            self.save_results()
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Analysis finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Print final summary
            if self.results and 'cagr' in self.results:
                best_sector = self.results['cagr'].idxmax()
                best_return = self.results['cagr'].max()
                print(f"Best performing sector: {best_sector} ({best_return:.1%} CAGR)")
            
            if self.analysis_results and 'recommendations' in self.analysis_results:
                print(f"Generated {len(self.analysis_results['recommendations'])} investment recommendations")
            
        except Exception as e:
            print(f"\nERROR: Analysis failed with error: {str(e)}")
            print("Please check the error message and try again.")
            sys.exit(1)


def main():
    """Main function to run the application."""
    print("Stock Market Sector Performance Analytics")
    print("=========================================")
    
    # Configuration options
    config = {
        'use_real_data': True,           # Use real Yahoo Finance data
        'download_fresh_data': False,    # Download fresh data (True) or use existing (False)
        'start_date': None,              # Filter start date (e.g., '2024-01-01')
        'end_date': None,                # Filter end date (e.g., '2024-12-31')
        'sectors': None,                 # Filter sectors (e.g., ['IT', 'Pharma'])
        'remove_outliers': False,        # Remove outliers from data
        'handle_missing_dates': True,    # Fill missing dates
        'investment_method': 'market_cap',  # 'market_cap', 'performance', 'equal_weight'
        'generate_charts': True,         # Generate visualization charts
        'generate_reports': True,        # Generate insights and reports
        'save_results': True             # Save results to files
    }
    
    try:
        # Create and run the application with MongoDB
        with MongoDBContext() as mongo_manager:
            app = SectorAnalyticsApp(config, mongo_manager)
            app.run_complete_analysis()
    except Exception as e:
        print(f"Error running application: {e}")
        print("Make sure MongoDB is properly configured with MONGODB_URI environment variable")


if __name__ == "__main__":
    main()
