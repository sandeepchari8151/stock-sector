"""
Calculations Module
==================

This module handles all financial calculations and analytics:
- Daily returns calculation
- Volatility (standard deviation) computation
- CAGR (Compound Annual Growth Rate) calculation
- Risk metrics and performance indicators
- Portfolio allocation calculations
- MongoDB integration for data storage and retrieval

Dependencies: pandas, numpy, pymongo
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from .mongodb_utils import MongoDBManager


class FinancialCalculator:
    """Handles all financial calculations and analytics."""
    
    def __init__(self, mongo_manager: Optional[MongoDBManager] = None):
        self.mongo_manager = mongo_manager or MongoDBManager()
        self.calculation_results = {}
    
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
    
    def calculate_daily_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns for each sector.
        
        Args:
            data: DataFrame with Date, Sector, Close columns
        
        Returns:
            DataFrame with DailyReturn column added
        """
        print("Calculating daily returns...")
        
        # Sort by date to ensure correct order
        df_sorted = data.sort_values(['Sector', 'Date']).copy()
        
        # Calculate percentage change within each sector
        df_sorted['DailyReturn'] = df_sorted.groupby('Sector')['Close'].pct_change()
        
        # Remove first day per sector (NaN return)
        df_returns = df_sorted.dropna(subset=['DailyReturn']).copy()
        
        print(f"Calculated returns for {len(df_returns)} records")
        return df_returns
    
    def calculate_volatility(self, returns_data: pd.DataFrame, 
                           period: str = 'daily') -> pd.Series:
        """
        Calculate volatility (standard deviation of returns) for each sector.
        
        Args:
            returns_data: DataFrame with DailyReturn column
            period: 'daily', 'monthly', or 'annual'
        
        Returns:
            Series with volatility by sector
        """
        print(f"Calculating {period} volatility...")
        
        volatility = returns_data.groupby('Sector')['DailyReturn'].std()
        
        # Convert to different periods if needed
        if period == 'monthly':
            volatility = volatility * np.sqrt(21)  # ~21 trading days per month
        elif period == 'annual':
            volatility = volatility * np.sqrt(252)  # ~252 trading days per year
        
        self.calculation_results['volatility'] = volatility
        return volatility
    
    def calculate_cagr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Compound Annual Growth Rate (CAGR) for each sector.
        
        Args:
            data: DataFrame with Date, Sector, Close columns
        
        Returns:
            Series with CAGR by sector
        """
        print("Calculating CAGR...")
        
        cagr_values = {}
        df_sorted = data.sort_values(['Sector', 'Date'])
        
        for sector, df_sector in df_sorted.groupby('Sector'):
            if len(df_sector) < 2:
                continue
                
            first_close = df_sector['Close'].iloc[0]
            last_close = df_sector['Close'].iloc[-1]
            
            # Calculate time period in years
            num_days = (df_sector['Date'].iloc[-1] - df_sector['Date'].iloc[0]).days
            years = max(num_days / 365.25, 1e-9)  # Avoid division by zero
            
            # CAGR formula: (Ending Value / Beginning Value)^(1/years) - 1
            cagr = (last_close / first_close) ** (1.0 / years) - 1.0
            cagr_values[sector] = cagr
        
        cagr_series = pd.Series(cagr_values)
        self.calculation_results['cagr'] = cagr_series
        return cagr_series
    
    def calculate_cumulative_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative return index for each sector.
        
        Args:
            data: DataFrame with Date, Sector, Close columns
        
        Returns:
            DataFrame with CumReturnIndex column
        """
        print("Calculating cumulative returns...")
        
        df_sorted = data.sort_values(['Sector', 'Date']).copy()
        
        # Calculate cumulative return index (rebased to 1.0)
        df_sorted['CumReturnIndex'] = df_sorted.groupby('Sector')['Close'].transform(
            lambda s: s / s.iloc[0]
        )
        
        return df_sorted
    
    def calculate_average_daily_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """
        Calculate average daily returns for each sector.
        
        Args:
            returns_data: DataFrame with DailyReturn column
        
        Returns:
            Series with average daily returns by sector
        """
        print("Calculating average daily returns...")
        
        avg_returns = returns_data.groupby('Sector')['DailyReturn'].mean()
        self.calculation_results['avg_daily_returns'] = avg_returns
        return avg_returns
    
    def calculate_sharpe_ratio(self, returns_data: pd.DataFrame, 
                              risk_free_rate: float = 0.02) -> pd.Series:
        """
        Calculate Sharpe ratio for each sector.
        
        Args:
            returns_data: DataFrame with DailyReturn column
            risk_free_rate: Annual risk-free rate (default 2%)
        
        Returns:
            Series with Sharpe ratios by sector
        """
        print("Calculating Sharpe ratios...")
        
        # Convert annual risk-free rate to daily
        daily_rf_rate = risk_free_rate / 252
        
        sharpe_ratios = {}
        
        for sector, sector_returns in returns_data.groupby('Sector'):
            if len(sector_returns) < 2:
                continue
            
            returns = sector_returns['DailyReturn']
            excess_returns = returns - daily_rf_rate
            
            # Annualized Sharpe ratio
            sharpe = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
            sharpe_ratios[sector] = sharpe
        
        sharpe_series = pd.Series(sharpe_ratios)
        self.calculation_results['sharpe_ratio'] = sharpe_series
        return sharpe_series
    
    def calculate_max_drawdown(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate maximum drawdown for each sector.
        
        Args:
            data: DataFrame with Date, Sector, Close columns
        
        Returns:
            Series with maximum drawdown by sector
        """
        print("Calculating maximum drawdown...")
        
        max_drawdowns = {}
        df_sorted = data.sort_values(['Sector', 'Date'])
        
        for sector, df_sector in df_sorted.groupby('Sector'):
            if len(df_sector) < 2:
                continue
            
            prices = df_sector['Close']
            cumulative_max = prices.expanding().max()
            drawdown = (prices - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()
            max_drawdowns[sector] = max_drawdown
        
        max_dd_series = pd.Series(max_drawdowns)
        self.calculation_results['max_drawdown'] = max_dd_series
        return max_dd_series
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix between sectors.
        
        Args:
            returns_data: DataFrame with Date, Sector, DailyReturn columns
        
        Returns:
            Correlation matrix DataFrame
        """
        print("Calculating correlation matrix...")
        
        # Pivot to get returns by sector
        returns_pivot = returns_data.pivot(index='Date', columns='Sector', values='DailyReturn')
        
        # Calculate correlation matrix
        correlation_matrix = returns_pivot.corr()
        
        self.calculation_results['correlation_matrix'] = correlation_matrix
        return correlation_matrix
    
    def calculate_portfolio_metrics(self, returns_data: pd.DataFrame, 
                                  weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio-level metrics.
        
        Args:
            returns_data: DataFrame with Date, Sector, DailyReturn columns
            weights: Dictionary of sector weights
        
        Returns:
            Dictionary with portfolio metrics
        """
        print("Calculating portfolio metrics...")
        
        # Pivot to get returns by sector
        returns_pivot = returns_data.pivot(index='Date', columns='Sector', values='DailyReturn')
        
        # Align weights with available sectors
        available_sectors = returns_pivot.columns
        aligned_weights = {sector: weights.get(sector, 0) for sector in available_sectors}
        
        # Normalize weights to sum to 1
        total_weight = sum(aligned_weights.values())
        if total_weight > 0:
            aligned_weights = {k: v/total_weight for k, v in aligned_weights.items()}
        
        # Calculate weighted portfolio returns
        portfolio_returns = returns_pivot.multiply(
            pd.Series(aligned_weights), axis=1
        ).sum(axis=1)
        
        # Calculate portfolio metrics
        portfolio_metrics = {
            'expected_return': portfolio_returns.mean() * 252,  # Annualized
            'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': ((portfolio_returns.cumsum().expanding().max() - portfolio_returns.cumsum()).max())
        }
        
        self.calculation_results['portfolio_metrics'] = portfolio_metrics
        return portfolio_metrics
    
    def calculate_investment_distribution(self, data: pd.DataFrame, 
                                        method: str = 'market_cap') -> Dict[str, float]:
        """
        Calculate investment distribution based on different methods.
        
        Args:
            data: DataFrame with Date, Sector, Close columns
            method: 'market_cap', 'performance', 'equal_weight', or 'volatility_adjusted'
        
        Returns:
            Dictionary with sector weights
        """
        print(f"Calculating investment distribution using {method} method...")
        
        if method == 'equal_weight':
            sectors = data['Sector'].unique()
            weight = 1.0 / len(sectors)
            return {sector: weight for sector in sectors}
        
        elif method == 'market_cap':
            # Weight by latest market price (proxy for market cap)
            latest_prices = data.groupby('Sector')['Close'].last()
            total_value = latest_prices.sum()
            weights = latest_prices / total_value
            return weights.to_dict()
        
        elif method == 'performance':
            # Weight by cumulative performance
            df_sorted = data.sort_values(['Sector', 'Date'])
            df_sorted['CumReturnIndex'] = df_sorted.groupby('Sector')['Close'].transform(
                lambda s: s / s.iloc[0]
            )
            latest_performance = df_sorted.groupby('Sector')['CumReturnIndex'].last()
            total_performance = latest_performance.sum()
            weights = latest_performance / total_performance
            return weights.to_dict()
        
        elif method == 'volatility_adjusted':
            # Weight inversely to volatility (lower volatility = higher weight)
            returns_data = self.calculate_daily_returns(data)
            volatility = self.calculate_volatility(returns_data)
            inverse_vol = 1 / volatility
            total_inverse_vol = inverse_vol.sum()
            weights = inverse_vol / total_inverse_vol
            return weights.to_dict()
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def run_all_calculations(self, data: Optional[pd.DataFrame] = None, 
                           weights: Optional[Dict[str, float]] = None,
                           use_mongodb: bool = True,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           sectors: Optional[List[str]] = None) -> Dict:
        """
        Run all financial calculations and return comprehensive results.
        
        Args:
            data: DataFrame with Date, Sector, Close columns (if None, will fetch from MongoDB)
            weights: Optional portfolio weights
            use_mongodb: Whether to use MongoDB for data source and storage
            start_date: Optional start date filter
            end_date: Optional end date filter
            sectors: Optional list of sectors to include
        
        Returns:
            Dictionary with all calculation results
        """
        print("Running comprehensive financial calculations...")
        
        # Get data from MongoDB if not provided
        if data is None and use_mongodb:
            data = self.get_data_from_mongodb(start_date, end_date, sectors)
        
        if data is None or data.empty:
            print("No data available for calculations")
            return {}
        
        results = {}
        
        # Basic calculations
        returns_data = self.calculate_daily_returns(data)
        results['returns_data'] = returns_data
        
        results['volatility'] = self.calculate_volatility(returns_data)
        results['cagr'] = self.calculate_cagr(data)
        results['avg_daily_returns'] = self.calculate_average_daily_returns(returns_data)
        results['cumulative_returns'] = self.calculate_cumulative_returns(data)
        
        # Advanced metrics
        results['sharpe_ratio'] = self.calculate_sharpe_ratio(returns_data)
        results['max_drawdown'] = self.calculate_max_drawdown(data)
        results['correlation_matrix'] = self.calculate_correlation_matrix(returns_data)
        
        # Investment distribution
        results['investment_distribution'] = self.calculate_investment_distribution(data)
        
        # Portfolio metrics (if weights provided)
        if weights:
            results['portfolio_metrics'] = self.calculate_portfolio_metrics(returns_data, weights)
        
        # Store results in MongoDB if using MongoDB
        if use_mongodb:
            self._store_calculation_results_to_mongodb(results)
        
        # Store results
        self.calculation_results = results
        
        print("All calculations completed!")
        return results
    
    def _store_calculation_results_to_mongodb(self, results: Dict) -> None:
        """
        Store calculation results to MongoDB.
        
        Args:
            results: Dictionary containing calculation results
        """
        try:
            # Store sector metrics
            if 'cagr' in results and 'volatility' in results:
                metrics_data = []
                calculation_date = datetime.utcnow()
                
                for sector in results['cagr'].index:
                    metric_doc = {
                        "sector_code": sector,
                        "calculation_date": calculation_date,
                        "period_type": "daily",
                        "period_start": calculation_date - pd.Timedelta(days=365),
                        "period_end": calculation_date,
                        "cagr": float(results['cagr'][sector]),
                        "volatility": float(results['volatility'][sector]),
                        "sharpe_ratio": float(results.get('sharpe_ratio', {}).get(sector, 0)),
                        "max_drawdown": float(results.get('max_drawdown', {}).get(sector, 0)),
                        "avg_daily_return": float(results.get('avg_daily_returns', {}).get(sector, 0))
                    }
                    metrics_data.append(metric_doc)
                
                if metrics_data:
                    self.mongo_manager.insert_sector_metrics(metrics_data)
                    print(f"Stored {len(metrics_data)} sector metric records to MongoDB")
            
            # Store correlation matrix
            if 'correlation_matrix' in results:
                correlation_data = []
                calculation_date = datetime.utcnow()
                
                corr_matrix = results['correlation_matrix']
                for i, sector1 in enumerate(corr_matrix.index):
                    for j, sector2 in enumerate(corr_matrix.columns):
                        if i < j:  # Only store upper triangle
                            corr_doc = {
                                "calculation_date": calculation_date,
                                "sector_1": sector1,
                                "sector_2": sector2,
                                "correlation_value": float(corr_matrix.loc[sector1, sector2]),
                                "period_days": 30
                            }
                            correlation_data.append(corr_doc)
                
                if correlation_data:
                    # Note: This would need a separate method in MongoDBManager
                    # For now, we'll skip storing correlation matrix
                    pass
            
            # Store portfolio analysis if available
            if 'portfolio_metrics' in results and 'investment_distribution' in results:
                portfolio_doc = {
                    "analysis_name": f"Portfolio Analysis {datetime.utcnow().strftime('%Y-%m-%d')}",
                    "analysis_type": "sector_analysis",
                    "investment_method": "market_cap",  # Default method
                    "start_date": datetime.utcnow() - pd.Timedelta(days=365),
                    "end_date": datetime.utcnow(),
                    "sectors_included": list(results['investment_distribution'].keys()),
                    "weights": results['investment_distribution'],
                    "total_return": results['portfolio_metrics'].get('expected_return', 0),
                    "portfolio_volatility": results['portfolio_metrics'].get('volatility', 0),
                    "portfolio_sharpe": results['portfolio_metrics'].get('sharpe_ratio', 0),
                    "max_drawdown": results['portfolio_metrics'].get('max_drawdown', 0)
                }
                
                # Insert portfolio analysis
                self.mongo_manager.db.portfolio_analyses.insert_one(portfolio_doc)
                print("Stored portfolio analysis to MongoDB")
                
        except Exception as e:
            print(f"Error storing calculation results to MongoDB: {e}")
    
    def print_summary(self, results: Dict = None) -> None:
        """Print a summary of calculation results."""
        if results is None:
            results = self.calculation_results
        
        print("\n" + "="*60)
        print("FINANCIAL CALCULATIONS SUMMARY")
        print("="*60)
        
        if 'cagr' in results:
            print("\nCAGR (Compound Annual Growth Rate):")
            for sector, cagr in results['cagr'].items():
                print(f"  {sector}: {cagr:.2%}")
        
        if 'volatility' in results:
            print("\nVolatility (Daily Standard Deviation):")
            for sector, vol in results['volatility'].items():
                print(f"  {sector}: {vol:.2%}")
        
        if 'sharpe_ratio' in results:
            print("\nSharpe Ratio:")
            for sector, sharpe in results['sharpe_ratio'].items():
                print(f"  {sector}: {sharpe:.2f}")
        
        if 'max_drawdown' in results:
            print("\nMaximum Drawdown:")
            for sector, dd in results['max_drawdown'].items():
                print(f"  {sector}: {dd:.2%}")
        
        if 'investment_distribution' in results:
            print("\nInvestment Distribution:")
            for sector, weight in results['investment_distribution'].items():
                print(f"  {sector}: {weight:.1%}")
        
        if 'portfolio_metrics' in results:
            print("\nPortfolio Metrics:")
            for metric, value in results['portfolio_metrics'].items():
                print(f"  {metric}: {value:.2%}" if 'return' in metric or 'volatility' in metric or 'drawdown' in metric else f"  {metric}: {value:.2f}")
        
        print("="*60)


def main():
    """Example usage of the FinancialCalculator class."""
    try:
        from .mongodb_utils import MongoDBContext
        
        with MongoDBContext() as mongo_manager:
            calculator = FinancialCalculator(mongo_manager)
            
            # Run all calculations using MongoDB
            results = calculator.run_all_calculations(use_mongodb=True)
            
            if results:
                # Print summary
                calculator.print_summary(results)
            else:
                print("No data available for calculations")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
