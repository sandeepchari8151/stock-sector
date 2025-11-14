"""
Visualization Module
====================

This module handles all chart creation and dashboard generation:
- Line charts for performance trends
- Bar charts for comparisons
- Heatmaps for correlation and risk analysis
- Pie charts for portfolio allocation
- Comprehensive dashboard creation

Dependencies: matplotlib, pandas, numpy
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime


class ChartGenerator:
    """Handles all chart generation and visualization."""
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 6), charts_folder: str = 'charts'):
        self.style = style
        self.default_figsize = figsize
        self.charts_folder = charts_folder
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        self.setup_style()
        
        # Create charts folder if it doesn't exist
        import os
        if not os.path.exists(self.charts_folder):
            os.makedirs(self.charts_folder)
    
    def setup_style(self) -> None:
        """Setup matplotlib style and configuration."""
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_sector_trends(self, cumulative_data: pd.DataFrame) -> None:
        """
        Create line chart showing sector trends over time.
        Easy explanation: "This line chart shows the performance of each sector. IT and Pharma are going up, Banking is more stable."
        """
        print("Creating Line Chart - Sector Trends Over Time...")
        
        # Pivot data for plotting
        pivot = cumulative_data.pivot(index='Date', columns='Sector', values='CumReturnIndex')
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot each sector with different colors
        for i, (sector, series) in enumerate(pivot.items()):
            ax.plot(series.index, series.values, 
                   label=sector, 
                   linewidth=3, 
                   color=self.color_palette[i % len(self.color_palette)])
        
        # Add reference line at 1.0 (starting point)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        
        # Formatting
        ax.set_title("Line Chart – Sector Trends Over Time", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Performance Index (Starting = 1.0)", fontsize=14)
        ax.legend(title="Sectors", fontsize=12, title_fontsize=14, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        
        # Add explanation text
        explanation = "This line chart shows the performance of each sector.\nIT and Pharma are going up, Banking is more stable."
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.charts_folder}/line_chart_sector_trends.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_sector_performance_bars(self, cagr: pd.Series) -> None:
        """
        Create bar chart showing sector performance (CAGR).
        Easy explanation: "The taller the bar, the better the sector performed on average."
        """
        print("Creating Bar Chart - Sector Performance (CAGR)...")
        
        # Sort by value for better visualization
        sorted_cagr = cagr.sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create bars with gradient colors
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_cagr)))
        bars = ax.bar(sorted_cagr.index, sorted_cagr.values, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, sorted_cagr.values):
            height = bar.get_height()
            # Position label above positive bars, below negative bars
            if value >= 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.05,
                       f'{value:.1%}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height - abs(height)*0.05,
                       f'{value:.1%}', ha='center', va='top', 
                       fontweight='bold', fontsize=12)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_title("Bar Chart – Sector Performance (Annual Growth)", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Sector", fontsize=14)
        ax.set_ylabel("Compound Annual Growth Rate (CAGR)", fontsize=14)
        
        # Set y-axis limits to show all data clearly
        y_max = max(sorted_cagr.values) * 1.2 if max(sorted_cagr.values) > 0 else max(sorted_cagr.values) * 0.8
        y_min = min(sorted_cagr.values) * 1.2 if min(sorted_cagr.values) < 0 else min(sorted_cagr.values) * 0.8
        ax.set_ylim(y_min, y_max)
        
        ax.grid(axis="y", alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add explanation text
        explanation = "The taller the bar, the better the sector\nperformed on average."
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{self.charts_folder}/bar_chart_sector_performance.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_investment_distribution_pie(self, distribution: Dict[str, float]) -> None:
        """
        Create pie chart showing investment distribution.
        Easy explanation: "This pie chart shows how money is divided across sectors. Banking attracts the largest share."
        """
        print("Creating Pie Chart - Investment Distribution...")
        
        # Sort by weight for better visualization
        sorted_data = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        sectors, weights = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart with custom colors
        colors = [self.color_palette[i % len(self.color_palette)] for i in range(len(sectors))]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(weights, labels=sectors, autopct='%1.1f%%', 
                                         startangle=90, colors=colors, textprops={'fontsize': 12})
        
        # Enhance text formatting
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        # Formatting
        ax.set_title("Pie Chart – Investment Distribution", fontsize=16, fontweight='bold', pad=20)
        
        # Add explanation text
        explanation = "This pie chart shows how money is divided across sectors.\nBanking attracts the largest share."
        ax.text(0.02, 0.02, explanation, transform=ax.transAxes, 
               verticalalignment='bottom', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(f"{self.charts_folder}/pie_chart_investment_distribution.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_volatility_vs_growth_scatter(self, volatility: pd.Series, cagr: pd.Series) -> None:
        """
        Create scatter plot showing volatility vs growth.
        Easy explanation: "This scatter plot shows risk vs reward. Pharma has steady growth with low risk, IT has high growth but also high risk."
        """
        print("Creating Scatter Plot - Volatility vs Growth...")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create scatter plot
        for i, sector in enumerate(volatility.index):
            ax.scatter(volatility[sector], cagr[sector], 
                      s=300, alpha=0.7, 
                      color=self.color_palette[i % len(self.color_palette)],
                      label=sector, edgecolors='black', linewidth=2)
            
            # Add sector labels
            ax.annotate(sector, (volatility[sector], cagr[sector]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add quadrant lines
        ax.axhline(y=cagr.mean(), color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=volatility.mean(), color='gray', linestyle='--', alpha=0.5, linewidth=2)
        
        # Formatting
        ax.set_xlabel("Volatility (Risk)", fontsize=14)
        ax.set_ylabel("Growth (CAGR)", fontsize=14)
        ax.set_title("Scatter Plot – Volatility vs Growth", fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add explanation text
        explanation = "This scatter plot shows risk vs reward.\nPharma has steady growth with low risk,\nIT has high growth but also high risk."
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        # Format axes
        ax.tick_params(axis='both', labelsize=11)
        
        plt.tight_layout()
        plt.savefig(f"{self.charts_folder}/scatter_plot_volatility_vs_growth.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_volatility_vs_growth_heatmap(self, volatility: pd.Series, cagr: pd.Series,
                                        title: str = "Volatility vs Growth Analysis") -> None:
        """
        Create heatmap comparing volatility and growth.
        
        Args:
            volatility: Series with volatility by sector
            cagr: Series with CAGR by sector
            title: Chart title
        """
        print("Creating volatility vs growth heatmap...")
        
        # Align data
        sectors = sorted(set(volatility.index).intersection(set(cagr.index)))
        vol_values = volatility.reindex(sectors).values
        cagr_values = cagr.reindex(sectors).values
        
        # Create data matrix
        data = np.vstack([vol_values, cagr_values])
        
        fig, ax = plt.subplots(figsize=(max(8, len(sectors) * 1.5), 6))
        
        # Create heatmap
        im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Value", fontsize=12)
        
        # Set labels
        ax.set_xticks(np.arange(len(sectors)))
        ax.set_xticklabels(sectors, rotation=45, ha="right", fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Volatility (Daily Std)", "Growth (CAGR)"], fontsize=11)
        
        # Add values to cells
        for y in range(2):
            for x in range(len(sectors)):
                value = data[y, x]
                text = f"{value:.1%}" if y == 1 else f"{value:.2%}"
                text_color = "white" if value < data.max()/2 else "black"
                
                ax.text(x, y, text, ha="center", va="center", 
                       color=text_color, fontweight='bold', fontsize=10)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.set_xticks(np.arange(len(sectors)) - 0.5, minor=True)
        ax.set_yticks(np.arange(2) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
        
        # Add summary
        max_vol_sector = sectors[np.argmax(vol_values)]
        max_growth_sector = sectors[np.argmax(cagr_values)]
        
        summary_text = f"Highest Volatility: {max_vol_sector}\nHighest Growth: {max_growth_sector}"
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig("volatility_growth_heatmap.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                               title: str = "Sector Correlation Matrix") -> None:
        """
        Create correlation heatmap between sectors.
        
        Args:
            correlation_matrix: DataFrame with correlation values
            title: Chart title
        """
        print("Creating correlation heatmap...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap with seaborn
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_risk_return_scatter(self, volatility: pd.Series, cagr: pd.Series,
                               title: str = "Risk vs Return Analysis") -> None:
        """
        Create scatter plot of risk vs return.
        
        Args:
            volatility: Series with volatility by sector
            cagr: Series with CAGR by sector
            title: Chart title
        """
        print("Creating risk-return scatter plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        for i, sector in enumerate(volatility.index):
            ax.scatter(volatility[sector], cagr[sector], 
                      s=200, alpha=0.7, 
                      color=self.color_palette[i % len(self.color_palette)],
                      label=sector)
            
            # Add sector labels
            ax.annotate(sector, (volatility[sector], cagr[sector]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel("Volatility (Risk)", fontsize=12)
        ax.set_ylabel("CAGR (Return)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax.axhline(y=cagr.mean(), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=volatility.mean(), color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("risk_return_scatter.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self, results: Dict, 
                               title: str = "Performance Metrics Comparison") -> None:
        """
        Create bar chart comparing multiple performance metrics.
        
        Args:
            results: Dictionary with calculation results
            title: Chart title
        """
        print("Creating performance metrics chart...")
        
        # Extract metrics
        metrics_data = {}
        if 'cagr' in results:
            metrics_data['CAGR'] = results['cagr']
        if 'volatility' in results:
            metrics_data['Volatility'] = results['volatility']
        if 'sharpe_ratio' in results:
            metrics_data['Sharpe Ratio'] = results['sharpe_ratio']
        
        if not metrics_data:
            print("No metrics data available for plotting")
            return
        
        # Create subplots
        n_metrics = len(metrics_data)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric_name, metric_data) in enumerate(metrics_data.items()):
            ax = axes[i]
            
            # Sort by value
            sorted_data = metric_data.sort_values(ascending=False)
            
            # Create bar chart
            bars = ax.bar(sorted_data.index, sorted_data.values, 
                         color=self.color_palette[:len(sorted_data)])
            
            # Add value labels
            for bar, value in zip(bars, sorted_data.values):
                height = bar.get_height()
                if 'ratio' in metric_name.lower():
                    label = f'{value:.2f}'
                else:
                    label = f'{value:.1%}'
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       label, ha='center', va='bottom', fontweight='bold')
            
            # Formatting
            ax.set_title(f"{metric_name} by Sector", fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("performance_metrics_chart.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self, results: Dict, cumulative_data: pd.DataFrame) -> None:
        """
        Create a comprehensive dashboard with multiple charts.
        
        Args:
            results: Dictionary with all calculation results
            cumulative_data: DataFrame with cumulative returns
        """
        print("Creating comprehensive dashboard...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Define subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Sector Performance (top, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        pivot = cumulative_data.pivot(index='Date', columns='Sector', values='CumReturnIndex')
        for i, (sector, series) in enumerate(pivot.items()):
            ax1.plot(series.index, series.values, label=sector, 
                    color=self.color_palette[i % len(self.color_palette)], linewidth=2)
        ax1.set_title("Sector Performance Over Time", fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
        
        # 2. Investment Distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'investment_distribution' in results:
            dist = results['investment_distribution']
            ax2.pie(dist.values(), labels=dist.keys(), autopct='%1.1f%%', startangle=90)
            ax2.set_title("Investment Distribution", fontweight='bold')
        
        # 3. Volatility vs Growth Heatmap (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'volatility' in results and 'cagr' in results:
            vol = results['volatility']
            cagr = results['cagr']
            sectors = sorted(set(vol.index).intersection(set(cagr.index)))
            data = np.vstack([vol.reindex(sectors).values, cagr.reindex(sectors).values])
            im = ax3.imshow(data, aspect="auto", cmap="RdYlBu_r")
            ax3.set_xticks(np.arange(len(sectors)))
            ax3.set_xticklabels(sectors)
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["Volatility", "CAGR"])
            ax3.set_title("Volatility vs Growth", fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Risk-Return Scatter (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        if 'volatility' in results and 'cagr' in results:
            vol = results['volatility']
            cagr = results['cagr']
            for i, sector in enumerate(vol.index):
                ax4.scatter(vol[sector], cagr[sector], 
                           color=self.color_palette[i % len(self.color_palette)],
                           s=100, alpha=0.7, label=sector)
                ax4.annotate(sector, (vol[sector], cagr[sector]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax4.set_xlabel("Volatility")
            ax4.set_ylabel("CAGR")
            ax4.set_title("Risk vs Return", fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance Metrics (bottom, spans all columns)
        ax5 = fig.add_subplot(gs[2, :])
        if 'cagr' in results:
            cagr_data = results['cagr'].sort_values(ascending=False)
            bars = ax5.bar(cagr_data.index, cagr_data.values, 
                          color=self.color_palette[:len(cagr_data)])
            for bar, value in zip(bars, cagr_data.values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
            ax5.set_title("CAGR by Sector", fontweight='bold')
            ax5.set_ylabel("CAGR")
            ax5.grid(axis='y', alpha=0.3)
        
        # Add overall title
        fig.suptitle("Stock Market Sector Performance Dashboard", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig("sector_performance_dashboard.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_all_charts(self, results: Dict, cumulative_data: pd.DataFrame) -> List[str]:
        """
        Generate and save the 4 specific charts requested.
        
        Args:
            results: Dictionary with calculation results
            cumulative_data: DataFrame with cumulative returns
        
        Returns:
            List of saved chart filenames
        """
        saved_files = []
        
        try:
            # 1. Line Chart - Sector Trends Over Time
            self.plot_sector_trends(cumulative_data)
            saved_files.append("line_chart_sector_trends.png")
        except Exception as e:
            print(f"Error creating line chart: {e}")
        
        try:
            # 2. Bar Chart - Sector Performance (CAGR)
            if 'cagr' in results:
                self.plot_sector_performance_bars(results['cagr'])
                saved_files.append("bar_chart_sector_performance.png")
        except Exception as e:
            print(f"Error creating bar chart: {e}")
        
        try:
            # 3. Pie Chart - Investment Distribution
            if 'investment_distribution' in results:
                self.plot_investment_distribution_pie(results['investment_distribution'])
                saved_files.append("pie_chart_investment_distribution.png")
        except Exception as e:
            print(f"Error creating pie chart: {e}")
        
        try:
            # 4. Scatter Plot - Volatility vs Growth
            if 'volatility' in results and 'cagr' in results:
                self.plot_volatility_vs_growth_scatter(results['volatility'], results['cagr'])
                saved_files.append("scatter_plot_volatility_vs_growth.png")
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
        
        print(f"Saved {len(saved_files)} charts to {self.charts_folder}/ folder:")
        for chart in saved_files:
            print(f"  - {chart}")
        return saved_files


def main():
    """Example usage of the ChartGenerator class."""
    # This would typically be used with results from calculations.py
    generator = ChartGenerator()
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sectors = ['IT', 'Pharma', 'Banking', 'FMCG']
    
    # Sample cumulative returns
    cumulative_data = []
    for sector in sectors:
        np.random.seed(hash(sector) % 2**32)
        returns = np.random.normal(0.0005, 0.01, len(dates))
        cum_returns = np.cumprod(1 + returns)
        
        sector_data = pd.DataFrame({
            'Date': dates,
            'Sector': sector,
            'CumReturnIndex': cum_returns
        })
        cumulative_data.append(sector_data)
    
    cumulative_df = pd.concat(cumulative_data, ignore_index=True)
    
    # Sample results
    results = {
        'cagr': pd.Series([0.15, 0.12, 0.08, 0.10], index=sectors),
        'volatility': pd.Series([0.18, 0.12, 0.15, 0.10], index=sectors),
        'investment_distribution': {sector: 0.25 for sector in sectors}
    }
    
    # Generate charts
    generator.save_all_charts(results, cumulative_df)


if __name__ == "__main__":
    main()
