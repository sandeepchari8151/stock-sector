"""
Insights & Reporting Module
===========================

This module handles analysis, insights generation, and reporting:
- Performance analysis and ranking
- Risk assessment and recommendations
- Investment insights and strategies
- Report generation (text and HTML)
- Executive summary creation

Dependencies: pandas, numpy, datetime
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json


class InsightsGenerator:
    """Generates insights and reports from financial calculations."""
    
    def __init__(self):
        self.insights = {}
        self.recommendations = []
        self.risk_assessment = {}
    
    def analyze_performance(self, results: Dict) -> Dict:
        """
        Analyze sector performance and generate insights.
        
        Args:
            results: Dictionary with calculation results
        
        Returns:
            Dictionary with performance analysis
        """
        print("Analyzing sector performance...")
        
        analysis = {
            'performance_ranking': {},
            'key_insights': [],
            'performance_summary': {}
        }
        
        # CAGR Analysis
        if 'cagr' in results:
            cagr = results['cagr']
            analysis['performance_ranking']['cagr'] = cagr.sort_values(ascending=False)
            
            best_cagr = cagr.idxmax()
            worst_cagr = cagr.idxmin()
            cagr_spread = cagr.max() - cagr.min()
            
            analysis['key_insights'].append(
                f"Best performing sector: {best_cagr} ({cagr[best_cagr]:.1%} CAGR)"
            )
            analysis['key_insights'].append(
                f"Worst performing sector: {worst_cagr} ({cagr[worst_cagr]:.1%} CAGR)"
            )
            analysis['key_insights'].append(
                f"Performance spread: {cagr_spread:.1%} between best and worst sectors"
            )
        
        # Volatility Analysis
        if 'volatility' in results:
            volatility = results['volatility']
            analysis['performance_ranking']['volatility'] = volatility.sort_values(ascending=True)
            
            most_volatile = volatility.idxmax()
            least_volatile = volatility.idxmin()
            
            analysis['key_insights'].append(
                f"Most volatile sector: {most_volatile} ({volatility[most_volatile]:.2%} daily volatility)"
            )
            analysis['key_insights'].append(
                f"Least volatile sector: {least_volatile} ({volatility[least_volatile]:.2%} daily volatility)"
            )
        
        # Sharpe Ratio Analysis
        if 'sharpe_ratio' in results:
            sharpe = results['sharpe_ratio']
            analysis['performance_ranking']['sharpe_ratio'] = sharpe.sort_values(ascending=False)
            
            best_sharpe = sharpe.idxmax()
            analysis['key_insights'].append(
                f"Best risk-adjusted returns: {best_sharpe} (Sharpe ratio: {sharpe[best_sharpe]:.2f})"
            )
        
        # Overall Performance Summary
        if 'cagr' in results and 'volatility' in results:
            cagr = results['cagr']
            vol = results['volatility']
            
            # Calculate risk-adjusted performance
            risk_adjusted = cagr / vol
            best_risk_adjusted = risk_adjusted.idxmax()
            
            analysis['performance_summary'] = {
                'average_cagr': cagr.mean(),
                'average_volatility': vol.mean(),
                'best_risk_adjusted': best_risk_adjusted,
                'risk_adjusted_score': risk_adjusted[best_risk_adjusted]
            }
        
        self.insights['performance_analysis'] = analysis
        return analysis
    
    def assess_risk(self, results: Dict) -> Dict:
        """
        Assess risk across sectors and generate risk insights.
        
        Args:
            results: Dictionary with calculation results
        
        Returns:
            Dictionary with risk assessment
        """
        print("Assessing sector risk...")
        
        risk_assessment = {
            'risk_levels': {},
            'risk_insights': [],
            'diversification_analysis': {}
        }
        
        # Volatility Risk Assessment
        if 'volatility' in results:
            volatility = results['volatility']
            
            # Categorize risk levels
            vol_mean = volatility.mean()
            vol_std = volatility.std()
            
            for sector, vol in volatility.items():
                if vol > vol_mean + vol_std:
                    risk_level = "High"
                elif vol > vol_mean:
                    risk_level = "Medium-High"
                elif vol > vol_mean - vol_std:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                
                risk_assessment['risk_levels'][sector] = {
                    'volatility': vol,
                    'risk_level': risk_level
                }
        
        # Maximum Drawdown Analysis
        if 'max_drawdown' in results:
            max_dd = results['max_drawdown']
            worst_drawdown = max_dd.idxmin()
            best_drawdown = max_dd.idxmax()
            
            risk_assessment['risk_insights'].append(
                f"Worst maximum drawdown: {worst_drawdown} ({max_dd[worst_drawdown]:.1%})"
            )
            risk_assessment['risk_insights'].append(
                f"Best maximum drawdown: {best_drawdown} ({max_dd[best_drawdown]:.1%})"
            )
        
        # Correlation Analysis
        if 'correlation_matrix' in results:
            corr_matrix = results['correlation_matrix']
            
            # Find highest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    sector1 = corr_matrix.columns[i]
                    sector2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    corr_pairs.append((sector1, sector2, corr_value))
            
            # Sort by correlation strength
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            highest_corr = corr_pairs[0] if corr_pairs else None
            if highest_corr:
                risk_assessment['risk_insights'].append(
                    f"Highest correlation: {highest_corr[0]} & {highest_corr[1]} ({highest_corr[2]:.2f})"
                )
            
            # Diversification analysis
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            risk_assessment['diversification_analysis'] = {
                'average_correlation': avg_correlation,
                'diversification_benefit': 'High' if avg_correlation < 0.3 else 'Medium' if avg_correlation < 0.6 else 'Low'
            }
        
        self.risk_assessment = risk_assessment
        return risk_assessment
    
    def generate_investment_recommendations(self, results: Dict) -> List[Dict]:
        """
        Generate investment recommendations based on analysis.
        
        Args:
            results: Dictionary with calculation results
        
        Returns:
            List of recommendation dictionaries
        """
        print("Generating investment recommendations...")
        
        recommendations = []
        
        # Performance-based recommendations
        if 'cagr' in results and 'volatility' in results:
            cagr = results['cagr']
            vol = results['volatility']
            
            # High growth, moderate risk
            growth_sectors = cagr[cagr > cagr.quantile(0.75)]
            moderate_risk_sectors = vol[vol < vol.quantile(0.75)]
            growth_opportunities = growth_sectors.index.intersection(moderate_risk_sectors.index)
            
            if len(growth_opportunities) > 0:
                recommendations.append({
                    'type': 'Growth Opportunity',
                    'sectors': list(growth_opportunities),
                    'reasoning': 'High growth potential with moderate risk',
                    'allocation_suggestion': 'Consider overweighting these sectors'
                })
            
            # Low risk, stable returns
            stable_sectors = vol[vol < vol.quantile(0.25)]
            if len(stable_sectors) > 0:
                recommendations.append({
                    'type': 'Stability Play',
                    'sectors': list(stable_sectors.index),
                    'reasoning': 'Low volatility provides portfolio stability',
                    'allocation_suggestion': 'Good for risk-averse investors'
                })
        
        # Sharpe ratio recommendations
        if 'sharpe_ratio' in results:
            sharpe = results['sharpe_ratio']
            best_sharpe_sectors = sharpe[sharpe > sharpe.quantile(0.75)]
            
            if len(best_sharpe_sectors) > 0:
                recommendations.append({
                    'type': 'Risk-Adjusted Returns',
                    'sectors': list(best_sharpe_sectors.index),
                    'reasoning': 'Best risk-adjusted performance',
                    'allocation_suggestion': 'Core portfolio holdings'
                })
        
        # Diversification recommendations
        if 'correlation_matrix' in results:
            corr_matrix = results['correlation_matrix']
            
            # Find least correlated sectors
            avg_correlations = corr_matrix.mean()
            least_correlated = avg_correlations.nsmallest(2)
            
            if len(least_correlated) >= 2:
                recommendations.append({
                    'type': 'Diversification',
                    'sectors': list(least_correlated.index),
                    'reasoning': 'Low correlation provides diversification benefits',
                    'allocation_suggestion': 'Combine for portfolio diversification'
                })
        
        # Portfolio allocation recommendations
        if 'investment_distribution' in results:
            current_dist = results['investment_distribution']
            
            # Identify over/under-weighted sectors
            equal_weight = 1.0 / len(current_dist)
            over_weighted = {k: v for k, v in current_dist.items() if v > equal_weight * 1.5}
            under_weighted = {k: v for k, v in current_dist.items() if v < equal_weight * 0.5}
            
            if over_weighted:
                recommendations.append({
                    'type': 'Rebalancing',
                    'sectors': list(over_weighted.keys()),
                    'reasoning': 'Currently over-weighted relative to equal allocation',
                    'allocation_suggestion': 'Consider reducing allocation'
                })
            
            if under_weighted:
                recommendations.append({
                    'type': 'Rebalancing',
                    'sectors': list(under_weighted.keys()),
                    'reasoning': 'Currently under-weighted relative to equal allocation',
                    'allocation_suggestion': 'Consider increasing allocation'
                })
        
        self.recommendations = recommendations
        return recommendations
    
    def create_executive_summary(self, results: Dict) -> str:
        """
        Create an executive summary of the analysis.
        
        Args:
            results: Dictionary with calculation results
        
        Returns:
            Executive summary string
        """
        print("Creating executive summary...")
        
        summary_parts = []
        
        # Header
        summary_parts.append("STOCK MARKET SECTOR PERFORMANCE ANALYSIS")
        summary_parts.append("=" * 50)
        summary_parts.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_parts.append("")
        
        # Performance Overview
        if 'cagr' in results:
            cagr = results['cagr']
            best_sector = cagr.idxmax()
            worst_sector = cagr.idxmin()
            
            summary_parts.append("PERFORMANCE OVERVIEW:")
            summary_parts.append(f"• Best performing sector: {best_sector} ({cagr[best_sector]:.1%} CAGR)")
            summary_parts.append(f"• Worst performing sector: {worst_sector} ({cagr[worst_sector]:.1%} CAGR)")
            summary_parts.append(f"• Average sector performance: {cagr.mean():.1%} CAGR")
            summary_parts.append("")
        
        # Risk Overview
        if 'volatility' in results:
            vol = results['volatility']
            most_volatile = vol.idxmax()
            least_volatile = vol.idxmin()
            
            summary_parts.append("RISK OVERVIEW:")
            summary_parts.append(f"• Most volatile sector: {most_volatile} ({vol[most_volatile]:.2%} daily volatility)")
            summary_parts.append(f"• Least volatile sector: {least_volatile} ({vol[least_volatile]:.2%} daily volatility)")
            summary_parts.append(f"• Average sector volatility: {vol.mean():.2%} daily")
            summary_parts.append("")
        
        # Key Insights
        if hasattr(self, 'insights') and 'performance_analysis' in self.insights:
            insights = self.insights['performance_analysis']['key_insights']
            summary_parts.append("KEY INSIGHTS:")
            for insight in insights[:5]:  # Top 5 insights
                summary_parts.append(f"• {insight}")
            summary_parts.append("")
        
        # Top Recommendations
        if self.recommendations:
            summary_parts.append("TOP RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations[:3], 1):  # Top 3 recommendations
                summary_parts.append(f"{i}. {rec['type']}: {', '.join(rec['sectors'])}")
                summary_parts.append(f"   {rec['reasoning']}")
            summary_parts.append("")
        
        # Risk Assessment
        if hasattr(self, 'risk_assessment') and 'diversification_analysis' in self.risk_assessment:
            div_analysis = self.risk_assessment['diversification_analysis']
            summary_parts.append("DIVERSIFICATION ASSESSMENT:")
            summary_parts.append(f"• Average correlation: {div_analysis['average_correlation']:.2f}")
            summary_parts.append(f"• Diversification benefit: {div_analysis['diversification_benefit']}")
            summary_parts.append("")
        
        # Conclusion
        summary_parts.append("CONCLUSION:")
        if 'cagr' in results and 'volatility' in results:
            cagr = results['cagr']
            vol = results['volatility']
            
            if cagr.mean() > 0.1:  # 10% threshold
                summary_parts.append("• Overall sector performance is strong with positive returns across all sectors")
            else:
                summary_parts.append("• Sector performance is moderate with mixed results")
            
            if vol.mean() < 0.15:  # 15% volatility threshold
                summary_parts.append("• Risk levels are manageable with moderate volatility")
            else:
                summary_parts.append("• Risk levels are elevated, requiring careful portfolio management")
        
        summary_parts.append("• Consider diversification strategies to optimize risk-adjusted returns")
        summary_parts.append("• Regular rebalancing recommended to maintain target allocations")
        
        return "\n".join(summary_parts)
    
    def generate_detailed_report(self, results: Dict) -> str:
        """
        Generate a detailed analysis report.
        
        Args:
            results: Dictionary with calculation results
        
        Returns:
            Detailed report string
        """
        print("Generating detailed report...")
        
        report_parts = []
        
        # Executive Summary
        report_parts.append(self.create_executive_summary(results))
        report_parts.append("\n" + "="*80 + "\n")
        
        # Detailed Performance Analysis
        report_parts.append("DETAILED PERFORMANCE ANALYSIS")
        report_parts.append("="*40)
        
        if 'cagr' in results:
            cagr = results['cagr']
            report_parts.append("\nCAGR (Compound Annual Growth Rate):")
            for sector, value in cagr.sort_values(ascending=False).items():
                report_parts.append(f"  {sector}: {value:.2%}")
        
        if 'volatility' in results:
            vol = results['volatility']
            report_parts.append("\nVolatility (Daily Standard Deviation):")
            for sector, value in vol.sort_values(ascending=True).items():
                report_parts.append(f"  {sector}: {value:.2%}")
        
        if 'sharpe_ratio' in results:
            sharpe = results['sharpe_ratio']
            report_parts.append("\nSharpe Ratio (Risk-Adjusted Returns):")
            for sector, value in sharpe.sort_values(ascending=False).items():
                report_parts.append(f"  {sector}: {value:.2f}")
        
        if 'max_drawdown' in results:
            max_dd = results['max_drawdown']
            report_parts.append("\nMaximum Drawdown:")
            for sector, value in max_dd.sort_values(ascending=True).items():
                report_parts.append(f"  {sector}: {value:.2%}")
        
        # Investment Distribution
        if 'investment_distribution' in results:
            dist = results['investment_distribution']
            report_parts.append("\nCurrent Investment Distribution:")
            for sector, weight in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                report_parts.append(f"  {sector}: {weight:.1%}")
        
        # Risk Assessment
        if hasattr(self, 'risk_assessment'):
            report_parts.append("\n" + "="*40)
            report_parts.append("RISK ASSESSMENT")
            report_parts.append("="*40)
            
            if 'risk_levels' in self.risk_assessment:
                report_parts.append("\nRisk Levels by Sector:")
                for sector, risk_info in self.risk_assessment['risk_levels'].items():
                    report_parts.append(f"  {sector}: {risk_info['risk_level']} ({risk_info['volatility']:.2%} volatility)")
            
            if 'risk_insights' in self.risk_assessment:
                report_parts.append("\nRisk Insights:")
                for insight in self.risk_assessment['risk_insights']:
                    report_parts.append(f"  • {insight}")
        
        # Recommendations
        if self.recommendations:
            report_parts.append("\n" + "="*40)
            report_parts.append("INVESTMENT RECOMMENDATIONS")
            report_parts.append("="*40)
            
            for i, rec in enumerate(self.recommendations, 1):
                report_parts.append(f"\n{i}. {rec['type']}")
                report_parts.append(f"   Sectors: {', '.join(rec['sectors'])}")
                report_parts.append(f"   Reasoning: {rec['reasoning']}")
                report_parts.append(f"   Suggestion: {rec['allocation_suggestion']}")
        
        # Portfolio Metrics
        if 'portfolio_metrics' in results:
            report_parts.append("\n" + "="*40)
            report_parts.append("PORTFOLIO METRICS")
            report_parts.append("="*40)
            
            portfolio = results['portfolio_metrics']
            for metric, value in portfolio.items():
                if 'return' in metric or 'volatility' in metric or 'drawdown' in metric:
                    report_parts.append(f"  {metric}: {value:.2%}")
                else:
                    report_parts.append(f"  {metric}: {value:.2f}")
        
        return "\n".join(report_parts)
    
    def save_report(self, report: str, filename: str = None) -> str:
        """
        Save report to file.
        
        Args:
            report: Report content
            filename: Optional filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sector_analysis_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {filename}")
        return filename
    
    def run_complete_analysis(self, results: Dict) -> Dict:
        """
        Run complete analysis and generate all insights.
        
        Args:
            results: Dictionary with calculation results
        
        Returns:
            Dictionary with all analysis results
        """
        print("Running complete insights analysis...")
        
        # Run all analyses
        performance_analysis = self.analyze_performance(results)
        risk_assessment = self.assess_risk(results)
        recommendations = self.generate_investment_recommendations(results)
        
        # Generate reports
        executive_summary = self.create_executive_summary(results)
        detailed_report = self.generate_detailed_report(results)
        
        # Compile results
        analysis_results = {
            'performance_analysis': performance_analysis,
            'risk_assessment': risk_assessment,
            'recommendations': recommendations,
            'executive_summary': executive_summary,
            'detailed_report': detailed_report
        }
        
        # Save reports
        self.save_report(executive_summary, "executive_summary.txt")
        self.save_report(detailed_report, "detailed_analysis_report.txt")
        
        print("Complete analysis finished!")
        return analysis_results


def main():
    """Example usage of the InsightsGenerator class."""
    # This would typically be used with results from calculations.py
    generator = InsightsGenerator()
    
    # Create sample results
    sectors = ['IT', 'Pharma', 'Banking', 'FMCG']
    
    sample_results = {
        'cagr': pd.Series([0.25, 0.15, 0.08, 0.12], index=sectors),
        'volatility': pd.Series([0.20, 0.12, 0.15, 0.10], index=sectors),
        'sharpe_ratio': pd.Series([1.25, 1.25, 0.53, 1.20], index=sectors),
        'max_drawdown': pd.Series([-0.15, -0.08, -0.12, -0.06], index=sectors),
        'investment_distribution': {sector: 0.25 for sector in sectors},
        'correlation_matrix': pd.DataFrame(
            np.random.uniform(0.3, 0.8, (4, 4)),
            index=sectors, columns=sectors
        )
    }
    
    # Run complete analysis
    analysis_results = generator.run_complete_analysis(sample_results)
    
    # Print executive summary
    print("\n" + analysis_results['executive_summary'])


if __name__ == "__main__":
    main()
