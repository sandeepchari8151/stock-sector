# SectorScope – Stock Market Sector Performance Analytics

A comprehensive, modular Python application for analyzing stock market sector performance using real Yahoo Finance data.

## 🚀 Features

- **Real-time data collection** from Yahoo Finance
- **Advanced data preprocessing** with quality checks
- **Comprehensive financial calculations** (CAGR, volatility, Sharpe ratio, etc.)
- **Professional visualizations** (charts, heatmaps, dashboards)
- **AI-powered insights** and investment recommendations
- **Modular architecture** for easy maintenance and extension

## 📁 Project Structure

```
stock project/
├── main.py                     # Main application orchestrator
├── data_collection.py          # Yahoo Finance data download & loading
├── data_preprocessing.py       # Data cleaning, filtering, validation
├── calculations.py             # Financial metrics & analytics
├── visualization.py            # Charts, heatmaps, dashboards
├── insights_reporting.py       # Analysis, insights, reports
├── download_yahoo_data.py      # Standalone data downloader
├── data/                       # Data storage folder
│   ├── IT.csv
│   ├── Pharma.csv
│   ├── Banking.csv
│   └── FMCG.csv
└── README.md                   # This file
```

## 🛠️ Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install pandas matplotlib numpy yfinance seaborn
   ```

## 🎯 Quick Start

### Option 1: Run Complete Analysis (Recommended)
```bash
python main.py
```

### Option 2: Download Data Only
```bash
python scripts/download_yahoo_data.py
```

## 📊 What You Get

### 1. **Data Collection** (`data_collection.py`)
- Downloads 1-year historical data from Yahoo Finance
- Supports multiple sector ETFs (XLK, XLV, XLF, XLP)
- Handles data validation and error recovery
- Creates sample datasets for testing

### 2. **Data Preprocessing** (`data_preprocessing.py`)
- Validates data quality and structure
- Handles missing values and outliers
- Filters by date range and sectors
- Generates comprehensive quality reports

### 3. **Financial Calculations** (`calculations.py`)
- **Returns**: Daily returns, cumulative returns
- **Risk Metrics**: Volatility, maximum drawdown, Sharpe ratio
- **Performance**: CAGR, average returns
- **Portfolio**: Correlation matrix, portfolio metrics
- **Allocation**: Market cap, performance, equal weight methods

### 4. **Visualizations** (`visualization.py`)
- **Line Chart**: Sector performance over time
- **Bar Chart**: Investment distribution
- **Heatmap**: Volatility vs growth analysis
- **Scatter Plot**: Risk vs return analysis
- **Correlation Matrix**: Sector relationships
- **Dashboard**: Comprehensive overview

### 5. **Insights & Reporting** (`insights_reporting.py`)
- Performance analysis and ranking
- Risk assessment and categorization
- Investment recommendations
- Executive summary generation
- Detailed analysis reports

## 📈 Sample Results

### Performance Analysis
- **Best Sector**: IT (23.9% CAGR)
- **Most Stable**: FMCG (0.74% daily volatility)
- **Best Risk-Adjusted**: Banking (1.14 Sharpe ratio)

### Investment Distribution (Market Cap Based)
- **IT**: 49.8% (Technology dominance)
- **Pharma**: 25.5%
- **FMCG**: 14.8%
- **Banking**: 9.9%

### Key Insights
- IT sector shows highest growth but also highest volatility
- FMCG provides portfolio stability with low volatility
- Banking offers best risk-adjusted returns
- Diversification benefits are moderate (0.54 average correlation)

## ⚙️ Configuration

Edit `main.py` to customize:

```python
config = {
    'use_real_data': True,           # Use Yahoo Finance data
    'download_fresh_data': False,    # Download new data vs use existing
    'start_date': None,              # Filter start date
    'end_date': None,                # Filter end date
    'sectors': None,                 # Filter specific sectors
    'investment_method': 'market_cap', # Allocation method
    'generate_charts': True,         # Create visualizations
    'generate_reports': True,        # Generate insights
    'save_results': True             # Save outputs
}
```

## 📋 Generated Files

### Charts & Visualizations
- `sector_performance_chart.png` - Performance trends
- `investment_distribution_chart.png` - Portfolio allocation
- `volatility_growth_heatmap.png` - Risk vs growth
- `correlation_heatmap.png` - Sector relationships
- `risk_return_scatter.png` - Risk-return analysis
- `performance_metrics_chart.png` - Metrics comparison
- `sector_performance_dashboard.png` - Comprehensive dashboard

### Reports & Data
- `executive_summary.txt` - Key findings
- `detailed_analysis_report.txt` - Full analysis
- `calculation_results_[timestamp].json` - Raw results
- `processed_data_[timestamp].csv` - Cleaned data

## 🔧 Customization

### Add New Sectors
1. Update `YAHOO_SYMBOLS` in `data_collection.py`
2. Add sector to `INVESTMENT_DISTRIBUTION` in `main.py`
3. Run analysis

### Change Time Period
```python
config['start_date'] = '2024-01-01'
config['end_date'] = '2024-12-31'
```

### Modify Allocation Method
```python
config['investment_method'] = 'performance'  # or 'equal_weight'
```

### Add Custom Metrics
1. Add function to `calculations.py`
2. Call in `run_all_calculations()`
3. Update `visualization.py` if needed

## 🎓 Educational Value

This project demonstrates:
- **Modular software design** with separation of concerns
- **Financial data analysis** with real market data
- **Professional visualization** techniques
- **Data preprocessing** best practices
- **Statistical analysis** in finance
- **Report generation** and insights

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **matplotlib**: Basic plotting
- **numpy**: Numerical computations
- **yfinance**: Yahoo Finance data download
- **seaborn**: Statistical visualizations

## 🤝 Contributing

Feel free to:
- Add new financial metrics
- Improve visualizations
- Enhance data preprocessing
- Add new data sources
- Optimize performance

## 📄 License

This project is for educational purposes. Please respect Yahoo Finance's terms of service when downloading data.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Download Fails**: Check internet connection and Yahoo Finance availability
3. **Timezone Errors**: The system handles timezone-aware dates automatically
4. **Memory Issues**: Reduce date range or number of sectors

### Getting Help

- Check the console output for detailed error messages
- Verify data files exist in the `data/` folder
- Ensure all CSV files have the required columns: Date, Sector, Close

---

**Happy Analyzing! 📊📈**

## 🌐 MongoDB Integration (Required)

The application uses MongoDB for all data storage. Follow these steps to set up:

### Quick Setup

1. **Run the setup script:**
   ```bash
   python setup.py
   ```
   This will guide you through the complete setup process.

2. **Or manually create a `.env` file:**
   ```bash
   # MongoDB Configuration
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
   MONGODB_DB=sectorscope
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

### MongoDB Atlas Setup

1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a cluster
3. Get your connection string from the "Connect" button
4. Use the connection string in your `.env` file

### Database Management

- **View database status:** `python scripts/mongodb_dashboard.py`
