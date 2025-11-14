# MongoDB Setup Guide for Stock Market Analytics

This guide will help you set up MongoDB for your stock market analytics project and migrate from CSV-based storage to MongoDB.

## Prerequisites

1. **MongoDB Atlas Account** (Recommended for cloud storage)
   - Sign up at [MongoDB Atlas](https://www.mongodb.com/atlas)
   - Create a free cluster
   - Get your connection string

2. **Local MongoDB** (Alternative)
   - Install MongoDB Community Edition
   - Start MongoDB service

## Setup Steps

### 1. Environment Configuration

Create a `.env` file in your project root:

```bash
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DB=sectorscope

# Optional: Cache settings
CACHE_TTL_MIN=30
```

### 2. Install Dependencies

The MongoDB integration requires additional dependencies:

```bash
pip install pymongo python-dotenv
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Database Schema

The project uses the following MongoDB collections:

- **sectors**: Master data for market sectors
- **stocks**: Individual stock information
- **stock_prices**: Daily stock price data
- **sector_etf_prices**: Sector ETF price data
- **daily_returns**: Calculated daily returns
- **sector_metrics**: Calculated sector performance metrics
- **portfolio_analyses**: Portfolio analysis results
- **correlation_matrix**: Sector correlation data
- **data_quality_logs**: Data quality and processing status
- **user_preferences**: User-specific settings
- **alerts**: User alerts and notifications

### 4. Migration from CSV to MongoDB

If you have existing CSV data, migrate it to MongoDB:

```bash
python scripts/migrate_csv_to_mongodb.py
```

This will:
- Migrate sector master data
- Convert CSV price data to MongoDB format
- Preserve all existing data
- Create proper indexes for performance

### 5. Seed Initial Data

Populate the database with initial sector and stock data:

```bash
python scripts/seed_mongo.py
```

### 6. Run the Application

Start the application with MongoDB support:

```bash
python main.py
```

## MongoDB Collections Schema

### Sectors Collection
```javascript
{
  "code": "IT",
  "name": "Information Technology",
  "etf_symbol": "XLK",
  "is_active": true,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

### Sector ETF Prices Collection
```javascript
{
  "sector_code": "IT",
  "trade_date": ISODate("2024-09-13"),
  "open_price": 218.50,
  "high_price": 220.00,
  "low_price": 217.00,
  "close_price": 219.25,
  "adjusted_close": 219.25,
  "volume": 50000000,
  "created_at": ISODate
}
```

### Daily Returns Collection
```javascript
{
  "sector_code": "IT",
  "trade_date": ISODate("2024-09-13"),
  "daily_return": 0.0034,
  "cumulative_return": 0.1250,
  "return_type": "sector",
  "created_at": ISODate
}
```

### Sector Metrics Collection
```javascript
{
  "sector_code": "IT",
  "calculation_date": ISODate("2024-09-13"),
  "period_type": "daily",
  "cagr": 0.1250,
  "volatility": 0.0250,
  "sharpe_ratio": 1.25,
  "max_drawdown": -0.0850,
  "avg_daily_return": 0.0008,
  "created_at": ISODate
}
```

## Performance Optimization

### Indexes
The application automatically creates the following indexes for optimal performance:

- `sectors.code` (unique)
- `stocks.symbol` (unique)
- `stock_prices.symbol + trade_date` (compound, unique)
- `sector_etf_prices.sector_code + trade_date` (compound, unique)
- `daily_returns.sector_code + trade_date` (compound, unique)
- `sector_metrics.calculation_date`
- `portfolio_analyses.start_date + end_date`

### Query Optimization
- Use date ranges for time-series queries
- Filter by sector when possible
- Leverage compound indexes for complex queries

## Data Flow

1. **Data Collection**: Yahoo Finance data → MongoDB
2. **Data Preprocessing**: MongoDB → Cleaned data → MongoDB
3. **Calculations**: MongoDB → Financial metrics → MongoDB
4. **Visualization**: MongoDB → Charts and reports
5. **Web Interface**: MongoDB → Real-time data display

## Troubleshooting

### Connection Issues
- Verify MONGODB_URI is correct
- Check network connectivity
- Ensure MongoDB service is running

### Data Issues
- Check data quality logs in MongoDB
- Verify CSV migration completed successfully
- Run data validation checks

### Performance Issues
- Check index usage with MongoDB Compass
- Monitor query performance
- Consider data archiving for old records

## Benefits of MongoDB Integration

1. **Scalability**: Handle large volumes of historical data
2. **Flexibility**: Easy schema evolution and new features
3. **Performance**: Optimized indexes for common queries
4. **Real-time**: Fast updates and live data access
5. **Analytics**: Rich aggregation pipelines for complex analysis
6. **Reliability**: ACID transactions and data consistency
7. **Cloud-ready**: Easy deployment to MongoDB Atlas

## Migration Checklist

- [ ] Set up MongoDB Atlas account or local MongoDB
- [ ] Configure environment variables
- [ ] Install required dependencies
- [ ] Run migration script for existing CSV data
- [ ] Seed initial data
- [ ] Test application with MongoDB
- [ ] Verify data integrity
- [ ] Monitor performance

## Support

For issues or questions:
1. Check the error logs in MongoDB
2. Verify environment configuration
3. Test MongoDB connection independently
4. Review data quality logs in the database

The MongoDB integration provides a robust, scalable foundation for your stock market analytics application.
