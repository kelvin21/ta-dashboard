# 📊 MACD Reversal Dashboard

A comprehensive technical analysis dashboard for Vietnamese stock market (HOSE/HNX) with multi-timeframe MACD analysis, market breadth indicators, and crossover signal predictions.

## ✨ Features

- **Multi-Timeframe MACD Analysis**
  - Daily, Weekly, Monthly MACD histogram tracking
  - 6-stage reversal detection (Troughing, Confirmed Trough, Rising, Peaking, Confirmed Peak, Falling)
  - Crossover prediction (up to 5 days ahead)
  - Color-coded trend stages

- **Intraday Data Updates**
  - Real-time OHLCV updates from TCBS API
  - Auto-adjustment for dividends/splits
  - Manual and automatic refresh options
  - Trading hours volume adjustment (9AM-11:30AM, 1PM-2:45PM)

- **Market Breadth Analysis** *(optional page)*
  - Advance/Decline indicators
  - Moving average analysis (MA50, MA200)
  - Market sentiment tracking

- **Market Forecast** *(optional page)*
  - Predictive models based on breadth data
  - Trend forecasting

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
streamlit >= 1.28.0
pandas >= 2.0.0
numpy >= 1.24.0
plotly >= 5.17.0
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kelvin21/ta-dashboard.git
cd ta-dashboard
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment (optional):**
```bash
cp .env.example .env
# Edit .env with your configurations
```

4. **Initialize database:**
```bash
python init_database.py
```

5. **Run the dashboard:**
```bash
streamlit run ta_dashboard.py
```

## 📁 Project Structure

```
macd-reversal/
├── ta_dashboard.py              # Main dashboard (MACD Overview)
├── pages/
│   ├── 1_📊_Market_Breadth.py
│   └── 2_🎯_Market_Forecast.py
├── build_price_db.py            # TCBS data fetcher
├── ticker_manager.py            # Ticker management
├── intraday_updater.py          # Intraday OHLCV updates
├── dividend_adjuster.py         # Dividend/split adjustments
├── db_adapter.py                # Database adapter (SQLite/MongoDB)
├── init_database.py             # Initialize empty database
├── export_database.py           # Export/import database to CSV
├── mongodb_migration.py         # Migrate SQLite to MongoDB
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ⚙️ Configuration

### Local Development (SQLite)

No configuration needed! Just run:
```bash
streamlit run ta_dashboard.py
```

### Cloud Deployment (MongoDB)

1. **Create MongoDB Atlas account** (free tier available)
   - Visit: https://www.mongodb.com/cloud/atlas/register
   - Create M0 Free cluster

2. **Get connection string:**
   - Click "Connect" → "Connect your application"
   - Copy the URI

3. **Set environment variables:**
```bash
export USE_MONGODB=true
export MONGODB_URI="mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
```

4. **Install MongoDB dependencies:**
```bash
pip install pymongo dnspython
```

5. **Migrate data:**
```bash
python mongodb_migration.py
```

6. **Deploy!**

## 📊 Usage

### MACD Analysis

1. **Overview Table**: View all tickers with multi-timeframe MACD stages
2. **Crossover Signals**: See predicted crossovers (↗/↘ indicators)
3. **Detailed Charts**: Click any ticker for candlestick + MACD subplots
4. **Volume Analysis**: Vol/AvgVol ratio with intraday adjustment

### Intraday Updates

- **Auto-refresh**: Runs once during market hours (9AM-5PM)
- **Manual refresh**: Use sidebar button for subsequent updates
- **Single ticker**: Update individual ticker
- **Bulk update**: Refresh all tickers at once

### Data Management

- **Add Ticker**: Use Admin panel in sidebar
- **Remove Ticker**: Select ticker and source to remove
- **Dividend Adjustment**: Scan and apply price adjustments
- **TCBS Historical**: Fetch historical data from TCBS

## 🎨 Color Coding

**MACD Stages:**
- 🟢 **Pale Green**: Stage 1 - Troughing
- 🟢 **Neon Green**: Stage 2 - Confirmed Trough
- 🟢 **Dark Green**: Stage 3 - Rising above Zero
- 🔴 **Pale Red**: Stage 4 - Peaking
- 🔴 **Bright Red**: Stage 5 - Confirmed Peak
- 🔴 **Dark Red**: Stage 6 - Falling below Zero

**Signals:**
- ↗0d: Just crossed up (today)
- ↘0d: Just crossed down (today)
- ↗Nd: Will cross up in N days
- ↘Nd: Will cross down in N days

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Add secrets (if using MongoDB):
   ```toml
   USE_MONGODB = "true"
   MONGODB_URI = "mongodb+srv://..."
   ```
5. Deploy!

### Local with SQLite

```bash
streamlit run ta_dashboard.py
```

### Production with MongoDB

```bash
USE_MONGODB=true \
MONGODB_URI="mongodb+srv://..." \
streamlit run ta_dashboard.py --server.port 8501
```

## 🔧 Development

### Hide/Show Pages

**Option 1: Environment variables**
```bash
export SHOW_MARKET_BREADTH_PAGE=false
export SHOW_MARKET_FORECAST_PAGE=false
```

**Option 2: Edit `.env` file**
```bash
SHOW_MARKET_BREADTH_PAGE=false
SHOW_MARKET_FORECAST_PAGE=false
```

### Database Export/Import

**Export to CSV:**
```bash
python export_database.py
```

**Import from CSV:**
```bash
python export_database.py import
```

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open pull request

## 📧 Contact

- GitHub: [@kelvin21](https://github.com/kelvin21)
- Repository: [ta-dashboard](https://github.com/kelvin21/ta-dashboard)

## 🙏 Acknowledgments

- TCBS for market data API
- Streamlit for the dashboard framework
- AmiBroker for MACD calculation reference
- MongoDB Atlas for cloud database hosting

---

**⚠️ Disclaimer**: This tool is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.
