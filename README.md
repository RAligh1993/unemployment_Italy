# 🇮🇹 Italian Unemployment Nowcasting System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Professional Streamlit application for real-time unemployment nowcasting using Google Trends and advanced econometric models**

Built by **Rajabali Ghasempour** at **ISTAT** (Italian National Institute of Statistics)

---

## 📊 Overview

This application provides **real-time unemployment nowcasts** for Italy, combining:
- Official ISTAT unemployment data (monthly)
- Google Trends search data (weekly)
- Exogenous economic indicators (CCI, HICP)
- Multiple econometric and ML models

### Key Features

✅ **Multi-Model Framework**: MIDAS, Ridge, Lasso, Random Forest, XGBoost, LSTM  
✅ **Google Trends Integration**: Automatic 5-segment merging with quality checks  
✅ **Statistical Testing**: Clark-West, Diebold-Mariano tests  
✅ **Interactive Visualizations**: Beautiful Plotly charts  
✅ **Real-Time Nowcasting**: Live predictions with confidence intervals  
✅ **Early Warning System**: GT signal monitoring and alerts  

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/unemployment-nowcasting.git
cd unemployment-nowcasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure
```
streamlit_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── backend/
│   ├── __init__.py
│   ├── data_loader.py             # Data loading & preprocessing
│   ├── feature_engineering.py     # Feature creation
│   ├── models.py                  # All models (MIDAS, ML, etc.)
│   ├── evaluation.py              # Performance metrics & tests
│   └── forecaster.py              # Real-time forecasting
├── utils/
│   ├── __init__.py
│   ├── visualizations.py          # Plotly charts
│   └── helpers.py                 # Utility functions
└── data/
    └── .gitkeep                   # Placeholder
```

---

## 📖 User Guide

### Step 1: Upload Data

**Required:**
- **Unemployment CSV**: Must contain `date` and `unemp` columns

**Optional:**
- **Google Trends Excel**: Multiple 5-year segments (handled automatically)
- **Exogenous Variables CSV**: CCI, HICP, etc.

### Step 2: Configure Settings

- **Operating Mode**: Default (pre-configured) or Custom
- **Train/Test Split**: Choose percentage (default: 70%)
- **Models**: Select which models to train
- **Google Trends**: Enable/disable GT features

### Step 3: Load & Process

Click **"🚀 Load & Process Data"** in sidebar:
- Validates data quality
- Merges Google Trends segments
- Creates features and lags
- Displays summary statistics

### Step 4: Train Models

Click **"🤖 Train Models"**:
- Trains selected models
- Computes performance metrics
- Runs statistical tests
- Generates comparison charts

### Step 5: Explore Results

Navigate through tabs:
- **📊 Overview**: System status and quick stats
- **📈 Data Explorer**: Time series, correlations, data quality
- **🤖 Models**: Training configuration
- **📉 Results**: Performance comparison, period analysis
- **🔮 Live Nowcast**: Real-time predictions and GT signals
- **📚 Documentation**: Complete user guide

---

## 🎯 Use Cases

### 1. Real-Time Monitoring
- Generate nowcasts 2-3 weeks before official releases
- Monitor GT search intensity for early warnings
- Track confidence intervals for uncertainty

### 2. Model Comparison
- Compare MIDAS vs ML approaches
- Test different GT aggregation schemes
- Evaluate statistical significance

### 3. Research & Analysis
- Experiment with feature engineering
- Test new model architectures
- Analyze period-wise performance

### 4. Operational Deployment
- Integrate into ISTAT workflows
- Automated weekly updates
- Alert system for significant changes

---

## 📊 Model Details

### MIDAS (Mixed Data Sampling)
- **Exponential Weights**: θ=3.0 (95% on most recent week)
- **Beta Polynomial**: θ₁=5, θ₂=1
- Aggregates weekly GT to monthly frequency

### Machine Learning
- **Ridge/Lasso**: Regularized linear regression
- **Random Forest**: Ensemble of 100 trees
- **XGBoost**: Gradient boosting (100 estimators)
- **LSTM**: Deep learning (experimental)

### Evaluation
- **Metrics**: RMSE, MAE, R², Direction Accuracy
- **Tests**: Clark-West (nested), Diebold-Mariano (general)
- **Validation**: Walk-forward backtesting

---

## 🔧 Configuration

### Streamlit Settings

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"  # Your brand color
backgroundColor = "#ffffff"

[server]
port = 8501
maxUploadSize = 200  # MB
```

### Model Hyperparameters

Edit `backend/models.py`:
```python
# MIDAS Exponential
theta = 3.0        # Decay parameter
n_lags = 4         # Weekly lags
alpha = 50.0       # Ridge regularization

# Random Forest
n_estimators = 100
max_depth = 5
```

---

## 📈 Performance

**Based on Italian data (2023-2025 test period):**

| Model | RMSE | Improvement | p-value |
|-------|------|-------------|---------|
| **MIDAS Exp(θ=3.0)** | **0.4915** | **+7.4%** | **0.031*** |
| MIDAS Beta | 0.4943 | +6.9% | 0.042* |
| Ridge | 0.5012 | +5.6% | 0.067 |
| Random Forest | 0.5089 | +4.1% | 0.089 |
| Baseline | 0.5309 | — | — |

*Significant at 5% level

---

## 🔒 Data Privacy

- All data processing happens **locally**
- No data sent to external servers
- Google Trends data is **aggregated and anonymized**
- User uploads stored temporarily in session only

---

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black app.py backend/ utils/
```

### Type Checking
```bash
mypy app.py --ignore-missing-imports
```

---

## 📚 References

### MIDAS Methodology
- Ghysels et al. (2004, 2007): Mixed frequency data sampling
- Andreou et al. (2013): MIDAS for macroeconomic forecasting
- Marcellino & Schumacher (2016): Nowcasting with MIDAS

### Google Trends Forecasting
- Choi & Varian (2012): Predicting the present with GT
- D'Amuri & Marcucci (2017): Italian unemployment forecasting
- Castle et al. (2021): Critical reassessment of GT

### Statistical Testing
- Clark & West (2007): Testing nested forecast accuracy
- Diebold & Mariano (1995): Comparing predictive accuracy

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Ali Ghanbari**
- Institution: ISTAT (Italian National Institute of Statistics)
- Email: [
rajabali.ghasempour@studenti.unicampania.it]
- GitHub: [@aligh219](https://github.com/aligh219)

---

## 🙏 Acknowledgments

- **ISTAT** for internship opportunity and data access
- **Anthropic Claude** for development assistance
- **Streamlit** team for excellent framework
- **Research Community** for MIDAS and GT methodologies

---

## 📞 Support

For questions or issues:
- 📧 Email: [
rajabali.ghasempour@studenti.unicampania.it]


---

## 🔄 Version History

### v1.0.0 (December 2025)
- ✅ Initial release
- ✅ Multi-model framework
- ✅ Google Trends integration
- ✅ Interactive visualizations
- ✅ Real-time nowcasting

---

**⭐ If you find this project useful, please give it a star on GitHub!**

---

*Last updated: December 2025*
