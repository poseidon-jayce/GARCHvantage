<img width="646" height="369" alt="image" src="https://github.com/user-attachments/assets/89107cbd-ed4d-4475-87b6-901603aae731" /># GARCHvantage
An automated quantitative asset management platform integrating Oracle 21c ledgers with an EGARCH volatility pipeline and Black-Litterman portfolio optimization. Tracks systematic factor drift against the NIFTY 50 and generates actionable FIX/CSV rebalancing order batches for instant multi-broker execution.

GARCHvantage: Advanced Volatility Analytics & Capital AllocationGARCHvantage is an enterprise-grade quantitative finance platform designed to model, forecast, and exploit asset volatility.
By implementing generalized autoregressive conditional heteroskedasticity frameworks, the platform translates raw historical market data into actionable risk metrics and optimized capital allocation strategies.

## Live Deployment: Hugging Face Space : https://huggingface.co/spaces/krishna-0722/equity-analytics
## Source Code: GitHub Repository

## System Architecture & File Structure:
The project is structured modularly to separate data ingestion, mathematical modeling, and user interface layers:textGARCHvantage/

├── app.py                  # Core Streamlit application & UI layout
├── requirements.txt        # Production dependencies (arch, scipy, yfinance)
├── README.md               # System documentation
├── core/
│   ├── __init__.py
│   ├── data_loader.py      # Automated market data ingestion and cleaning
│   ├── models.py            # GARCH, EGARCH, GJR-GARCH engine wrappers
│   └── risk_engine.py       # VaR calculation and capital allocation logic
└── assets/
    └── framework_nav.png   # Architecture visualization assets
    
## Core Modeling Engine Explained: 

The platform moves beyond static variance assumptions by fitting three foundational volatility equations to asset returns:Standard GARCH(1,1): Captures symmetric volatility clustering. 
It assumes positive and negative shocks of equal magnitude impact future variance identically.
- EGARCH (Exponential GARCH): Models asymmetric leverage effects logarithmically. It ensures variance remains positive without imposing strict parameter constraints.GJR-GARCH: Integrates an indicator function to specifically capture the "bad news" phenomenon, where negative market shocks trigger higher volatility than positive ones.
- Key Benefits & Market ImportanceTraditional risk models often treat risk as a constant, leaving portfolios exposed during black swan events or market regimes shifts.
- Dynamic Capital Sizing: Reallocates capital away from assets experiencing volatility spikes before losses compound.
- Asymmetric Risk Pricing: Quantifies the "leverage effect," allowing traders to price downside protection accurately.
- Precision Risk Calibration: Computes Value-at-Risk (VaR) using time-varying conditional variance rather than historic averages.

## Methodology Comparison 
<img width="646" height="369" alt="image" src="https://github.com/user-attachments/assets/2b034ba6-0043-4738-b86c-35cffc38c4c2" />

## Practical Use Cases & Target AudienceWho Uses It?
- Quantitative Researchers: To backtest volatility trading strategies and alpha signals.
- Risk Managers: To monitor live portfolio VaR violations and adjust stress-test parameters.
- Portfolio Managers: To execute volatility-targeted asset allocation.

## Real-World Workflow Ingest: 
- Pull live ticker data via the interface.
- Fit: Run the asymmetric engines to identify if recent drops have structurally broken the volatility regime.
- Deploy: Use the calculated conditional variance to downsize position limits prior to market opens.

# Future Scope & RoadmapTo evolve GARCHvantage into a institutional portfolio utility

The following technical integrations are planned:
- Multivariate GARCH (DCC-GARCH): Model time-varying correlations between multiple assets simultaneously for true portfolio optimization.
- Alternative Distributions: Implement Student's t and Skewed Student's t distributions to better capture market fat-tails.
- Machine Learning Hybrids: Build LSTM-GARCH pipelines using TensorFlow to capture non-linear residual variance.
- High-Frequency Data Pipelines: Integrate WebSockets via Polygon.io for intraday volatility forecasting.
