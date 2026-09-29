# Macro-intelligence-system-v1
# Macro Intelligence Machine v1

A machine-learning research project that combines macroeconomic data,
market structure, regime detection, and supervised learning to study
future Bitcoin market direction.

## Objective

The goal of this project is to explore whether macroeconomic and
financial-market variables can provide useful information about the
probability of a positive 30-day Bitcoin return.

## Data Used

The project analyzes:

- Bitcoin (BTC)
- U.S. 10-Year Treasury Yield
- U.S. 2-Year Treasury proxy
- U.S. Dollar Index (DXY)
- VIX
- Crude Oil
- Gold
- Nasdaq
- S&P 500

## Machine Learning

### Random Forest
Used to classify whether the 30-day forward BTC return is positive
or negative.

### K-Means
Used to identify market regimes based on macroeconomic and volatility
features.

## Feature Engineering

The model uses:

- Macro percentage changes
- BTC 5-day returns
- BTC 20-day returns
- 30-day moving average
- 90-day moving average
- BTC volatility
- Price relative to moving averages
- U.S. 10Y–2Y yield spread

## Methodology

1. Download market data
2. Clean and align the data
3. Engineer features
4. Create a 30-day forward-return target
5. Split the data chronologically
6. Scale the training data
7. Detect market regimes using K-Means
8. Train a Random Forest model
9. Evaluate the model on unseen test data
10. Generate probability-based signals
11. Analyze performance and feature importance

## Signal Logic

Probability >= 0.65 → Bullish

Probability <= 0.35 → Bearish

Otherwise → Neutral

## Evaluation

The project evaluates:

- Classification accuracy
- Win rate
- Maximum drawdown
- Sharpe-like return ratio
- Feature importance
- Regime performance

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- yfinance
- Matplotlib

## Limitations

This is a research prototype and not a production trading system.

Future improvements include:

- Rolling walk-forward validation
- Transaction costs and slippage
- Better time-series cross-validation
- Additional macroeconomic variables
- Improved risk management
- Longer out-of-sample testing
- Paper/live testing

## Author

Prince Nwalozie

GitHub: https://github.com/Prince-code1
