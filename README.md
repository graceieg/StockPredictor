# S&P 500 Market Predictor

A machine learning tool that predicts the daily direction of the S&P 500 index using historical data and random forest classification.

## Overview

This program uses machine learning to predict whether the S&P 500 index will go up or down on the next trading day. It:

1. Fetches historical S&P 500 data from Yahoo Finance
2. Creates predictive features based on price patterns and trends
3. Trains a Random Forest Classifier model
4. Evaluates the model's performance on recent data
5. Provides a prediction for the next trading day

## Features

- **Data Preparation**: Creates technical features based on various time horizons (2, 5, 60, 250, and 1000 days)
- **Model Training**: Uses Random Forest with optimized parameters
- **Backtesting**: Tests model performance across historical data
- **Visualization**: Creates plots of predictions vs actual outcomes
- **Feature Importance**: Shows which factors most influence predictions
- **Next-Day Prediction**: Forecasts whether the market will go up or down on the next trading day

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sp500-predictor.git
   cd sp500-predictor
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the program with default settings:

```bash
python sp500_predictor.py
```

This will:
- Fetch S&P 500 data from 1990 to present
- Train the model on all but the last 100 days
- Test on the most recent 100 days
- Show prediction accuracy
- Make a prediction for the next trading day

### Advanced Options

```bash
python sp500_predictor.py --start 2010-01-01 --test-size 250 --threshold 0.65 --backtest
```

Command-line options:
- `--start`: Start date for data (YYYY-MM-DD)
- `--end`: End date for data (defaults to today)
- `--test-size`: Number of recent days to use for testing
- `--threshold`: Probability threshold for predictions (default 0.6)
- `--backtest`: Run backtesting on historical data
- `--backtest-start`: Sample index to start backtesting from
- `--backtest-step`: Step size for backtesting

## Interpretation

The model outputs:
- Prediction (UP or DOWN) for the next trading day
- Probability of the market going up
- Precision score (percentage of "UP" predictions that were correct)

## Output

All results are saved to the `output` directory:
- Prediction charts
- Feature importance analysis
- Trained model file
- Recent data

## Important Notes

- This tool is for educational purposes only and should not be used for actual investment decisions
- Past performance does not guarantee future results
- The stock market is influenced by many factors not captured in this model
- Always consult with a qualified financial advisor before making investment decisions

## License

This project is licensed under the MIT License - see the LICENSE file for details.