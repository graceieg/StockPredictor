#!/usr/bin/env python
# S&P 500 Market Predictor
#
# A machine learning tool that predicts the daily direction of the S&P 500 index
# using historical data and random forest classification.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import yfinance as yf
import argparse
import os
from datetime import datetime


def get_data(start_date="1990-01-01", end_date=None):
    """
    Fetch S&P 500 historical data from Yahoo Finance

    Args:
        start_date (str): Start date for data fetch in YYYY-MM-DD format
        end_date (str): End date for data fetch in YYYY-MM-DD format

    Returns:
        pandas.DataFrame: S&P 500 historical data
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"Fetching S&P 500 data from {start_date} to {end_date}...")
    sp500 = yf.Ticker("^GSPC").history(period="max")

    # Filter by date range
    sp500 = sp500.loc[start_date:end_date].copy()

    # Drop dividends and stock splits
    if "Dividends" in sp500.columns:
        sp500 = sp500.drop("Dividends", axis=1)
    if "Stock Splits" in sp500.columns:
        sp500 = sp500.drop("Stock Splits", axis=1)

    print(f"Data fetched: {sp500.shape[0]} trading days")
    return sp500


def prepare_data(data):
    """
    Prepare data for prediction by adding target and predictive features

    Args:
        data (pandas.DataFrame): S&P 500 historical data

    Returns:
        pandas.DataFrame: Processed data with prediction features
    """
    # Create 'Tomorrow' column (next day's closing price)
    data["Tomorrow"] = data["Close"].shift(-1)

    # Create target: 1 if tomorrow's price is higher than today's
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

    # Create prediction features based on various time horizons
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []

    for horizon in horizons:
        # Calculate rolling averages
        rolling_averages = data.rolling(horizon).mean()

        # Price ratio compared to moving average
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]

        # Past target trend (how many up days)
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_column, trend_column]

    # Drop rows with NaN values
    data = data.dropna()

    print(f"Data prepared: {data.shape[0]} rows available for prediction")
    print(f"Created features: {new_predictors}")

    return data, new_predictors


def train_model(data, predictors, test_size=100):
    """
    Train a random forest model on the data

    Args:
        data (pandas.DataFrame): Prepared S&P 500 data
        predictors (list): List of predictor column names
        test_size (int): Number of days to use for testing

    Returns:
        tuple: Trained model, train data, test data
    """
    # Split data into training and test sets
    train = data.iloc[:-test_size].copy()
    test = data.iloc[-test_size:].copy()

    print(f"Training on {train.shape[0]} days, testing on {test.shape[0]} days")

    # Create and train model
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=50,
        random_state=1
    )

    model.fit(train[predictors], train["Target"])

    return model, train, test


def evaluate_model(model, test_data, predictors, threshold=0.6):
    """
    Evaluate model performance on test data

    Args:
        model (RandomForestClassifier): Trained model
        test_data (pandas.DataFrame): Test data
        predictors (list): List of predictor column names
        threshold (float): Probability threshold for predictions

    Returns:
        pandas.DataFrame: Test results with predictions
    """
    # Get probability predictions
    pred_proba = model.predict_proba(test_data[predictors])[:, 1]

    # Apply threshold
    predictions = (pred_proba >= threshold).astype(int)

    # Create predictions series
    preds = pd.Series(predictions, index=test_data.index, name="Predictions")

    # Combine with actual targets
    combined = pd.concat([test_data["Target"], preds], axis=1)

    # Calculate precision
    precision = precision_score(combined["Target"], combined["Predictions"])

    print(f"Prediction distribution: {combined['Predictions'].value_counts().to_dict()}")
    print(f"Actual distribution: {combined['Target'].value_counts().to_dict()}")
    print(f"Precision score: {precision:.4f}")

    return combined


def backtest(data, model, predictors, start=2500, step=250, threshold=0.6):
    """
    Perform backtesting of the model on historical data

    Args:
        data (pandas.DataFrame): Prepared S&P 500 data
        model (RandomForestClassifier): Model to use
        predictors (list): List of predictor column names
        start (int): Index to start backtesting from
        step (int): Step size for backtesting
        threshold (float): Probability threshold for predictions

    Returns:
        pandas.DataFrame: Combined backtesting results
    """
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()

        model.fit(train[predictors], train["Target"])

        pred_proba = model.predict_proba(test[predictors])[:, 1]
        predictions = (pred_proba >= threshold).astype(int)
        preds = pd.Series(predictions, index=test.index, name="Predictions")

        combined = pd.concat([test["Target"], preds], axis=1)
        all_predictions.append(combined)

        print(
            f"Backtest period {i} to {i + step}: Precision = {precision_score(combined['Target'], combined['Predictions']):.4f}")

    results = pd.concat(all_predictions)

    overall_precision = precision_score(results["Target"], results["Predictions"])
    print(f"\nOverall backtest precision: {overall_precision:.4f}")

    return results


def plot_results(results, title="S&P 500 Prediction Results"):
    """
    Plot the prediction results

    Args:
        results (pandas.DataFrame): Prediction results
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["Target"], label="Actual", alpha=0.7)
    plt.plot(results.index, results["Predictions"], label="Predicted", alpha=0.7)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Direction (1 = Up, 0 = Down)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    os.makedirs("output", exist_ok=True)
    filename = f"output/prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

    plt.show()


def feature_importance(model, predictors):
    """
    Show feature importance from the model

    Args:
        model (RandomForestClassifier): Trained model
        predictors (list): List of predictor column names
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [predictors[i] for i in indices], rotation=90)
    plt.tight_layout()

    # Save the plot
    os.makedirs("output", exist_ok=True)
    filename = f"output/feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    print(f"Feature importance plot saved to {filename}")

    plt.show()

    # Print feature importance
    print("\nFeature Importance:")
    for i in indices:
        print(f"{predictors[i]}: {importances[i]:.4f}")


def predict_next_day(model, data, predictors, threshold=0.6):
    """
    Predict the market direction for the next trading day

    Args:
        model (RandomForestClassifier): Trained model
        data (pandas.DataFrame): Prepared S&P 500 data
        predictors (list): List of predictor column names
        threshold (float): Probability threshold for predictions

    Returns:
        tuple: (prediction, probability)
    """
    # Get the most recent data point
    last_data = data.iloc[-1:].copy()

    # Get prediction probability
    prob = model.predict_proba(last_data[predictors])[0, 1]
    prediction = 1 if prob >= threshold else 0

    print(f"\nPrediction for next trading day after {last_data.index[0].date()}:")
    print(f"Probability of market going up: {prob:.2%}")
    print(f"Prediction: {'UP' if prediction == 1 else 'DOWN'} (threshold: {threshold})")

    return prediction, prob


def main():
    parser = argparse.ArgumentParser(description="S&P 500 Market Direction Predictor")
    parser.add_argument("--start", default="1990-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--test-size", type=int, default=100, help="Test set size in days")
    parser.add_argument("--threshold", type=float, default=0.6, help="Prediction threshold")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting")
    parser.add_argument("--backtest-start", type=int, default=2500, help="Backtest start index")
    parser.add_argument("--backtest-step", type=int, default=250, help="Backtest step size")
    args = parser.parse_args()

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Get and prepare data
    data = get_data(args.start, args.end)
    processed_data, predictors = prepare_data(data)

    # Train model
    model, train, test = train_model(processed_data, predictors, args.test_size)

    # Evaluate on test set
    results = evaluate_model(model, test, predictors, args.threshold)

    # Plot test results
    plot_results(results, "S&P 500 Prediction Test Results")

    # Show feature importance
    feature_importance(model, predictors)

    # Run backtesting if requested
    if args.backtest:
        print("\nRunning backtesting...")
        backtest_results = backtest(
            processed_data,
            model,
            predictors,
            args.backtest_start,
            args.backtest_step,
            args.threshold
        )
        plot_results(backtest_results, "S&P 500 Backtesting Results")

    # Predict next trading day
    predict_next_day(model, processed_data, predictors, args.threshold)

    # Save model and data
    import joblib
    model_filename = f"output/sp500_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

    # Save last week of data for reference
    data_filename = f"output/sp500_last_week_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    processed_data.iloc[-5:].to_csv(data_filename)
    print(f"Last week of data saved to {data_filename}")


if __name__ == "__main__":
    main()