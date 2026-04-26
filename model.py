"""
Beginner-friendly service layer for the Stock Trend Prediction API.

This file reuses the existing machine learning logic from stock_trend_model.py
and wraps it in simple functions that the Flask app can call.
"""

import contextlib
import io
import math
import re

from stock_trend_model import (
    analyze_risk,
    fetch_stock_data,
    get_feature_columns,
    monte_carlo_simulation,
    prepare_dataset,
    print_feature_importance,
    select_features_by_importance,
    train_model,
)


# Very simple in-memory cache so we do not retrain the same ticker repeatedly.
# The cache lasts as long as the Flask app process is running.
MODEL_CACHE = {}


def validate_ticker(ticker):
    """
    Validate ticker input to keep the API predictable and beginner-friendly.
    """
    cleaned_ticker = (ticker or "").strip().upper()

    if not cleaned_ticker:
        raise ValueError("Ticker is required.")

    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,10}", cleaned_ticker):
        raise ValueError("Ticker contains invalid characters.")

    return cleaned_ticker


def run_silently(function, *args, **kwargs):
    """
    Run a helper function without printing extra training logs to the API console.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def is_finite_number(value):
    """
    Check whether a numeric value is safe to return in JSON.
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean_numeric_list(values):
    """
    Keep only finite numeric values so the frontend never receives NaN.
    """
    cleaned_values = []

    for value in values:
        if is_finite_number(value):
            cleaned_values.append(round(float(value), 2))

    return cleaned_values


def build_training_bundle(ticker, period="5y"):
    """
    Train a model for one ticker and keep the most useful features.
    """
    stock_data = fetch_stock_data(ticker=ticker, period=period)
    dataset = prepare_dataset(stock_data)

    if len(dataset) < 60:
        raise ValueError("Not enough historical data to train the model.")

    all_features = get_feature_columns()
    x_full = dataset[all_features]
    y_full = dataset["Trend"]

    # First training pass uses all features.
    base_model = train_model(x_full, y_full)
    importance_df = run_silently(print_feature_importance, base_model, all_features)

    # Remove the least useful features, then retrain once more.
    selected_features, removed_features = run_silently(
        select_features_by_importance,
        importance_df,
        min_features_to_keep=8,
        features_to_remove=4,
    )

    final_model = train_model(dataset[selected_features], y_full)

    return {
        "ticker": ticker,
        "stock_data": stock_data,
        "dataset": dataset,
        "model": final_model,
        "selected_features": selected_features,
        "removed_features": removed_features,
    }


def get_or_train_bundle(ticker, period="5y"):
    """
    Reuse a trained ticker model if it already exists in memory.
    """
    ticker = validate_ticker(ticker)

    if ticker not in MODEL_CACHE:
        MODEL_CACHE[ticker] = build_training_bundle(ticker, period=period)

    return MODEL_CACHE[ticker]


def predict_trend_for_ticker(ticker):
    """
    Return prediction data in a format that the frontend can consume easily.
    """
    bundle = get_or_train_bundle(ticker)
    model = bundle["model"]
    dataset = bundle["dataset"]
    selected_features = bundle["selected_features"]
    stock_data = bundle["stock_data"]

    latest_row = dataset.iloc[[-1]]
    latest_features = latest_row[selected_features]

    predicted_trend = model.predict(latest_features)[0]
    probabilities = model.predict_proba(latest_features)[0]
    probability_map = {
        label: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    confidence = max(probability_map.values())

    if not is_finite_number(confidence):
        raise ValueError("Model confidence could not be calculated for this ticker.")

    _, risk_level = run_silently(analyze_risk, stock_data)
    signal_strength = "Strong" if confidence >= 0.60 else "Weak"

    cleaned_probabilities = {}
    for label, probability in probability_map.items():
        cleaned_probabilities[label] = round(float(probability), 4) if is_finite_number(probability) else 0.0

    return {
        "ticker": bundle["ticker"],
        "trend": predicted_trend,
        "confidence": round(confidence * 100, 2),
        "risk": risk_level,
        "risk_level": risk_level,
        "signal": signal_strength,
        "signal_strength": signal_strength,
        "probabilities": cleaned_probabilities,
    }


def simulate_prices_for_ticker(ticker, simulations=100, days=30):
    """
    Return simple Monte Carlo simulation output for the requested ticker.
    """
    ticker = validate_ticker(ticker)
    stock_data = fetch_stock_data(ticker=ticker, period="5y")
    simulated_prices = run_silently(
        monte_carlo_simulation,
        stock_data,
        simulations=simulations,
        days=days,
    )
    cleaned_prices = clean_numeric_list(simulated_prices.tolist())

    if not cleaned_prices:
        raise ValueError("Monte Carlo simulation did not produce valid price values.")

    return {
        "ticker": ticker,
        "simulated_prices": cleaned_prices,
        "average": round(sum(cleaned_prices) / len(cleaned_prices), 2),
        "min": round(min(cleaned_prices), 2),
        "max": round(max(cleaned_prices), 2),
        "simulation_summary": {
            "average": round(sum(cleaned_prices) / len(cleaned_prices), 2),
            "min": round(min(cleaned_prices), 2),
            "max": round(max(cleaned_prices), 2),
        },
    }


def health_status():
    """
    Small helper for the /health endpoint.
    """
    return {
        "status": "ok",
        "message": "Stock Trend Prediction API is running.",
    }
