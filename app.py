"""
Simple Flask API for the Stock Trend Prediction web app.

This backend only exposes REST endpoints.
It does not render HTML pages.
"""

import math
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from model import health_status, predict_trend_for_ticker, simulate_prices_for_ticker


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=None)
CORS(app)


def make_json_safe(data):
    """
    Recursively replace NaN and Infinity values so the browser can parse JSON safely.
    """
    if isinstance(data, dict):
        return {key: make_json_safe(value) for key, value in data.items()}

    if isinstance(data, list):
        return [make_json_safe(item) for item in data]

    if isinstance(data, float):
        return data if math.isfinite(data) else None

    return data


@app.get("/")
def home():
    """
    Serve the landing page as a static file.
    This avoids browser quirks from opening HTML directly with file://
    """
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    """
    Serve frontend files without using Flask templates.
    This keeps the backend API-first while still making the app easy to run.
    """
    allowed_files = {
        "index.html",
        "login.html",
        "signup.html",
        "dashboard.html",
        "style.css",
        "script.js",
    }

    if filename in allowed_files:
        return send_from_directory(BASE_DIR, filename)

    return jsonify({"error": "File not found."}), 404


@app.get("/health")
def health():
    """
    Small endpoint to confirm that the API server is running.
    """
    return jsonify(make_json_safe(health_status())), 200


@app.get("/predict")
def predict():
    """
    Predict stock trend, confidence, risk level, and signal strength.

    Example:
    /predict?ticker=AAPL
    """
    ticker = request.args.get("ticker", "AAPL")

    try:
        result = predict_trend_for_ticker(ticker)
        return jsonify(make_json_safe(result)), 200
    except ValueError as error:
        return jsonify(make_json_safe({"error": str(error), "ticker": ticker})), 400
    except Exception:
        return (
            jsonify(
                make_json_safe(
                {
                    "error": "Prediction failed due to an internal server error.",
                    "ticker": ticker,
                }
                )
            ),
            500,
        )


@app.get("/simulate")
def simulate():
    """
    Run Monte Carlo simulation for a ticker.

    Example:
    /simulate?ticker=AAPL
    """
    ticker = request.args.get("ticker", "AAPL")

    try:
        result = simulate_prices_for_ticker(ticker)
        return jsonify(make_json_safe(result)), 200
    except ValueError as error:
        return jsonify(make_json_safe({"error": str(error), "ticker": ticker})), 400
    except Exception:
        return (
            jsonify(
                make_json_safe(
                {
                    "error": "Simulation failed due to an internal server error.",
                    "ticker": ticker,
                }
                )
            ),
            500,
        )


if __name__ == "__main__":
    print("Flask API running at http://127.0.0.1:5000")
    print("Install dependencies with: pip install -r requirements.txt")
    print("Start the server with: python app.py")
    app.run(host="127.0.0.1", port=5000, debug=True)
