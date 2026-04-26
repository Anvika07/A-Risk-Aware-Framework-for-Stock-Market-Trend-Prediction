from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from flask import Flask, jsonify, redirect, render_template, request, url_for

from stock_trend_analyzer import StockTrendAnalyzer

app = Flask(__name__)


def _parse_args() -> Dict[str, Any]:
    ticker = request.args.get("ticker", "NVDA").strip().upper()
    period = request.args.get("period", "2y").strip()
    interval = request.args.get("interval", "1d").strip()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
    }


@app.route("/")
def index() -> str:
    return render_template("login.html")


@app.route("/terminal")
def terminal() -> str:
    return render_template("index.html")


@app.route("/login", methods=["POST"])
def login() -> Any:
    # Demo login flow for local usage.
    return redirect(url_for("terminal"))


def _build_simulation(result: Dict[str, Any], capital: float) -> Dict[str, Any]:
    trend = result["predicted_trend"]
    confidence = float(result["confidence_score"]) / 100.0
    risk = float(result["risk_score"])
    band = result["confidence_band"]

    risk_factor = max(0.25, 1.0 - (risk / 130.0))
    position_fraction = min(0.4, 0.1 + 0.25 * confidence * risk_factor)
    position_size = capital * position_fraction

    if trend == "DOWN":
        strategy = "Short Bias"
        expected_return = -float(band["median_return_pct"]) / 100.0
        worst_return = -float(band["upper_return_pct"]) / 100.0
        best_return = -float(band["lower_return_pct"]) / 100.0
    elif trend == "UP":
        strategy = "Long Bias"
        expected_return = float(band["median_return_pct"]) / 100.0
        worst_return = float(band["lower_return_pct"]) / 100.0
        best_return = float(band["upper_return_pct"]) / 100.0
    else:
        strategy = "Market Neutral"
        expected_return = float(band["median_return_pct"]) / 200.0
        worst_return = float(band["lower_return_pct"]) / 200.0
        best_return = float(band["upper_return_pct"]) / 200.0

    projected_pnl = position_size * expected_return
    worst_pnl = position_size * worst_return
    best_pnl = position_size * best_return
    stop_loss_pct = max(0.02, abs(worst_return) * 0.75)
    take_profit_pct = max(0.02, abs(best_return) * 0.85)

    return {
        "initial_capital": round(capital, 2),
        "position_size": round(position_size, 2),
        "position_fraction_pct": round(position_fraction * 100, 2),
        "strategy_bias": strategy,
        "projected_return_pct": round(expected_return * 100, 2),
        "projected_pnl": round(projected_pnl, 2),
        "worst_case_pnl": round(worst_pnl, 2),
        "best_case_pnl": round(best_pnl, 2),
        "stop_loss_pct": round(stop_loss_pct * 100, 2),
        "take_profit_pct": round(take_profit_pct * 100, 2),
    }


@app.route("/api/analyze")
def analyze() -> Any:
    try:
        params = _parse_args()
        analyzer = StockTrendAnalyzer()
        result = analyzer.analyze(
            ticker=params["ticker"],
            period=params["period"],
            interval=params["interval"],
        )
        payload = asdict(result)
        payload["selected_stable_threshold"] = analyzer.stable_threshold
        payload["selected_model_params"] = analyzer.model_params
        capital = float(request.args.get("capital", "10000"))
        payload["simulation"] = _build_simulation(payload, capital)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/history")
def history() -> Any:
    try:
        params = _parse_args()
        analyzer = StockTrendAnalyzer()
        data = analyzer.download_data(
            ticker=params["ticker"],
            period=params["period"],
            interval=params["interval"],
        )
        featured = analyzer.add_indicators(data)
        close = featured["Close"].dropna()
        sma_20 = featured["sma_20"].dropna()
        sma_50 = featured["sma_50"].dropna()

        payload = {
            "ticker": params["ticker"],
            "labels": [idx.strftime("%Y-%m-%d") for idx in close.index],
            "close": [round(float(v), 2) for v in close.values],
            "sma_20": [round(float(v), 2) for v in sma_20.reindex(close.index).values],
            "sma_50": [round(float(v), 2) for v in sma_50.reindex(close.index).values],
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/simulate")
def simulate() -> Any:
    try:
        params = _parse_args()
        capital = float(request.args.get("capital", "10000"))
        analyzer = StockTrendAnalyzer()
        result = analyzer.analyze(
            ticker=params["ticker"],
            period=params["period"],
            interval=params["interval"],
        )
        payload = asdict(result)
        simulation = _build_simulation(payload, capital)
        return jsonify(
            {
                "ticker": params["ticker"],
                "capital": capital,
                "simulation": simulation,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)
