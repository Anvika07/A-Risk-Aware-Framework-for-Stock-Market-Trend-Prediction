from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TREND_UP = "UP"
TREND_DOWN = "DOWN"
TREND_STABLE = "STABLE"


@dataclass
class AnalysisResult:
    ticker: str
    as_of_date: str
    predicted_trend: str
    confidence_score: float
    backtest_accuracy_score: float
    backtest_summary: Dict[str, object]
    confidence_band: Dict[str, float]
    risk_score: float
    volatility_classification: str
    indicators: Dict[str, float]
    explanation: List[str]
    failure_analysis_guidance: List[str]


class StockTrendAnalyzer:
    def __init__(
        self,
        horizon_days: int = 5,
        stable_threshold: float = 0.006,
        monte_carlo_paths: int = 5000,
        random_state: int = 42,
    ) -> None:
        self.horizon_days = horizon_days
        self.stable_threshold = stable_threshold
        self.monte_carlo_paths = monte_carlo_paths
        self.random_state = random_state
        self.model_params: Dict[str, object] = {
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        }
        self.model = self._build_model(self.model_params)

    def _build_model(self, model_params: Optional[Dict[str, object]] = None) -> Pipeline:
        params = dict(self.model_params)
        if model_params:
            params.update(model_params)

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=int(params["max_depth"]),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_features=params["max_features"],
                        random_state=self.random_state,
                        class_weight=params["class_weight"],
                    ),
                ),
            ]
        )

    def download_data(
        self, ticker: str, period: str = "2y", interval: str = "1d"
    ) -> pd.DataFrame:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'.")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required.difference(set(data.columns))
        if missing:
            raise ValueError(f"Downloaded data is missing required columns: {missing}")

        return data.sort_index()

    def period_rank(self, period: str) -> int:
        order = {
            "1mo": 1,
            "3mo": 2,
            "6mo": 3,
            "1y": 4,
            "2y": 5,
            "5y": 6,
            "10y": 7,
            "max": 8,
        }
        return order.get(period, 0)

    def expand_history_if_needed(
        self, ticker: str, period: str, interval: str, min_rows: int = 50
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        candidate_periods = ["2y", "5y", "10y", "max"]

        data = self.download_data(ticker=ticker, period=period, interval=interval)
        featured = self.add_indicators(data)
        training_frame = self.build_training_frame(featured)
        usable_rows = len(self.get_usable_rows(training_frame))
        if usable_rows >= min_rows:
            return featured, training_frame

        current_rank = self.period_rank(period)
        for alt_period in candidate_periods:
            if self.period_rank(alt_period) <= current_rank:
                continue

            alt_data = self.download_data(ticker=ticker, period=alt_period, interval=interval)
            alt_featured = self.add_indicators(alt_data)
            alt_training_frame = self.build_training_frame(alt_featured)
            alt_usable_rows = len(self.get_usable_rows(alt_training_frame))
            if alt_usable_rows >= min_rows:
                return alt_featured, alt_training_frame

        return featured, training_frame

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        close = df["Close"]

        df["return_1d"] = close.pct_change()
        df["return_5d"] = close.pct_change(5)
        df["momentum_10d"] = close / close.shift(10) - 1.0
        df["sma_10"] = close.rolling(10).mean()
        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(14).mean()

        df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)
        df["volume_ratio_20d"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["distance_sma_20"] = (close - df["sma_20"]) / df["sma_20"]
        df["distance_sma_50"] = (close - df["sma_50"]) / df["sma_50"]
        return df

    def build_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if "future_return" not in frame.columns:
            frame["future_return"] = (
                frame["Close"].shift(-self.horizon_days) / frame["Close"] - 1.0
            )
        return self.apply_target_labels(frame, self.stable_threshold)

    def apply_target_labels(self, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
        labeled = frame.copy()
        if "future_return" not in labeled.columns:
            labeled["future_return"] = (
                labeled["Close"].shift(-self.horizon_days) / labeled["Close"] - 1.0
            )
        labeled["target"] = np.select(
            [
                labeled["future_return"] > threshold,
                labeled["future_return"] < -threshold,
            ],
            [TREND_UP, TREND_DOWN],
            default=TREND_STABLE,
        )
        return labeled

    def get_usable_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.dropna(subset=self.feature_columns + ["target"]).copy()

    def tune_stable_threshold(self, frame: pd.DataFrame) -> float:
        candidates = [0.003, 0.0045, 0.006, 0.008, 0.01, 0.012]
        best_threshold = self.stable_threshold
        best_score = -1.0
        best_accuracy = -1.0

        for threshold in candidates:
            labeled = self.apply_target_labels(frame, threshold)
            usable = self.get_usable_rows(labeled)
            try:
                summary = self.backtest(usable, purge_gap=self.horizon_days)
            except ValueError:
                continue

            macro_f1 = float(summary["macro_f1_score"]) / 100.0
            accuracy = float(summary["overall_accuracy_pct"]) / 100.0
            if macro_f1 > best_score or (
                np.isclose(macro_f1, best_score) and accuracy > best_accuracy
            ):
                best_threshold = threshold
                best_score = macro_f1
                best_accuracy = accuracy

        return best_threshold

    def select_model_params(self, usable: pd.DataFrame) -> Dict[str, object]:
        validation_size = max(40, int(len(usable) * 0.2))
        min_train_rows = 120
        if len(usable) <= validation_size + min_train_rows:
            return dict(self.model_params)

        train_slice = usable.iloc[:-validation_size]
        validation_slice = usable.iloc[-validation_size:]
        labels = [TREND_UP, TREND_DOWN, TREND_STABLE]

        candidate_params: List[Dict[str, object]] = [
            {
                "n_estimators": 300,
                "max_depth": 6,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
            {
                "n_estimators": 500,
                "max_depth": 8,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
                "class_weight": "balanced",
            },
            {
                "n_estimators": 400,
                "max_depth": 5,
                "min_samples_leaf": 6,
                "max_features": 0.7,
                "class_weight": "balanced_subsample",
            },
        ]

        best_params = dict(self.model_params)
        best_score = -1.0

        for params in candidate_params:
            model = self._build_model(params)
            model.fit(train_slice[self.feature_columns], train_slice["target"])
            predictions = model.predict(validation_slice[self.feature_columns])
            score = f1_score(
                validation_slice["target"],
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
            if score > best_score:
                best_score = score
                best_params = dict(params)

        return best_params

    @property
    def feature_columns(self) -> List[str]:
        return [
            "return_1d",
            "return_5d",
            "momentum_10d",
            "sma_10",
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr_14",
            "volatility_20d",
            "volume_ratio_20d",
            "distance_sma_20",
            "distance_sma_50",
        ]

    def fit(self, frame: pd.DataFrame) -> pd.DataFrame:
        usable = self.get_usable_rows(frame)
        if len(usable) < 50:
            raise ValueError(
                "Not enough historical rows to train reliably. Try a longer period."
            )

        X = usable[self.feature_columns]
        y = usable["target"]
        self.model.fit(X, y)
        return usable

    def backtest(
        self,
        usable: pd.DataFrame,
        train_window: Optional[int] = None,
        test_window: Optional[int] = None,
        step_size: Optional[int] = None,
        purge_gap: Optional[int] = None,
    ) -> Dict[str, object]:
        gap = self.horizon_days if purge_gap is None else purge_gap
        train_window, test_window, step_size = self.resolve_backtest_windows(
            sample_size=len(usable),
            purge_gap=gap,
            requested_train=train_window,
            requested_test=test_window,
            requested_step=step_size,
        )

        if len(usable) < train_window + gap + test_window:
            raise ValueError(
                "Not enough historical rows for backtesting. Try a longer history period."
            )

        predictions: List[str] = []
        actuals: List[str] = []
        evaluated_indices: List[pd.Timestamp] = []
        walk_forward_periods = 0

        max_start = len(usable) - train_window - gap - test_window
        for start in range(0, max_start + 1, step_size):
            train_end = start + train_window
            test_start = train_end + gap
            test_end = test_start + test_window

            train_slice = usable.iloc[start:train_end]
            test_slice = usable.iloc[test_start:test_end]

            model = self._build_model(self.model_params)
            model.fit(train_slice[self.feature_columns], train_slice["target"])
            test_predictions = model.predict(test_slice[self.feature_columns])

            predictions.extend(test_predictions.tolist())
            actuals.extend(test_slice["target"].tolist())
            evaluated_indices.extend(test_slice.index.tolist())
            walk_forward_periods += 1

        if not predictions:
            raise ValueError("Backtest produced no evaluation samples.")

        labels = [TREND_UP, TREND_DOWN, TREND_STABLE]
        report = classification_report(
            actuals,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(actuals, predictions, labels=labels)
        failure_analysis = self.analyze_prediction_failures(
            usable=usable,
            actuals=actuals,
            predictions=predictions,
            evaluated_indices=evaluated_indices,
        )

        return {
            "overall_accuracy_pct": round(float(accuracy_score(actuals, predictions) * 100), 2),
            "macro_f1_score": round(
                float(
                    f1_score(
                        actuals,
                        predictions,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                    * 100
                ),
                2,
            ),
            "samples_evaluated": len(predictions),
            "walk_forward_periods": walk_forward_periods,
            "train_window_days": train_window,
            "test_window_days": test_window,
            "step_size_days": step_size,
            "purge_gap_days": gap,
            "per_class_recall_pct": {
                label: round(float(report[label]["recall"] * 100), 2)
                for label in labels
            },
            "confusion_matrix": {
                "labels": labels,
                "matrix": matrix.tolist(),
            },
            "failure_analysis": failure_analysis,
        }

    def resolve_backtest_windows(
        self,
        sample_size: int,
        purge_gap: int,
        requested_train: Optional[int],
        requested_test: Optional[int],
        requested_step: Optional[int],
    ) -> Tuple[int, int, int]:
        if requested_train and requested_test and requested_step:
            return requested_train, requested_test, requested_step

        train_window = min(180, max(40, int(sample_size * 0.55)))
        test_window = min(60, max(10, int(sample_size * 0.2)))

        while sample_size < train_window + purge_gap + test_window and train_window > 35:
            train_window -= 5

        max_test = sample_size - train_window - purge_gap
        if max_test < 10:
            train_window = max(20, sample_size - purge_gap - 10)
            max_test = sample_size - train_window - purge_gap
        test_window = max(10, min(test_window, max_test))
        step_size = max(5, min(20, test_window // 2))
        return train_window, test_window, step_size

    def fallback_backtest_summary(self, usable: pd.DataFrame) -> Dict[str, object]:
        split = max(10, int(len(usable) * 0.2))
        if len(usable) <= split + 10:
            raise ValueError(
                "Not enough historical rows for fallback backtesting. Try a longer history period."
            )

        train_slice = usable.iloc[:-split]
        test_slice = usable.iloc[-split:]
        model = self._build_model(self.model_params)
        model.fit(train_slice[self.feature_columns], train_slice["target"])
        predictions = model.predict(test_slice[self.feature_columns])
        actuals = test_slice["target"].tolist()
        labels = [TREND_UP, TREND_DOWN, TREND_STABLE]
        report = classification_report(
            actuals,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(actuals, predictions, labels=labels)
        failure_analysis = self.analyze_prediction_failures(
            usable=usable,
            actuals=actuals,
            predictions=predictions.tolist(),
            evaluated_indices=test_slice.index.tolist(),
        )

        return {
            "overall_accuracy_pct": round(float(accuracy_score(actuals, predictions) * 100), 2),
            "macro_f1_score": round(
                float(
                    f1_score(
                        actuals,
                        predictions,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                    * 100
                ),
                2,
            ),
            "samples_evaluated": len(predictions),
            "walk_forward_periods": 1,
            "train_window_days": len(train_slice),
            "test_window_days": len(test_slice),
            "step_size_days": len(test_slice),
            "purge_gap_days": 0,
            "per_class_recall_pct": {
                label: round(float(report[label]["recall"] * 100), 2)
                for label in labels
            },
            "confusion_matrix": {
                "labels": labels,
                "matrix": matrix.tolist(),
            },
            "failure_analysis": failure_analysis,
            "backtest_mode": "fallback_holdout",
        }

    def analyze_prediction_failures(
        self,
        usable: pd.DataFrame,
        actuals: List[str],
        predictions: List[str],
        evaluated_indices: List[pd.Timestamp],
    ) -> List[str]:
        if not actuals or not predictions or not evaluated_indices:
            return self.failure_analysis_guidance()

        eval_frame = usable.loc[evaluated_indices].copy()
        eval_frame["actual"] = actuals
        eval_frame["predicted"] = predictions
        eval_frame["is_error"] = eval_frame["actual"] != eval_frame["predicted"]

        total = len(eval_frame)
        errors = int(eval_frame["is_error"].sum())
        if errors == 0:
            return [
                "No backtest misclassifications were observed in evaluated windows.",
                "Residual risk remains from regime shifts and event-driven price shocks.",
            ]

        failure_rows = eval_frame[eval_frame["is_error"]]
        success_rows = eval_frame[~eval_frame["is_error"]]
        error_rate = (errors / total) * 100

        analysis: List[str] = [
            f"Backtest logged {errors} misclassifications across {total} evaluated samples ({error_rate:.1f}% error rate)."
        ]

        if not success_rows.empty:
            fail_vol = float(failure_rows["volatility_20d"].mean())
            ok_vol = float(success_rows["volatility_20d"].mean())
            if fail_vol > ok_vol:
                analysis.append(
                    f"Mispredictions cluster in higher-volatility periods (error avg {fail_vol * 100:.2f}% annualized vs {ok_vol * 100:.2f}% when correct)."
                )

            fail_abs_momentum = float(failure_rows["momentum_10d"].abs().mean())
            ok_abs_momentum = float(success_rows["momentum_10d"].abs().mean())
            if fail_abs_momentum < ok_abs_momentum:
                analysis.append(
                    "Many errors occur during low-momentum price action, where trend signals are weak and noisy."
                )

        confusion = (
            failure_rows.groupby(["predicted", "actual"])
            .size()
            .sort_values(ascending=False)
        )
        if not confusion.empty:
            top_pair, top_count = confusion.index[0], int(confusion.iloc[0])
            analysis.append(
                f"Most frequent confusion pattern is predicted {top_pair[0]} but realized {top_pair[1]} ({top_count} cases)."
            )

        if len(analysis) < 4:
            analysis.append(
                "Event risk, gaps, and abrupt sentiment shifts can still invalidate indicator-driven forecasts."
            )

        return analysis

    def monte_carlo_band(self, returns: pd.Series, last_close: float) -> Dict[str, float]:
        clean_returns = returns.dropna().tail(60)
        if len(clean_returns) < 20:
            raise ValueError("Not enough return history for Monte Carlo simulation.")

        mu = clean_returns.mean()
        sigma = clean_returns.std(ddof=1)
        rng = np.random.default_rng(self.random_state)
        random_returns = rng.normal(
            loc=mu,
            scale=sigma,
            size=(self.monte_carlo_paths, self.horizon_days),
        )
        random_returns = np.clip(random_returns, -0.95, None)
        terminal_prices = last_close * np.exp(np.log1p(random_returns).sum(axis=1))
        terminal_returns = terminal_prices / last_close - 1.0

        return {
            "lower_return_pct": round(float(np.percentile(terminal_returns, 10) * 100), 2),
            "median_return_pct": round(float(np.percentile(terminal_returns, 50) * 100), 2),
            "upper_return_pct": round(float(np.percentile(terminal_returns, 90) * 100), 2),
            "annualized_volatility_pct": round(float(sigma * np.sqrt(252) * 100), 2),
        }

    def calculate_risk_score(
        self, probabilities: np.ndarray, recent_volatility: float, band: Dict[str, float]
    ) -> float:
        max_entropy = np.log(len(probabilities))
        uncertainty = float(entropy(probabilities) / max_entropy) if max_entropy else 0.0
        vol_component = min(recent_volatility / 0.6, 1.0)
        band_width = abs(band["upper_return_pct"] - band["lower_return_pct"]) / 20.0
        band_component = min(band_width, 1.0)
        score = (0.45 * uncertainty) + (0.35 * vol_component) + (0.20 * band_component)
        return round(score * 100, 2)

    def classify_volatility(self, annualized_volatility: float) -> str:
        if annualized_volatility < 0.18:
            return "Stable"
        if annualized_volatility < 0.35:
            return "Risky"
        return "Highly Volatile"

    def build_explanation(
        self,
        latest: pd.Series,
        predicted_trend: str,
        confidence_score: float,
        volatility_label: str,
        backtest_accuracy_score: float,
    ) -> List[str]:
        explanation: List[str] = [
            f"Model suggests a {predicted_trend} short-term trend with {confidence_score:.1f}% confidence."
        ]
        explanation.append(
            f"Walk-forward backtesting measured directional accuracy at {backtest_accuracy_score:.1f}%."
        )

        if latest["sma_20"] > latest["sma_50"]:
            explanation.append("Short-term moving averages are above longer-term averages, supporting bullish structure.")
        else:
            explanation.append("Short-term moving averages are below longer-term averages, which weakens near-term momentum.")

        rsi = latest["rsi_14"]
        if rsi >= 70:
            explanation.append("RSI is elevated, which can signal overbought conditions and raise reversal risk.")
        elif rsi <= 30:
            explanation.append("RSI is depressed, which can indicate oversold conditions and rebound potential.")
        else:
            explanation.append("RSI is in a neutral range, so momentum is present without an extreme exhaustion signal.")

        if latest["macd"] > latest["macd_signal"]:
            explanation.append("MACD is above its signal line, indicating positive momentum acceleration.")
        else:
            explanation.append("MACD is below its signal line, indicating momentum is softening.")

        if latest["momentum_10d"] > 0:
            explanation.append("10-day price momentum remains positive.")
        else:
            explanation.append("10-day price momentum is negative, which points to recent weakness.")

        explanation.append(
            f"Volatility regime is classified as {volatility_label}, which is factored into the risk score."
        )
        return explanation

    def failure_analysis_guidance(self) -> List[str]:
        return [
            "A wrong directional call can happen when sudden volatility spikes overwhelm recent technical patterns.",
            "Unexpected macro news, earnings surprises, or sector-specific events can invalidate indicator-based signals.",
            "Thin liquidity, bad ticks, or missing data can distort RSI, MACD, and volatility estimates.",
            "The classifier learns from historical relationships, so regime shifts and rare events remain model limitations.",
        ]

    def analyze(self, ticker: str, period: str = "2y", interval: str = "1d") -> AnalysisResult:
        featured, training_frame = self.expand_history_if_needed(
            ticker=ticker,
            period=period,
            interval=interval,
            min_rows=50,
        )

        self.stable_threshold = self.tune_stable_threshold(training_frame)
        training_frame = self.apply_target_labels(training_frame, self.stable_threshold)

        usable_for_selection = self.get_usable_rows(training_frame)
        self.model_params = self.select_model_params(usable_for_selection)
        self.model = self._build_model(self.model_params)

        usable = self.fit(training_frame)
        try:
            backtest_summary = self.backtest(usable, purge_gap=self.horizon_days)
        except ValueError:
            backtest_summary = self.fallback_backtest_summary(usable)
        backtest_accuracy_score = float(backtest_summary["overall_accuracy_pct"])

        latest = featured.iloc[-1]
        latest_features = pd.DataFrame([latest[self.feature_columns]])
        probabilities = self.model.predict_proba(latest_features)[0]
        labels = list(self.model.classes_)
        probability_map = dict(zip(labels, probabilities))
        predicted_trend = max(probability_map, key=probability_map.get)
        confidence_score = round(float(probability_map[predicted_trend] * 100), 2)

        band = self.monte_carlo_band(featured["return_1d"], float(latest["Close"]))
        recent_volatility = float(latest["volatility_20d"])
        risk_score = self.calculate_risk_score(probabilities, recent_volatility, band)
        volatility_label = self.classify_volatility(recent_volatility)

        indicators = {
            "close": round(float(latest["Close"]), 2),
            "sma_20": round(float(latest["sma_20"]), 2),
            "sma_50": round(float(latest["sma_50"]), 2),
            "rsi_14": round(float(latest["rsi_14"]), 2),
            "macd": round(float(latest["macd"]), 4),
            "macd_signal": round(float(latest["macd_signal"]), 4),
            "momentum_10d_pct": round(float(latest["momentum_10d"] * 100), 2),
            "atr_14": round(float(latest["atr_14"]), 2),
            "annualized_volatility_pct": round(float(recent_volatility * 100), 2),
        }

        return AnalysisResult(
            ticker=ticker.upper(),
            as_of_date=str(featured.index[-1].date()),
            predicted_trend=predicted_trend,
            confidence_score=confidence_score,
            backtest_accuracy_score=backtest_accuracy_score,
            backtest_summary=backtest_summary,
            confidence_band=band,
            risk_score=risk_score,
            volatility_classification=volatility_label,
            indicators=indicators,
            explanation=self.build_explanation(
                latest=latest,
                predicted_trend=predicted_trend,
                confidence_score=confidence_score,
                volatility_label=volatility_label,
                backtest_accuracy_score=backtest_accuracy_score,
            ),
            failure_analysis_guidance=backtest_summary.get(
                "failure_analysis", self.failure_analysis_guidance()
            ),
        )

    def create_plot(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        output_path: Optional[str] = None,
    ) -> str:
        data = self.download_data(ticker=ticker, period=period, interval=interval)
        featured = self.add_indicators(data)

        fig, (ax_price, ax_rsi) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        ax_price.plot(featured.index, featured["Close"], label="Close", linewidth=1.5)
        ax_price.plot(featured.index, featured["sma_20"], label="SMA 20", linewidth=1.2)
        ax_price.plot(featured.index, featured["sma_50"], label="SMA 50", linewidth=1.2)
        ax_price.set_title(f"{ticker.upper()} price with moving averages")
        ax_price.set_ylabel("Price")
        ax_price.legend()
        ax_price.grid(alpha=0.2)

        ax_rsi.plot(featured.index, featured["rsi_14"], label="RSI 14", color="tab:orange")
        ax_rsi.axhline(70, color="red", linestyle="--", linewidth=1)
        ax_rsi.axhline(30, color="green", linestyle="--", linewidth=1)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.set_xlabel("Date")
        ax_rsi.grid(alpha=0.2)

        fig.tight_layout()
        output = output_path or f"{ticker.upper()}_trend_plot.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output


def format_text_report(result: AnalysisResult) -> str:
    lines = [
        f"Ticker: {result.ticker}",
        f"As of: {result.as_of_date}",
        f"Trend classification: {result.predicted_trend}",
        f"Confidence score: {result.confidence_score:.2f}%",
        f"Backtest accuracy score: {result.backtest_accuracy_score:.2f}%",
        (
            "Confidence band (Monte Carlo, horizon returns): "
            f"{result.confidence_band['lower_return_pct']:.2f}% to "
            f"{result.confidence_band['upper_return_pct']:.2f}%"
        ),
        f"Risk score: {result.risk_score:.2f}/100",
        f"Volatility classification: {result.volatility_classification}",
        "Indicators:",
    ]
    lines.extend(f"  - {key}: {value}" for key, value in result.indicators.items())
    lines.append("Backtest summary:")
    lines.extend(f"  - {key}: {value}" for key, value in result.backtest_summary.items())
    lines.append("Explanation:")
    lines.extend(f"  - {item}" for item in result.explanation)
    lines.append("Prediction failure analysis:")
    lines.extend(f"  - {item}" for item in result.failure_analysis_guidance)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Risk-aware short-term stock market trend analyzer."
    )
    parser.add_argument("ticker", help="Ticker symbol to analyze, e.g. AAPL")
    parser.add_argument("--period", default="2y", help="yfinance history period, default: 2y")
    parser.add_argument("--interval", default="1d", help="yfinance interval, default: 1d")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the analysis result as JSON instead of text.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a PNG chart with close price, moving averages, and RSI.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional output path for the generated plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = StockTrendAnalyzer()
    result = analyzer.analyze(ticker=args.ticker, period=args.period, interval=args.interval)

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(format_text_report(result))

    if args.plot:
        output = analyzer.create_plot(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            output_path=args.plot_output,
        )
        print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
