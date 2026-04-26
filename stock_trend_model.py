import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import TimeSeriesSplit


warnings.filterwarnings("ignore")

PREVIOUS_AVERAGE_ACCURACY = 0.4788


def get_feature_columns():
    """
    Return the full list of engineered features used by the model.
    Keeping this in one place makes the script easier to reuse from the web app.
    """
    return [
        "MA_10",
        "EMA_10",
        "MA_diff",
        "EMA_diff",
        "RSI",
        "Daily_Return",
        "Momentum_3",
        "Volatility_5",
        "Return_3",
        "Return_5",
        "Return_3_lag",
        "Return_5_lag",
        "RSI_lag_2",
        "Volume_Change",
        "Volume_MA_5",
        "RSI_Momentum",
        "Volatility_Return",
    ]


def fetch_stock_data(ticker="AAPL", period="5y"):
    """
    Download daily stock data using yfinance.
    We use 5 years by default so the model has more history to learn patterns.
    """
    print(f"\nDownloading historical data for {ticker}...")

    # Keep yfinance cache files inside the current project folder.
    # This avoids permission issues in restricted environments.
    cache_dir = os.path.join(os.getcwd(), ".yf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    yf.set_tz_cache_location(cache_dir)

    data = yf.download(ticker, period=period, interval="1d", progress=False)

    if data.empty:
        raise ValueError(
            "No data was downloaded. Check the ticker symbol or your internet connection."
        )

    # Some yfinance versions can return multi-level columns.
    # This keeps the script beginner-friendly by flattening them.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.copy()
    data.dropna(inplace=True)
    return data


def calculate_rsi(series, window=14):
    """
    Calculate the Relative Strength Index (RSI).

    RSI helps measure whether a stock has had many recent gains or losses.
    Values above 70 can suggest overbought conditions.
    Values below 30 can suggest oversold conditions.
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(window=window).mean()
    average_loss = loss.rolling(window=window).mean()

    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_features(data):
    """
    Create simple technical indicators for the model.
    These features use current and past data only, which helps avoid data leakage.
    """
    df = data.copy()

    # Daily percentage return from one day to the next.
    df["Daily_Return"] = df["Close"].pct_change()

    # Momentum compares today's close with the close from 3 days ago.
    # Positive values can suggest upward movement over the last few days.
    df["Momentum_3"] = df["Close"] - df["Close"].shift(3)

    # 10-day simple moving average.
    df["MA_10"] = df["Close"].rolling(window=10).mean()

    # 10-day exponential moving average.
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # These features show how far the current price is from the moving averages.
    df["MA_diff"] = df["Close"] - df["MA_10"]
    df["EMA_diff"] = df["Close"] - df["EMA_10"]

    # RSI is a common momentum indicator.
    df["RSI"] = calculate_rsi(df["Close"], window=14)

    # Rolling volatility measures how much returns have been moving recently.
    # We use 5 days to keep the feature simple.
    df["Volatility_5"] = df["Daily_Return"].rolling(window=5).std()

    # Multi-day returns help the model see slightly longer short-term movement.
    df["Return_3"] = df["Close"].pct_change(periods=3)
    df["Return_5"] = df["Close"].pct_change(periods=5)

    # Lagged return features give the model a small view into recent past behavior.
    df["Return_3_lag"] = df["Return_3"].shift(1)
    df["Return_5_lag"] = df["Return_5"].shift(1)

    # A lagged RSI can show whether momentum has been building for more than one day.
    df["RSI_lag_2"] = df["RSI"].shift(2)

    # Volume features help capture changes in trading activity.
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA_5"] = df["Volume"].rolling(window=5).mean()

    # Simple interaction features combine momentum and volatility information.
    df["RSI_Momentum"] = df["RSI"] * df["Momentum_3"]
    df["Volatility_Return"] = df["Volatility_5"] * df["Daily_Return"]

    return df


def create_labels(data):
    """
    Create the target label based on the NEXT day's return.
    This is now a binary problem, so we only keep clear UP or DOWN moves.

    UP   -> next day return > 1%
    DOWN -> next day return < -1%
    """
    df = data.copy()

    df["Next_Day_Return"] = df["Close"].shift(-1) / df["Close"] - 1

    df["Trend"] = pd.Series(index=df.index, dtype="object")
    df.loc[df["Next_Day_Return"] > 0.01, "Trend"] = "UP"
    df.loc[df["Next_Day_Return"] < -0.01, "Trend"] = "DOWN"

    return df


def prepare_dataset(data):
    """
    Combine feature engineering and label creation, then remove rows with missing values.
    Missing values happen naturally because indicators like MA and RSI need past data.
    """
    df = create_features(data)
    df = create_labels(df)
    # Rows without a label are removed because they are neither clear UP nor clear DOWN cases.
    df.dropna(inplace=True)
    return df


def time_series_split_data(data, feature_columns, target_column="Trend", n_splits=5):
    """
    Split the data using TimeSeriesSplit.
    Each split trains on older data and tests on newer data.
    This is a safer approach for time-series problems than random shuffling.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits = []

    for fold_number, (train_index, test_index) in enumerate(splitter.split(data), start=1):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        split_info = {
            "fold_number": fold_number,
            "x_train": train_data[feature_columns],
            "y_train": train_data[target_column],
            "x_test": test_data[feature_columns],
            "y_test": test_data[target_column],
            "train_data": train_data,
            "test_data": test_data,
        }
        splits.append(split_info)

    return splits


def train_model(x_train, y_train):
    """
    Train a Random Forest classifier.
    We keep the model type the same, but use stronger parameters than before.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test, fold_name="Final Fold", show_details=True):
    """
    Print standard classification metrics.
    These help us understand how often the model is correct and where it makes mistakes.
    """
    labels = ["UP", "DOWN"]
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0,
    )
    _, recall_scores, f1_scores, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=labels,
        zero_division=0,
    )

    if show_details:
        print("\n" + "=" * 60)
        print(f"MODEL EVALUATION - {fold_name}")
        print("=" * 60)
        print(f"Accuracy Score: {accuracy:.2%}")
        print("\nConfusion Matrix")
        print("Rows = Actual, Columns = Predicted")
        print(pd.DataFrame(matrix, index=labels, columns=labels))
        print("\nClassification Report")
        print(report)

        print("Recall and F1-Score by Class")
        for label, recall_value, f1_value in zip(labels, recall_scores, f1_scores):
            print(f"  {label:<6} Recall: {recall_value:.2%} | F1-Score: {f1_value:.2%}")

        print("\nHighlight: Performance on both classes")
        print(
            f"  UP   -> Recall: {recall_scores[0]:.2%}, F1-Score: {f1_scores[0]:.2%}"
        )
        print(
            f"  DOWN -> Recall: {recall_scores[1]:.2%}, F1-Score: {f1_scores[1]:.2%}"
        )

    metrics = {
        "accuracy": accuracy,
        "up_recall": recall_scores[0],
        "up_f1": f1_scores[0],
        "down_recall": recall_scores[1],
        "down_f1": f1_scores[1],
    }

    return metrics, predictions


def print_feature_importance(model, feature_columns):
    """
    Show which features the Random Forest used most.
    Higher importance means the model relied on that feature more often.
    """
    importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    print(importance_df.to_string(index=False))

    return importance_df


def select_features_by_importance(importance_df, min_features_to_keep=8, features_to_remove=4):
    """
    Remove the lowest-importance features and keep the rest.
    This is a simple way to reduce noise while staying beginner-friendly.
    """
    sorted_importance = importance_df.sort_values(by="Importance", ascending=True)

    removable_count = min(features_to_remove, len(sorted_importance) - min_features_to_keep)
    removable_count = max(removable_count, 0)

    removed_features = sorted_importance.head(removable_count)["Feature"].tolist()
    selected_features = [
        feature for feature in importance_df["Feature"].tolist() if feature not in removed_features
    ]

    print("\n" + "=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)
    if removed_features:
        print("Removed lowest-importance features:")
        for feature in removed_features:
            print(f"  {feature}")
    else:
        print("No features were removed because the feature set is already small.")

    print("\nFeatures kept for retraining:")
    for feature in selected_features:
        print(f"  {feature}")

    return selected_features, removed_features


def evaluate_time_series_folds(dataset, feature_columns, target_column="Trend", n_splits=5, show_fold_details=True):
    """
    Train and evaluate one model per time-series fold.
    We print each fold so a beginner can see how results change over time.
    """
    splits = time_series_split_data(
        dataset,
        feature_columns=feature_columns,
        target_column=target_column,
        n_splits=n_splits,
    )

    fold_metrics = []
    final_model = None

    print("\n" + "=" * 60)
    print("TIME SERIES SPLIT SUMMARY")
    print("=" * 60)

    for split in splits:
        print(
            f"Fold {split['fold_number']}: "
            f"train={len(split['train_data'])} rows, "
            f"test={len(split['test_data'])} rows"
        )

        model = train_model(split["x_train"], split["y_train"])
        metrics, _ = evaluate_model(
            model,
            split["x_test"],
            split["y_test"],
            fold_name=f"Fold {split['fold_number']}",
            show_details=show_fold_details,
        )
        fold_metrics.append(metrics)
        final_model = model

    summary = {
        "accuracy": float(np.mean([metric["accuracy"] for metric in fold_metrics])),
        "up_recall": float(np.mean([metric["up_recall"] for metric in fold_metrics])),
        "up_f1": float(np.mean([metric["up_f1"] for metric in fold_metrics])),
        "down_recall": float(np.mean([metric["down_recall"] for metric in fold_metrics])),
        "down_f1": float(np.mean([metric["down_f1"] for metric in fold_metrics])),
    }

    print("\nAverage Metrics Across Folds")
    print(f"  Accuracy: {summary['accuracy']:.2%}")
    print(f"  UP Recall: {summary['up_recall']:.2%} | UP F1: {summary['up_f1']:.2%}")
    print(
        f"  DOWN Recall: {summary['down_recall']:.2%} | DOWN F1: {summary['down_f1']:.2%}"
    )

    return final_model, summary


def build_prediction_result(ticker="AAPL", period="5y", n_splits=5, simulations=100, days=30, verbose=True):
    """
    Run the full prediction pipeline and return a dictionary that is easy to use
    in both the console script and the frontend API.
    """
    all_feature_columns = get_feature_columns()

    stock_data = fetch_stock_data(ticker=ticker, period=period)
    dataset = prepare_dataset(stock_data)

    if verbose:
        print("\nPrepared dataset preview:")
        print(dataset[["Close"] + all_feature_columns + ["Trend"]].tail())

        print("\nClass distribution:")
        print(dataset["Trend"].value_counts())

    baseline_model, baseline_summary = evaluate_time_series_folds(
        dataset,
        feature_columns=all_feature_columns,
        target_column="Trend",
        n_splits=n_splits,
        show_fold_details=verbose,
    )

    importance_df = print_feature_importance(baseline_model, all_feature_columns)
    selected_features, removed_features = select_features_by_importance(
        importance_df,
        min_features_to_keep=8,
        features_to_remove=4,
    )

    if verbose:
        print("\nRetraining model after removing the lowest-importance features...")

    refined_model, refined_summary = evaluate_time_series_folds(
        dataset,
        feature_columns=selected_features,
        target_column="Trend",
        n_splits=n_splits,
        show_fold_details=False,
    )

    predicted_trend, confidence, class_probabilities, signal_strength = predict_latest_trend(
        refined_model,
        dataset,
        selected_features,
    )
    volatility, risk_level = analyze_risk(stock_data)
    simulation_points = monte_carlo_simulation(stock_data, simulations=simulations, days=days)

    if verbose:
        print("\nFinal feature list used for prediction:")
        for feature in selected_features:
            print(f"  {feature}")

        compare_accuracy_results(
            PREVIOUS_AVERAGE_ACCURACY,
            baseline_summary["accuracy"],
            refined_summary["accuracy"],
        )
        interpret_results(refined_summary["accuracy"], confidence, risk_level, signal_strength)

    return {
        "ticker": ticker.upper(),
        "trend": predicted_trend,
        "confidence": float(confidence),
        "risk_level": risk_level,
        "signal_strength": signal_strength,
        "class_probabilities": {key: float(value) for key, value in class_probabilities.items()},
        "volatility": float(volatility),
        "baseline_accuracy": float(baseline_summary["accuracy"]),
        "refined_accuracy": float(refined_summary["accuracy"]),
        "previous_accuracy": float(PREVIOUS_AVERAGE_ACCURACY),
        "accuracy_change_vs_previous": float(refined_summary["accuracy"] - PREVIOUS_AVERAGE_ACCURACY),
        "accuracy_change_vs_baseline": float(refined_summary["accuracy"] - baseline_summary["accuracy"]),
        "selected_features": selected_features,
        "removed_features": removed_features,
        "feature_importance": [
            {
                "feature": row["Feature"],
                "importance": float(row["Importance"]),
            }
            for _, row in importance_df.iterrows()
        ],
        "simulation_points": [float(value) for value in simulation_points.tolist()],
        "simulation_summary": {
            "mean_price": float(simulation_points.mean()),
            "best_case": float(simulation_points.max()),
            "worst_case": float(simulation_points.min()),
            "days": days,
            "simulations": simulations,
        },
        "metrics": {
            "up_recall": float(refined_summary["up_recall"]),
            "up_f1": float(refined_summary["up_f1"]),
            "down_recall": float(refined_summary["down_recall"]),
            "down_f1": float(refined_summary["down_f1"]),
        },
    }


def predict_latest_trend(model, dataset, feature_columns):
    """
    Use the newest available feature row to predict the next day's trend.
    """
    latest_features = dataset[feature_columns].iloc[[-1]]
    latest_date = dataset.index[-1]

    predicted_trend = model.predict(latest_features)[0]
    probabilities = model.predict_proba(latest_features)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))
    confidence = max(probabilities)
    signal_strength = "Strong Signal" if confidence >= 0.60 else "Weak Signal"

    print("\n" + "=" * 60)
    print("LATEST PREDICTION")
    print("=" * 60)
    print(f"Latest feature date used: {latest_date.date()}")
    print(f"Predicted Trend: {predicted_trend}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Signal Strength: {signal_strength}")
    print("Class Probabilities:")
    for label in sorted(class_probabilities):
        print(f"  {label}: {class_probabilities[label]:.2%}")

    return predicted_trend, confidence, class_probabilities, signal_strength


def analyze_risk(data):
    """
    Estimate risk using the standard deviation of daily returns.

    Simple thresholds:
    - below 1%  -> Stable
    - 1% to 2%  -> Risky
    - above 2%  -> Highly Volatile
    """
    daily_returns = data["Close"].pct_change().dropna()
    volatility = daily_returns.std()

    if volatility < 0.01:
        risk_level = "Stable"
    elif volatility < 0.02:
        risk_level = "Risky"
    else:
        risk_level = "Highly Volatile"

    print("\n" + "=" * 60)
    print("RISK ANALYSIS")
    print("=" * 60)
    print(f"Daily Volatility (std of returns): {volatility:.2%}")
    print(f"Risk Level: {risk_level}")

    return volatility, risk_level


def monte_carlo_simulation(data, simulations=100, days=30):
    """
    Optional simple Monte Carlo simulation.
    It uses historical mean return and volatility to simulate future prices.

    This is a basic educational example, not a professional forecasting engine.
    """
    daily_returns = data["Close"].pct_change().dropna()
    mean_return = daily_returns.mean()
    volatility = daily_returns.std()
    last_price = data["Close"].iloc[-1]

    simulated_final_prices = []

    for _ in range(simulations):
        price = last_price
        for _ in range(days):
            random_return = np.random.normal(mean_return, volatility)
            price *= 1 + random_return
        simulated_final_prices.append(price)

    simulated_final_prices = np.array(simulated_final_prices)

    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION (OPTIONAL)")
    print("=" * 60)
    print(f"Simulations Run: {simulations}")
    print(f"Days Simulated: {days}")
    print(f"Average Simulated Price after {days} days: {simulated_final_prices.mean():.2f}")
    print(f"Best Case (approx): {simulated_final_prices.max():.2f}")
    print(f"Worst Case (approx): {simulated_final_prices.min():.2f}")

    # We import matplotlib because the user requested it.
    # For this console-only project, we do not display a chart.
    plt.close("all")

    return simulated_final_prices


def compare_accuracy_results(previous_accuracy, baseline_accuracy, refined_accuracy):
    """
    Compare the newest model accuracy against both the previous run and
    the current baseline before feature selection.
    """
    previous_change = refined_accuracy - previous_accuracy
    baseline_change = refined_accuracy - baseline_accuracy

    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)
    print(f"Previous recorded accuracy: {previous_accuracy:.2%}")
    print(f"Current baseline accuracy:   {baseline_accuracy:.2%}")
    print(f"Refined model accuracy:      {refined_accuracy:.2%}")
    print(f"Change vs previous result:   {previous_change:+.2%}")
    print(f"Change after feature selection: {baseline_change:+.2%}")


def interpret_results(accuracy, confidence, risk_level, signal_strength):
    """
    Print a short beginner-friendly interpretation of model quality and reliability.
    """
    if accuracy >= 0.70:
        accuracy_message = "The model is performing reasonably well for a simple binary stock trend classifier."
        reliability_message = "Its predictions may be useful as a rough signal, but they should not be trusted alone."
    elif accuracy >= 0.50:
        accuracy_message = "The model has moderate accuracy and is only somewhat better than guessing in some cases."
        reliability_message = "Its predictions are not highly reliable and should be treated with caution."
    else:
        accuracy_message = "The model accuracy is low, which means it struggles to identify the correct trend."
        reliability_message = "Its predictions are not reliable enough for decision-making."

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"Model Accuracy Interpretation: {accuracy_message}")
    print(f"Prediction Reliability: {reliability_message}")
    print(f"Latest Prediction Confidence: {confidence:.2%}")
    print(f"Signal Strength: {signal_strength}")
    print(f"Risk Context: The stock is currently classified as '{risk_level}' based on recent volatility.")


def main():
    build_prediction_result(ticker="AAPL", period="5y", n_splits=5, simulations=100, days=30, verbose=True)


if __name__ == "__main__":
    main()
