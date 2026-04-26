let priceChart = null;
let isAnalyzing = false;

function setLoading(isLoading, ticker = "") {
  const runButton = document.getElementById("runButton");
  const runButtonText = document.getElementById("runButtonText");
  const runSpinner = document.getElementById("runSpinner");

  runButton.disabled = isLoading;
  runButtonText.textContent = isLoading
    ? `Running${ticker ? ` ${ticker}` : ""}...`
    : "Run Model";
  runSpinner.classList.toggle("loading", isLoading);
}

function getTrendClass(trend) {
  if (trend === "UP") return "trend-up";
  if (trend === "DOWN") return "trend-down";
  return "trend-stable";
}

function setStatus(message, isError = false) {
  const status = document.getElementById("statusText");
  status.textContent = message;
  status.style.color = isError ? "#ff8ca3" : "#8fa7d9";
}

function renderList(targetId, items) {
  const container = document.getElementById(targetId);
  container.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    container.appendChild(li);
  });
}

function updateMetrics(result) {
  document.getElementById("headerTicker").textContent = result.ticker;
  document.getElementById("asOfText").textContent = `As of ${result.as_of_date}`;

  const trendEl = document.getElementById("predictedTrend");
  trendEl.className = getTrendClass(result.predicted_trend);
  trendEl.textContent = result.predicted_trend;

  document.getElementById("confidenceText").textContent = `Confidence: ${result.confidence_score.toFixed(2)}%`;
  document.getElementById("backtestAcc").textContent = `${result.backtest_accuracy_score.toFixed(2)}%`;

  const macroF1 = result.backtest_summary.macro_f1_score ?? 0;
  document.getElementById("macroF1").textContent = `Macro F1: ${macroF1.toFixed(2)}%`;

  document.getElementById("riskScore").textContent = `${result.risk_score.toFixed(2)} / 100`;
  document.getElementById("volClass").textContent = `Volatility: ${result.volatility_classification}`;

  const lower = result.confidence_band.lower_return_pct.toFixed(2);
  const upper = result.confidence_band.upper_return_pct.toFixed(2);
  const median = result.confidence_band.median_return_pct.toFixed(2);
  document.getElementById("bandRange").textContent = `${lower}% to ${upper}%`;
  document.getElementById("bandMedian").textContent = `Median: ${median}%`;

  renderList("explanationList", result.explanation);
  renderList("failureList", result.failure_analysis_guidance);

  const simulation = result.simulation || {};
  document.getElementById("simStrategy").textContent = simulation.strategy_bias || "-";
  document.getElementById("simPosition").textContent = simulation.position_size
    ? `$${simulation.position_size.toLocaleString()}`
    : "-";

  const projectedPnl = Number(simulation.projected_pnl || 0);
  const simPnl = document.getElementById("simPnl");
  simPnl.textContent = simulation.projected_pnl ? `$${projectedPnl.toLocaleString()}` : "-";
  simPnl.className = projectedPnl >= 0 ? "trend-up" : "trend-down";

  if (simulation.worst_case_pnl !== undefined && simulation.best_case_pnl !== undefined) {
    document.getElementById("simRange").textContent = `$${Number(simulation.worst_case_pnl).toLocaleString()} to $${Number(simulation.best_case_pnl).toLocaleString()}`;
  } else {
    document.getElementById("simRange").textContent = "-";
  }

  document.getElementById("simMeta").textContent = simulation.position_fraction_pct
    ? `Capital $${Number(simulation.initial_capital).toLocaleString()} | Allocation ${Number(simulation.position_fraction_pct).toFixed(1)}% | Return ${Number(simulation.projected_return_pct).toFixed(2)}%`
    : "";
  document.getElementById("simStop").textContent = simulation.stop_loss_pct
    ? `SL: ${Number(simulation.stop_loss_pct).toFixed(2)}%`
    : "SL: -";
  document.getElementById("simTake").textContent = simulation.take_profit_pct
    ? `TP: ${Number(simulation.take_profit_pct).toFixed(2)}%`
    : "TP: -";

  document.getElementById("modelMeta").textContent =
    `Threshold ${Number(result.selected_stable_threshold).toFixed(4)} | Depth ${result.selected_model_params.max_depth} | Trees ${result.selected_model_params.n_estimators}`;
}

function renderChart(history) {
  const context = document.getElementById("priceChart");
  if (priceChart) {
    priceChart.destroy();
  }

  priceChart = new Chart(context, {
    type: "line",
    data: {
      labels: history.labels,
      datasets: [
        {
          label: "Close",
          data: history.close,
          borderColor: "#57a5ff",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
        },
        {
          label: "SMA 20",
          data: history.sma_20,
          borderColor: "#53f1dc",
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.25,
        },
        {
          label: "SMA 50",
          data: history.sma_50,
          borderColor: "#9caef7",
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#dce9ff" },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8ea7d6", maxTicksLimit: 8 },
          grid: { color: "rgba(96, 128, 190, 0.18)" },
        },
        y: {
          ticks: { color: "#8ea7d6" },
          grid: { color: "rgba(96, 128, 190, 0.18)" },
        },
      },
    },
  });
}

async function runAnalysis(event) {
  if (event) {
    event.preventDefault();
  }

  if (isAnalyzing) {
    return;
  }

  const ticker = document.getElementById("ticker").value.trim().toUpperCase();
  const period = document.getElementById("period").value;
  const interval = document.getElementById("interval").value;
  const capital = document.getElementById("capital").value;

  if (!ticker) {
    setStatus("Please enter a ticker symbol.", true);
    return;
  }

  isAnalyzing = true;
  setLoading(true, ticker);
  setStatus(`Running model for ${ticker}...`);

  try {
    const query = `ticker=${encodeURIComponent(ticker)}&period=${encodeURIComponent(period)}&interval=${encodeURIComponent(interval)}&capital=${encodeURIComponent(capital)}`;

    const [analysisResponse, historyResponse] = await Promise.all([
      fetch(`/api/analyze?${query}`),
      fetch(`/api/history?${query}`),
    ]);

    const analysisData = await analysisResponse.json();
    const historyData = await historyResponse.json();

    if (!analysisResponse.ok) {
      throw new Error(analysisData.error || "Analysis request failed.");
    }
    if (!historyResponse.ok) {
      throw new Error(historyData.error || "History request failed.");
    }

    updateMetrics(analysisData);
    renderChart(historyData);
    setStatus(`Model run complete for ${ticker}.`);
  } catch (error) {
    setStatus(error.message || "Failed to run analysis.", true);
  } finally {
    isAnalyzing = false;
    setLoading(false);
  }
}

document.getElementById("analyzeForm").addEventListener("submit", runAnalysis);
window.addEventListener("load", () => {
  runAnalysis();
});
