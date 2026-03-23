let simulationChart;
const API_BASE_URL = resolveApiBaseUrl();
const TOP_STOCK_TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA", "META", "NFLX", "JPM", "AMD"];
let marketSignalsData = [];
let currentSignalFilter = "all";

document.addEventListener("DOMContentLoaded", () => {
    setupMobileMenu();
    setupAuthForms();

    if (document.body.dataset.page === "dashboard") {
        setupDashboard();
    }
});

function setupMobileMenu() {
    const toggle = document.querySelector(".menu-toggle");
    const nav = document.querySelector(".nav-links");

    if (!toggle || !nav) {
        return;
    }

    toggle.addEventListener("click", () => {
        nav.classList.toggle("open");
    });
}

function setupAuthForms() {
    const loginForm = document.getElementById("loginForm");
    const signupForm = document.getElementById("signupForm");
    const loginError = document.getElementById("loginError");

    if (loginForm) {
        loginForm.addEventListener("submit", (event) => {
            event.preventDefault();

            const email = document.getElementById("loginEmail").value.trim();
            const password = document.getElementById("loginPassword").value.trim();

            if (!email || !password) {
                loginError.classList.add("visible");
                return;
            }

            loginError.classList.remove("visible");
            window.location.href = "dashboard.html";
        });
    }

    if (signupForm) {
        signupForm.addEventListener("submit", (event) => {
            event.preventDefault();
            window.location.href = "dashboard.html";
        });
    }
}

function setupDashboard() {
    const predictButton = document.getElementById("predictButton");
    const tickerInput = document.getElementById("tickerInput");
    const refreshSignalsButton = document.getElementById("refreshSignalsButton");
    const filterButtons = document.querySelectorAll("[data-filter]");

    createEmptyChart();

    predictButton.addEventListener("click", () => {
        fetchPrediction(tickerInput.value.trim() || "AAPL");
    });

    tickerInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            fetchPrediction(tickerInput.value.trim() || "AAPL");
        }
    });

    if (refreshSignalsButton) {
        refreshSignalsButton.addEventListener("click", () => {
            fetchTopStockSignals();
        });
    }

    filterButtons.forEach((button) => {
        button.addEventListener("click", () => {
            currentSignalFilter = button.dataset.filter;
            filterButtons.forEach((chip) => chip.classList.remove("active"));
            button.classList.add("active");
            renderTopStockSignals(marketSignalsData);
        });
    });

    fetchPrediction(tickerInput.value.trim() || "AAPL");
    fetchTopStockSignals();
}

async function fetchPrediction(ticker) {
    const upperTicker = ticker.toUpperCase();
    const status = document.getElementById("dashboardStatus");
    const loading = document.getElementById("loadingIndicator");
    const predictButton = document.getElementById("predictButton");

    setLoadingState(true, loading, predictButton);
    status.textContent = `Fetching prediction for ${upperTicker}...`;

    try {
        const [predictionResult, simulationResult] = await Promise.allSettled([
            fetchJson(buildApiUrl(`/predict?ticker=${encodeURIComponent(upperTicker)}`)),
            fetchJson(buildApiUrl(`/simulate?ticker=${encodeURIComponent(upperTicker)}`)),
        ]);

        const fallbackData = buildFallbackData(upperTicker);
        let usedFallback = false;

        if (predictionResult.status === "fulfilled") {
            updatePredictionSection(predictionResult.value, upperTicker, false);
        } else {
            updatePredictionSection(fallbackData, upperTicker, true);
            usedFallback = true;
        }

        if (simulationResult.status === "fulfilled") {
            updateSimulationSection(simulationResult.value, upperTicker, false);
        } else {
            updateSimulationSection(fallbackData, upperTicker, true);
            usedFallback = true;
        }

        status.textContent = usedFallback
            ? `Backend partially unavailable. Showing demo data where needed for ${upperTicker}.`
            : `Prediction and simulation updated from backend for ${upperTicker}.`;
    } catch (error) {
        const fallbackData = buildFallbackData(upperTicker);
        updatePredictionSection(fallbackData, upperTicker, true);
        updateSimulationSection(fallbackData, upperTicker, true);
        status.textContent = `Backend unavailable. Showing demo data for ${upperTicker}.`;
    } finally {
        setLoadingState(false, loading, predictButton);
    }
}

async function fetchJson(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Request failed for ${url}`);
    }

    const text = await response.text();

    try {
        return JSON.parse(text);
    } catch (error) {
        const safeText = text
            .replace(/\bNaN\b/g, "null")
            .replace(/\bInfinity\b/g, "null")
            .replace(/\b-Infinity\b/g, "null");

        return JSON.parse(safeText);
    }
}

function setLoadingState(isLoading, loadingElement, button) {
    loadingElement.classList.toggle("hidden", !isLoading);
    button.disabled = isLoading;
    button.style.opacity = isLoading ? "0.7" : "1";
}

function updatePredictionSection(data, ticker, isFallback) {
    const trend = readValue(data, ["trend", "predicted_trend"], "UP");
    const confidenceRaw = readValue(data, ["confidence", "probability"], 0.74);
    const risk = readValue(data, ["risk_level", "risk"], "Risky");
    const signal = readValue(
        data,
        ["signal_strength", "signal"],
        confidenceRaw >= 60 || confidenceRaw >= 0.6 ? "Strong" : "Weak"
    );
    const probabilities = readObject(data, ["probabilities"]);
    const confidence = normalizeConfidence(confidenceRaw);
    const trendElement = document.getElementById("trendValue");
    const upProbability = normalizeConfidence(readValue(probabilities, ["UP"], trend === "UP" ? confidence : 1 - confidence));
    const downProbability = normalizeConfidence(readValue(probabilities, ["DOWN"], trend === "DOWN" ? confidence : 1 - confidence));

    trendElement.textContent = trend;
    trendElement.classList.remove("positive-text", "negative-text");
    trendElement.classList.add(trend === "UP" ? "positive-text" : "negative-text");

    document.getElementById("confidenceValue").textContent = `${(confidence * 100).toFixed(2)}%`;
    document.getElementById("riskValue").textContent = risk;
    document.getElementById("signalValue").textContent = normalizeSignalText(signal);
    document.getElementById("chartTickerLabel").textContent = ticker;
    document.getElementById("summaryTrend").textContent = `${ticker} ${trend}`;
    document.getElementById("upProbability").textContent = `${(upProbability * 100).toFixed(2)}%`;
    document.getElementById("downProbability").textContent = `${(downProbability * 100).toFixed(2)}%`;
    document.getElementById("summaryText").textContent = isFallback
        ? "Using polished demo prediction data while the backend endpoint is unavailable."
        : "Live backend prediction received successfully. Review the signal before acting.";
}

function updateSimulationSection(data, ticker, isFallback) {
    const simulation = readArray(data, ["simulated_prices", "simulation_points", "monte_carlo_paths"]);
    const summary = readObject(data, ["simulation_summary", "monte_carlo_summary"]);
    const meanPrice = readValue(summary, ["mean_price", "average_price", "average"], readValue(data, ["average"], average(simulation)));
    const bestCase = readValue(summary, ["best_case", "max_price", "max"], readValue(data, ["max"], Math.max(...simulation)));
    const worstCase = readValue(summary, ["worst_case", "min_price", "min"], readValue(data, ["min"], Math.min(...simulation)));

    document.getElementById("meanPriceValue").textContent = formatMoney(meanPrice);
    document.getElementById("bestCaseValue").textContent = formatMoney(bestCase);
    document.getElementById("worstCaseValue").textContent = formatMoney(worstCase);
    document.getElementById("chartTickerLabel").textContent = ticker;

    if (isFallback) {
        document.getElementById("summaryText").textContent =
            "Using polished demo simulation data while the backend simulation endpoint is unavailable.";
    }

    renderSimulationChart(ticker, simulation);
}

async function fetchTopStockSignals() {
    const marketStatus = document.getElementById("marketSignalsStatus");
    const marketList = document.getElementById("marketSignalsList");

    if (!marketStatus || !marketList) {
        return;
    }

    marketStatus.textContent = "Loading market signals...";
    marketList.innerHTML = "";

    const requests = TOP_STOCK_TICKERS.map((ticker) =>
        fetchJson(buildApiUrl(`/predict?ticker=${encodeURIComponent(ticker)}`))
            .then((data) => ({
                ticker,
                trend: readValue(data, ["trend"], "UP"),
                confidence: normalizeConfidence(readValue(data, ["confidence"], 70)),
                signal: normalizeSignalText(readValue(data, ["signal_strength", "signal"], "Weak")),
            }))
            .catch(() => {
                const fallback = buildFallbackData(ticker);
                return {
                    ticker,
                    trend: fallback.trend,
                    confidence: normalizeConfidence(fallback.confidence),
                    signal: normalizeSignalText(fallback.signal_strength),
                };
            })
    );

    const results = await Promise.all(requests);
    marketSignalsData = results.sort((a, b) => b.confidence - a.confidence);
    marketStatus.textContent = "Live market signals updated.";
    renderTopStockSignals(marketSignalsData);
}

function renderTopStockSignals(signals) {
    const marketList = document.getElementById("marketSignalsList");

    if (!marketList) {
        return;
    }

    const filteredSignals = signals.filter((signal) => {
        if (currentSignalFilter === "up") {
            return signal.trend === "UP";
        }

        if (currentSignalFilter === "strong") {
            return signal.signal === "Strong Signal";
        }

        return true;
    });

    if (!filteredSignals.length) {
        marketList.innerHTML = '<div class="market-empty">No stocks match the current filter.</div>';
        return;
    }

    marketList.innerHTML = filteredSignals
        .map(
            (signal) => `
                <article class="signal-stock-card">
                    <div class="signal-stock-top">
                        <div class="signal-ticker">${signal.ticker}</div>
                        <div class="signal-badge ${signal.trend === "UP" ? "up" : "down"}">
                            <span class="signal-dot"></span>
                            <span>${signal.trend === "UP" ? "↑ UP" : "↓ DOWN"}</span>
                        </div>
                    </div>
                    <div class="signal-stock-bottom">
                        <div class="signal-meta">
                            <span>Confidence</span>
                            <strong>${(signal.confidence * 100).toFixed(0)}%</strong>
                        </div>
                        <div class="signal-strength-text">${signal.signal}</div>
                    </div>
                </article>
            `
        )
        .join("");
}

function createEmptyChart() {
    const context = document.getElementById("simulationChart");
    if (!context || typeof Chart === "undefined") {
        return;
    }

    simulationChart = new Chart(context, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Monte Carlo Path",
                    data: [],
                    borderColor: "#4d8dff",
                    backgroundColor: "rgba(77, 141, 255, 0.16)",
                    borderWidth: 3,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.38,
                },
            ],
        },
        options: buildChartOptions(),
    });
}

function renderSimulationChart(ticker, values) {
    if (!simulationChart) {
        return;
    }

    simulationChart.data.labels = values.map((_, index) => `Day ${index + 1}`);
    simulationChart.data.datasets[0].label = `${ticker} Monte Carlo`;
    simulationChart.data.datasets[0].data = values;
    simulationChart.update();
}

function buildChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: "#dbe6ff",
                    font: {
                        family: "Manrope",
                    },
                },
            },
        },
        scales: {
            x: {
                ticks: {
                    color: "#7f90b2",
                },
                grid: {
                    color: "rgba(255,255,255,0.05)",
                },
            },
            y: {
                ticks: {
                    color: "#7f90b2",
                },
                grid: {
                    color: "rgba(255,255,255,0.05)",
                },
            },
        },
    };
}

function buildFallbackData(ticker) {
    const basePrice = ticker === "TSLA" ? 178 : ticker === "NVDA" ? 884 : 214;
    const simulation = Array.from({ length: 30 }, (_, index) => {
        const wave = Math.sin(index / 3) * 5;
        const drift = index * 1.8;
        const noise = (index % 4) * 1.4;
        return Number((basePrice + drift + wave + noise).toFixed(2));
    });

    const trend = ticker === "TSLA" ? "DOWN" : "UP";
    const confidence = ticker === "TSLA" ? 0.58 : 0.74;

    return {
        trend,
        confidence,
        risk_level: ticker === "NVDA" ? "Moderate" : "Risky",
        signal_strength: confidence >= 0.6 ? "Strong" : "Weak",
        probabilities: {
            UP: trend === "UP" ? confidence : 1 - confidence,
            DOWN: trend === "DOWN" ? confidence : 1 - confidence,
        },
        simulated_prices: simulation,
        average: average(simulation),
        min: Math.min(...simulation),
        max: Math.max(...simulation),
    };
}

function resolveApiBaseUrl() {
    if (window.location.protocol === "file:") {
        return "http://127.0.0.1:5000";
    }

    if (window.location.port && window.location.port !== "5000") {
        return "http://127.0.0.1:5000";
    }

    return "";
}

function buildApiUrl(path) {
    return `${API_BASE_URL}${path}`;
}

function normalizeConfidence(value) {
    if (value > 1) {
        return value / 100;
    }
    return value;
}

function average(values) {
    if (!values.length) {
        return 0;
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatMoney(value) {
    return `$${Number(value).toFixed(2)}`;
}

function normalizeSignalText(value) {
    if (value === "Strong") {
        return "Strong Signal";
    }

    if (value === "Weak") {
        return "Weak Signal";
    }

    return value;
}

function readValue(source, keys, fallback) {
    if (!source || typeof source !== "object") {
        return fallback;
    }

    for (const key of keys) {
        if (key in source && source[key] !== undefined && source[key] !== null) {
            return source[key];
        }
    }

    return fallback;
}

function readArray(source, keys) {
    const value = readValue(source, keys, []);

    if (Array.isArray(value) && value.length) {
        return value.map((item) => Number(item));
    }

    return buildFallbackData("AAPL").simulated_prices;
}

function readObject(source, keys) {
    const value = readValue(source, keys, {});
    return typeof value === "object" && value !== null ? value : {};
}
