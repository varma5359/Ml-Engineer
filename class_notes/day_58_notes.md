# Day 58: Time-Series Decomposition and Feature Creation

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Intermediate  
**Time:** 2–3 hours  
**Approved mini-project:** Demand Trend Analyzer

## 1. Learning objectives
- Explain time-series decomposition.
- Distinguish additive and multiplicative models.
- Create rolling, lag and calendar features.
- Avoid future-information leakage.
- Interpret trend and residual components.

## 2. What is decomposition?
Decomposition separates an observed series into components.

- **Additive:** observed = trend + seasonal + residual. Useful when seasonal variation is roughly constant.
- **Multiplicative:** observed = trend × seasonal × residual. Useful when seasonal variation changes with the series level; values should generally be positive.

Decomposition describes patterns; it does not prove their causes.

## 3. Create monthly demand data
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dates = pd.date_range("2020-01-01", periods=60, freq="MS")
rng = np.random.default_rng(7)
trend = np.linspace(100, 180, 60)
season = 1 + 0.20 * np.sin(2 * np.pi * np.arange(60) / 12)
demand = trend * season + rng.normal(0, 5, 60)
df = pd.DataFrame({"date": dates, "demand": demand}).set_index("date")
print(df.head())
```

## 4. Plot the original series
```python
df["demand"].plot(figsize=(10, 4), title="Monthly Demand")
plt.ylabel("Units")
plt.show()
```

Look for trend, repeating patterns, unusual observations and changes in variation.

## 5. Decompose the series
Install statsmodels if needed:
```bash
python -m pip install statsmodels
```

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(
    df["demand"], model="multiplicative", period=12
)
decomposition.plot()
plt.tight_layout()
plt.show()
```

**Expected output:** Observed, trend, seasonal and residual panels.

**Explanation:** Period 12 represents a 12-observation seasonal cycle for monthly annual seasonality. Enough observations are needed; edge trend values may be missing because neighbouring observations are required.

## 6. Rolling statistics
```python
df["rolling_mean_3"] = df["demand"].rolling(window=3).mean()
print(df[["demand", "rolling_mean_3"]].head(6))
```

The first two rolling means are missing because three observations are needed.

```python
df["rolling_std_3"] = df["demand"].rolling(window=3).std()
print(df[["demand", "rolling_std_3"]].tail())
```

Rolling standard deviation summarizes recent variability.

## 7. Create leakage-safe features
```python
features = df.copy()
features["lag_1"] = features["demand"].shift(1)
features["lag_3"] = features["demand"].shift(3)
features["lag_12"] = features["demand"].shift(12)
features["rolling_mean_3"] = (
    features["demand"].shift(1).rolling(window=3).mean()
)
features["month"] = features.index.month
features["quarter"] = features.index.quarter
features = features.dropna()
print(features.head())
```

**Important:** Shift before rolling for next-period prediction. This ensures the feature at time t uses only values before t.

## 8. Calendar features
Examples:
- Month and quarter.
- Day of week for daily data.
- Week of year.
- Known holiday or promotion indicator.
- Days until a planned event.

Use only calendar/event information genuinely known at prediction time.

## 9. Approved mini-project — Demand Trend Analyzer

**Scenario:** A distribution team wants to understand demand patterns and prepare forecasting features.

**Tasks**
1. Load date and demand columns.
2. Parse and sort dates; check frequency and missing periods.
3. Plot observed demand.
4. Choose additive or multiplicative decomposition and justify it.
5. Plot and interpret trend, seasonal and residual components.
6. Create lag, rolling and calendar features.
7. Verify features use only past information.
8. Write a short interpretation and limitations report.

**Suggested structure**
```text
demand_trend_analyzer/
├── data/
├── notebooks/demand_analysis.ipynb
├── reports/demand_trends.md
└── README.md
```

## 10. Common mistakes
- Using period 12 for every data frequency.
- Using multiplicative decomposition with zero/negative values.
- Calculating rolling means including the target period.
- Ignoring missing periods.
- Calling a seasonal pattern a causal effect.
- Randomly splitting a forecasting dataset.

## 11. Completion checklist
- [ ] Explain additive and multiplicative decomposition.
- [ ] Interpret decomposition plots.
- [ ] Create lag and rolling features.
- [ ] Explain why shift is required.
- [ ] Complete Demand Trend Analyzer.

**Next lesson:** Day 59 — Recommendation system fundamentals.
