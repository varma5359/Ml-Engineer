# Day 57: Time-Series Forecasting Fundamentals

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Beginner to Intermediate  
**Time:** 2–3 hours  
**Approved mini-project:** Monthly Sales Forecast

## 1. Learning objectives
- Explain time-series data and forecasting.
- Identify trend, seasonality, cycles and noise.
- Split time-series data chronologically.
- Build naive baselines and lag-feature models.
- Evaluate forecasts with appropriate metrics.

## 2. What is time-series data?
A time series is a sequence of observations recorded over time: daily sales, monthly demand, hourly electricity usage or weekly traffic.

A record normally has a timestamp, measured value and optional explanatory features. Forecasting uses historical information to estimate future values.

## 3. Common patterns

| Component | Meaning | Example |
|---|---|---|
| Trend | Long-term movement | Sales rising over years |
| Seasonality | Repeating pattern at known interval | December sales peak |
| Cycle | Longer-term fluctuation without fixed interval | Economic cycles |
| Noise | Irregular variation | Unexpected disruption |

## 4. Why chronological splitting matters
Random splitting may place future observations in training and earlier observations in testing. That does not represent real forecasting.

Use:
- Earlier dates for training.
- Later dates for validation.
- Most recent untouched dates for testing.

Use time-series cross-validation when appropriate. All features must be available at the forecast timestamp.

## 5. Create monthly sales data
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dates = pd.date_range("2021-01-01", periods=48, freq="MS")
rng = np.random.default_rng(42)
trend = np.arange(48) * 4
seasonality = 30 * np.sin(2 * np.pi * np.arange(48) / 12)
noise = rng.normal(0, 10, 48)
sales = 200 + trend + seasonality + noise

df = pd.DataFrame({"date": dates, "sales": sales})
print(df.head())
```

**Expected output:** Five monthly rows with date and sales columns. The synthetic series contains trend, seasonality and noise.

## 6. Visualize
```python
plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["sales"], marker="o")
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()
```

Inspect the direction, repeating pattern, unusual values and changes in variability.

## 7. Naive baseline
A naive forecast uses the latest observed value for the next period.

```python
split_index = 36
train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

last_value = train["sales"].iloc[-1]
naive_forecast = np.repeat(last_value, len(test))
print(naive_forecast[:3])
```

**Expected output:** The last training sales value repeated for each test month.

A seasonal naive baseline for monthly annual seasonality uses the last 12 training observations for the next 12 months, when that alignment is valid.

## 8. Evaluate forecast error
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test["sales"], naive_forecast)
rmse = mean_squared_error(test["sales"], naive_forecast) ** 0.5
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
```

MAE averages absolute errors. RMSE penalizes large errors more strongly. MAPE can be unstable when actual values are zero or near zero.

## 9. Lag features
A lag feature uses earlier observations as model inputs.

```python
supervised = df.copy()
supervised["lag_1"] = supervised["sales"].shift(1)
supervised["lag_12"] = supervised["sales"].shift(12)
supervised["month"] = supervised["date"].dt.month
supervised = supervised.dropna().reset_index(drop=True)
print(supervised.head())
```

- lag_1: previous month's sales.
- lag_12: sales from the same month one year earlier.
- Shift prevents current target from being used as an input.

## 10. Train a basic lag-feature model
```python
from sklearn.ensemble import RandomForestRegressor

cutoff = int(len(supervised) * 0.8)
train_ml = supervised.iloc[:cutoff]
test_ml = supervised.iloc[cutoff:]
features = ["lag_1", "lag_12", "month"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(train_ml[features], train_ml["sales"])
pred = model.predict(test_ml[features])
print("MAE:", mean_absolute_error(test_ml["sales"], pred))
```

**Important limitation:** This one-step illustration uses observed lag values in held-out rows. A real multi-step forecast must generate unknown future lags recursively or use a direct multi-horizon approach.

## 11. Approved mini-project — Monthly Sales Forecast

**Scenario:** A retailer needs a monthly forecast to support inventory and staffing.

**Tasks**
1. Load date and sales columns.
2. Parse, sort and validate dates.
3. Inspect missing months and duplicate dates.
4. Plot the series and describe patterns.
5. Create chronological train/validation/test periods.
6. Build naive and seasonal-naive baselines.
7. Build at least one past-only lag-feature model.
8. Compare MAE and RMSE on the same horizon.
9. Plot actual versus forecast and document limitations.

**Submission:** Clean series, forecast plot, metric comparison and recommendation.

## 12. Common mistakes
- Randomly splitting time-series rows.
- Using future values in lag features.
- Ignoring missing periods.
- Comparing models over different horizons.
- Using MAPE with zero or near-zero actuals.
- Assuming a trend continues indefinitely.

## 13. Completion checklist
- [ ] Explain trend, seasonality, cycle and noise.
- [ ] Create chronological split.
- [ ] Build naive baseline.
- [ ] Create lag features safely.
- [ ] Evaluate with MAE and RMSE.
- [ ] Complete Monthly Sales Forecast.

**Next lesson:** Day 58 — Time-series decomposition and feature creation.
