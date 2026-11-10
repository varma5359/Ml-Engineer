# Day 57: Interview Questions — Time-Series Forecasting Fundamentals

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is time-series forecasting?
**Answer:**
- It estimates future values using observations indexed by time.
- Examples include sales, demand, traffic and energy use.
- Time order must be preserved.
- Forecast horizon and data frequency should be defined.

### Q2. Explain trend, seasonality, cycle and noise.
**Answer:**
- Trend: long-term direction.
- Seasonality: repeating pattern at a known interval.
- Cycle: longer fluctuation without a fixed calendar period.
- Noise: irregular variation.

### Q3. Why should time-series data not be randomly split?
**Answer:**
- Random splitting can mix past and future observations.
- It does not simulate forecasting future values.
- Use chronological train, validation and test periods.
- Time-series cross-validation can evaluate multiple forecast origins.

### Q4. What is a naive forecast?
**Answer:**
- It uses the latest observed value as the next forecast.
- It is a simple baseline.
- It may work poorly when seasonality is strong.
- Complex models should be compared against it.

### Q5. What is a seasonal naive forecast?
**Answer:**
- It uses the value from the same seasonal period in the previous cycle.
- For monthly data it may use the same month last year.
- It is useful when seasonal patterns repeat.
- It assumes the previous seasonal cycle is informative.

### Q6. What is a lag feature?
**Answer:**
- A previous observation used as a predictor.
- lag_1 means the previous period.
- lag_12 in monthly data means twelve months earlier.
- Correct shifting prevents target leakage.

## 2. Metrics and validation

### Q7. Explain MAE and RMSE.
**Answer:**
- MAE is average absolute forecast error.
- RMSE squares errors before averaging and taking the square root.
- RMSE penalizes large errors more strongly.
- Both are in target units.

### Q8. When is MAPE misleading?
**Answer:**
- Actual values of zero make it undefined.
- Near-zero actuals can produce huge percentages.
- Use a metric suited to the target and business context.

### Q9. What is rolling-origin evaluation?
**Answer:**
- Train up to a point in time.
- Forecast a later period.
- Move the origin forward and repeat.
- It evaluates performance across multiple historical forecast points.

## 3. Scenario questions

### Q10. Training error is low but future-period error is high. What do you inspect?
**Answer:**
- Overfitting and leakage.
- Trend or seasonal shifts.
- Missing periods and outliers.
- Baseline performance.
- Feature availability and forecast horizon.

### Q11. Can actual sales from the first forecast month be used to predict the second?
**Answer:**
- Only if that actual value is available at the second forecast time.
- Otherwise future lags must be estimated recursively or by a direct multi-step method.
- Evaluation must simulate production information availability.

### Q12. How would you forecast monthly retail sales?
**Answer:**
1. Define horizon and frequency.
2. Validate monthly date coverage.
3. Inspect trend and seasonality.
4. Split chronologically.
5. Build naive baselines.
6. Engineer past-only features.
7. Compare models on the same horizon.
8. Report errors and limitations.

### Q13. What is time-series data leakage?
**Answer:**
- Using information not available at forecast time.
- Examples include future sales or future-derived rolling statistics.
- Define the forecast timestamp and available data.
- Build features using only information known then.

### Q14. How do you communicate forecasts to business users?
**Answer:**
- State horizon and units.
- Show actual history and forecast.
- Explain assumptions and uncertainty.
- Highlight known data limitations.
- Do not present forecasts as guarantees.

## 4. Rapid revision
- Preserve chronology.
- Define frequency and horizon.
- Establish naive baselines.
- Lag features use past information only.
- MAE and RMSE measure forecast error.
- Multi-step forecasts need realistic future-lag handling.
