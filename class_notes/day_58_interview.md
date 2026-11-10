# Day 58: Interview Questions — Time-Series Decomposition and Feature Creation

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is time-series decomposition?
**Answer:**
- It separates a series into trend, seasonal and residual components.
- It helps describe patterns in historical data.
- Additive and multiplicative are common forms.
- It does not prove the cause of a pattern.

### Q2. Compare additive and multiplicative decomposition.
**Answer:**
- Additive: observed = trend + seasonal + residual.
- Multiplicative: observed = trend × seasonal × residual.
- Additive suits roughly constant seasonal amplitude.
- Multiplicative suits seasonal amplitude that changes with level and generally positive data.

### Q3. What is a rolling mean?
**Answer:**
- It calculates an average over a moving window.
- A three-period rolling mean summarizes three observations.
- It smooths short-term fluctuations.
- For forecasting, shift the window to avoid target-period information.

### Q4. What is a lag feature?
**Answer:**
- It represents a previous observation.
- lag_1 is the previous period.
- lag_12 in monthly data is twelve periods earlier.
- It helps a model use historical information.

### Q5. Why shift before rolling?
**Answer:**
- An unshifted rolling statistic may include the current target.
- That target is not available when forecasting it.
- Shifting first ensures only earlier observations are used.
- It prevents target leakage.

### Q6. What are calendar features?
**Answer:**
- Timestamp-derived features such as month, quarter or weekday.
- They help capture calendar patterns.
- Known holidays and promotions can be useful.
- Only information available at prediction time should be included.

## 2. Scenario questions

### Q7. Demand rises every year and peaks in December. What patterns may exist?
**Answer:**
- An upward trend may exist.
- Annual seasonality may exist.
- Decomposition can visualize both.
- Business context is needed to understand why the peak occurs.

### Q8. Seasonal variation increases as demand increases. Which decomposition may fit?
**Answer:**
- Multiplicative decomposition may fit.
- Check that values are positive.
- Compare with additive decomposition and inspect residuals.
- Choose based on observed behavior.

### Q9. lag_1 is very predictive in training but fails in production. What do you check?
**Answer:**
- Lag alignment and forecast timestamp.
- Whether training used the current target accidentally.
- Whether production has the lag value available.
- Multi-step forecasting strategy.
- Changes in the demand process.

### Q10. A residual has a large spike. What does it mean?
**Answer:**
- The observation is not explained well by the estimated trend and seasonal components.
- It may be an unusual event, data issue or model limitation.
- Investigate source data and business events.
- Do not automatically label it as fraud or a cause.

## 3. Project questions

### Q11. How would you build a Demand Trend Analyzer?
**Answer:**
1. Validate and sort dates.
2. Check frequency and missing periods.
3. Plot the observed series.
4. Select decomposition type and period.
5. Interpret components.
6. Create shifted lags and rolling features.
7. Validate feature availability.
8. Summarize findings and limitations.

### Q12. Why can decomposition have missing values at the edges?
**Answer:**
- Trend estimation uses neighbouring observations.
- A centred moving window needs values on both sides.
- Edge observations may not have enough neighbours.
- This is expected in common classical decomposition methods.

### Q13. Can a rolling average be used as a forecast?
**Answer:**
- Yes, as a simple baseline when correctly aligned.
- Its performance depends on series behavior and forecast horizon.
- Compare it with naive and seasonal-naive baselines.
- Never use future observations in the window.

## 4. Rapid revision
- Decomposition: trend, seasonal and residual.
- Additive: components summed.
- Multiplicative: components multiplied.
- Lag: earlier observation.
- Rolling: moving-window statistic.
- Shift before rolling for next-period prediction.
