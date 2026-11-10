# Day 11: Outlier Detection and Treatment

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define an outlier and explain why it needs investigation.
- Detect unusual values using domain rules, IQR and z-scores.
- Distinguish valid rare events from data errors.
- Choose whether to retain, correct, cap or transform a value.

## 1. What Is an Outlier?
An **outlier** is an observation that is unusually far from the pattern of the rest of the data. An outlier is not automatically a mistake.

Three possibilities:
1. Data-entry error, such as entering 900000 instead of 9000.
2. Rare but valid event, such as a high-value insurance claim.
3. A real process change, such as a new premium customer segment.

## 2. Detect Outliers
### Method 1 — Domain rules
Use known business limits. Examples: negative age may be invalid; a temperature outside a sensor's physical range may need checking; a transaction far above usual value may require investigation but may still be valid.

### Method 2 — IQR
IQR = Q3 - Q1. Lower fence = Q1 - 1.5 × IQR. Upper fence = Q3 + 1.5 × IQR.

```python
import pandas as pd
amount = pd.Series([100, 120, 130, 125, 140, 150, 900])
q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(amount[(amount < lower) | (amount > upper)])
```
The output flags the unusually high value for this sample.

### Method 3 — Z-score
A z-score tells how many standard deviations a value is from the mean. It is most useful when the distribution is reasonably compatible with the assumptions behind this measure.

```python
from scipy.stats import zscore
values = [10, 11, 10, 12, 11, 50]
print(zscore(values))
```

## 3. Three Practical Detection Examples
1. Use IQR to screen unusually high order values.
2. Use domain limits to flag negative product quantities when they are not allowed by the business.
3. Use a box plot or histogram to investigate an unusual salary distribution.

## 4. Treatment Options
- Correct a confirmed source error.
- Retain a valid rare observation.
- Cap extreme values only when justified.
- Transform a skewed positive feature when appropriate.
- Use robust algorithms or metrics when extremes are valid.

Do not automatically delete all IQR-flagged records. Learn thresholds from training data only when they are part of model preprocessing.

## 5. Scenario Questions
1. A fraud dataset contains very large transactions. Should you remove them as outliers? No; they may be important fraud examples.
2. A sensor reports -500°C although the device cannot measure that value. What first? Verify the sensor/source and unit conversion.
3. A few executive salaries are far above the rest. Which centre may better describe a typical employee? Median, while still reporting the distribution.
