# Day 11: Outlier Detection and Treatment

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain what an outlier is.
- Distinguish an unusual observation from a confirmed data error.
- Inspect extreme values using sorting and descriptive statistics.
- Apply business-rule checks.
- Understand the IQR and z-score approaches.
- Visualize unusual values using box plots and histograms.
- Compare retaining, correcting, capping, transforming and excluding values.
- Avoid fitting outlier thresholds using held-out test information.
- Document and validate outlier treatment.

## 2. What is an outlier?

An **outlier** is an observation that is unusually far from the general pattern of the data.

Examples:
- A transaction amount is much higher than most other transactions.
- A delivery takes 900 minutes while most deliveries take less than an hour.
- A customer age is recorded as 250.
- A machine sensor reports a value outside its physical operating range.

An outlier is not automatically wrong. It may represent:
1. A data-entry or system error.
2. A rare but valid event.
3. A different population or customer segment.
4. A genuine process change.
5. A measurement problem.

**Important:** Detection methods flag unusual values for investigation. They do not know the business truth.

## 3. Create a practice dataset

### 3.1 Import libraries

### 3.1. Import Pandas and NumPy

```python
import pandas as pd
import numpy as np

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

Pandas is used for tabular inspection and NumPy for numerical operations.



### 3.2 Create a transaction amount series

### 3.2. Create a small dataset

```python
amounts = pd.Series([
    100, 110, 115, 120, 125, 130, 135, 140, 150, 900
])

print(amounts)
```

**Expected output**

```text
A Series containing ten transaction amounts
```

**Explanation**

The value 900 is much larger than the others. We will investigate it rather than immediately delete it.



### 3.3 Inspect count, mean and median

### 3.3. Compare central summaries

```python
print("Count:", amounts.count())
print("Mean:", amounts.mean())
print("Median:", amounts.median())
```

**Expected output**

```text
Count: 10; Mean: 202.5; Median: 127.5
```

**Explanation**

The high value increases the mean. The median remains closer to the central values.



### 3.4 Sort values

### 3.4. Sort amounts

```python
print(amounts.sort_values())
```

**Expected output**

```text
The amounts in ascending order
```

**Explanation**

Sorting makes it easier to inspect the lowest and highest observations.



### 3.5 Inspect the largest values

### 3.5. Use nlargest()

```python
print(amounts.nlargest(3))
```

**Expected output**

```text
The three largest values, including 900
```

**Explanation**

This is a simple exploratory technique for finding extreme observations.



### 3.6 Inspect the smallest values

### 3.6. Use nsmallest()

```python
print(amounts.nsmallest(3))
```

**Expected output**

```text
The three smallest values
```

**Explanation**

Very low values may also be unusual. Outlier investigation should not focus only on large numbers.



## 4. First method: business rules

A business rule is a domain-informed condition used to identify values that are impossible, invalid or require review.

Examples:
- Age cannot be negative.
- Quantity may need to be a positive whole number.
- A delivery timestamp cannot occur before order creation.
- A sensor cannot exceed its documented physical measurement range.

Business rules must be confirmed with the domain owner. A large transaction is not necessarily invalid just because it is unusual.

### 4.1 Create customer age data

### 4.1. Create age observations

```python
customers = pd.DataFrame({
    "customer_id": ["C01", "C02", "C03", "C04", "C05"],
    "age": [25, 34, 42, -5, 150]
})

print(customers)
```

**Expected output**

```text
A table containing ages 25, 34, 42, -5 and 150
```

**Explanation**

The negative age is likely invalid. Age 150 should be reviewed against the source and business rules.



### 4.2 Flag negative ages

### 4.2. Find ages below zero

```python
negative_age = customers[customers["age"] < 0]

print(negative_age)
```

**Expected output**

```text
The record with age -5
```

**Explanation**

This is a rule-based flag. It identifies records that violate the assumed age definition.



### 4.3 Flag ages above an agreed limit

### 4.3. Flag ages over an illustrative limit

```python
age_review = customers[customers["age"] > 120]

print(age_review)
```

**Expected output**

```text
The record with age 150
```

**Explanation**

The upper limit is illustrative. In production, agree on valid ranges for the specific population and application.



### 4.4 Create a combined review flag

### 4.4. Flag invalid or suspicious age values

```python
customers["age_review"] = (
    (customers["age"] < 0) |
    (customers["age"] > 120)
)

print(customers)
```

**Expected output**

```text
The table with True for the two records requiring review
```

**Explanation**

The code creates a flag rather than deleting records. Review flags preserve the evidence for investigation.



## 5. Second method: Interquartile Range (IQR)

The **Interquartile Range** describes the spread of the middle half of ordered values. It is the distance between the third quartile and first quartile.

A common exploratory rule flags values below the lower fence or above the upper fence. This rule is a screening method, not a universal definition of an error.

### 5.1 Calculate the first quartile

### 5.1. Calculate Q1

```python
q1 = amounts.quantile(0.25)

print(q1)
```

**Expected output**

```text
116.25
```

**Explanation**

Q1 is the 25th percentile in this sample. Pandas uses an interpolation method when a percentile falls between observations.



### 5.2 Calculate the third quartile

### 5.2. Calculate Q3

```python
q3 = amounts.quantile(0.75)

print(q3)
```

**Expected output**

```text
138.75
```

**Explanation**

Q3 is the 75th percentile in this sample.



### 5.3 Calculate IQR

### 5.3. Calculate the interquartile range

```python
iqr = q3 - q1

print(iqr)
```

**Expected output**

```text
26.25
```

**Explanation**

IQR describes the spread between the first and third quartiles.



### 5.4 Calculate lower and upper fences

### 5.4. Calculate screening boundaries

```python
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print(lower_fence)
print(upper_fence)
```

**Expected output**

```text
82.5 followed by 172.5
```

**Explanation**

These are common IQR screening boundaries. They should not be interpreted as business-valid limits automatically.



### 5.5 Flag values outside the fences

### 5.5. Identify IQR outliers

```python
iqr_outliers = amounts[
    (amounts < lower_fence) |
    (amounts > upper_fence)
]

print(iqr_outliers)
```

**Expected output**

```text
The value 900
```

**Explanation**

The value is flagged because it is above the upper fence.



### 5.6 Keep non-flagged observations for comparison

### 5.6. Create a non-outlier view

```python
within_fences = amounts[
    (amounts >= lower_fence) &
    (amounts <= upper_fence)
]

print(within_fences)
```

**Expected output**

```text
The values from 100 through 150
```

**Explanation**

This creates a filtered view for comparison. It does not mean the flagged value should be permanently removed.



### 5.7 Use a reusable IQR function

### 5.7. Create an IQR flagging function

```python
def find_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

flags = find_iqr_outliers(amounts)
print(flags)
```

**Expected output**

```text
A Boolean Series showing False for typical values and True for 900
```

**Explanation**

The function makes the check reusable. It returns flags, not a cleaned dataset, so the caller can decide what to do next.



## 6. Third method: z-score

A **z-score** describes how far a value is from the mean in standard-deviation units. It can help flag extreme values when its assumptions are suitable.

The approach can be sensitive to skewed distributions and extreme values because both the mean and standard deviation can be affected by the observations.

### 6.1 Calculate z-scores with SciPy

### 6.1. Import zscore

```python
from scipy.stats import zscore

values = pd.Series([10, 11, 10, 12, 11, 50])

scores = zscore(values)

print(scores.round(2))
```

**Expected output**

```text
A sequence of standardized scores, with the value 50 having the largest positive score
```

**Explanation**

SciPy's zscore function standardizes values relative to the supplied sample. The exact convention can be configured; inspect the function's settings if sample/population convention matters.



### 6.2 Flag large absolute z-scores

### 6.2. Apply an illustrative z-score threshold

```python
z_flags = abs(scores) > 2

print(z_flags)
```

**Expected output**

```text
A Boolean sequence flagging values more than two standard deviations from the mean
```

**Explanation**

A threshold such as two is a screening choice, not a universal rule. It may flag valid observations or miss unusual values in skewed data.



### 6.3 Inspect flagged observations

### 6.3. Display records flagged by z-score

```python
print(values[z_flags])
```

**Expected output**

```text
The value or values selected by the threshold
```

**Explanation**

Always inspect the original values and business context, not only the standardized score.



## 7. Compare IQR and z-score

| Consideration | IQR screening | Z-score screening |
|---|---|---|
| Uses | Quartiles | Mean and standard deviation |
| Sensitivity to extreme values | More robust in many cases | Can be strongly affected |
| Distribution | Useful for exploratory skewed data | Easier to interpret when distribution is reasonably well behaved |
| Output | Values outside quartile-based fences | Values far from the mean |
| Limitation | Can flag valid rare values | Threshold assumptions may be unsuitable |

Neither method proves that a value is erroneous. Combine statistical flags with domain knowledge and data-source investigation.

## 8. Visualize outliers

Visualizations help show where unusual observations sit relative to the rest of the data.

### 8.1 Create a histogram

### 8.1. Plot the transaction distribution

```python
import matplotlib.pyplot as plt

plt.hist(amounts, bins=10)
plt.xlabel("Transaction amount")
plt.ylabel("Number of transactions")
plt.title("Transaction amount distribution")
plt.show()
```

**Expected output**

```text
A histogram in which the high value stretches the horizontal range
```

**Explanation**

The extreme observation can compress the display of the majority of values. A histogram is a diagnostic view, not a cleaning rule.



### 8.2 Create a box plot

### 8.2. Plot a box plot

```python
plt.boxplot(amounts)
plt.ylabel("Transaction amount")
plt.title("Transaction amount box plot")
plt.show()
```

**Expected output**

```text
A box plot with the high value shown beyond the whisker
```

**Explanation**

The box plot uses quartiles and whiskers to display spread and potential outliers. Points beyond the whiskers require investigation.



### 8.3 Plot with a logarithmic scale for exploration

### 8.3. Use a log scale for highly skewed positive data

```python
plt.hist(amounts, bins=10)
plt.yscale("log")
plt.xlabel("Transaction amount")
plt.ylabel("Count (log scale)")
plt.title("Transaction amount distribution")
plt.show()
```

**Expected output**

```text
A histogram with a logarithmic vertical count scale
```

**Explanation**

A log scale can help display frequencies across different magnitudes. It changes visual interpretation, so label axes clearly and explain the scale.



## 9. Investigate the business meaning

Suppose the amount 900 was flagged by IQR. Ask:

1. Is the amount valid in the source system?
2. Is it a premium transaction or bulk order?
3. Does the transaction have a different currency or unit?
4. Is it a refund, adjustment or aggregated order?
5. Is the customer or product from a different segment?
6. Is there a known operational incident?
7. Is this type of transaction available at prediction time?

The answer determines the treatment.

## 10. Treatment options

### Option 1 — Retain the value

Keep the observation when it is valid and relevant to the population the model must handle.

### Option 2 — Correct the value

Correct it only when a verified source or business rule provides the true value. Do not guess a replacement.

### Option 3 — Remove the record

Removal may be justified for confirmed corruption or records outside the task's scope. Document the rule and measure the impact.

### Option 4 — Cap or winsorize

Capping replaces values beyond a chosen boundary with a boundary value. It can reduce the influence of extremes but also removes information about their magnitude.

### Option 5 — Transform the feature

A transformation such as a logarithm can reduce right skew for positive values. It changes the scale and requires careful handling of zero and negative values.

### Option 6 — Use a robust method

Some algorithms and summary measures are less sensitive to extreme values. Robustness does not remove the need to investigate data errors.

## 11. Capping example

### 11.1 Create a copy of the feature

### 11.1. Preserve the original amounts

```python
amounts_df = pd.DataFrame({"amount": amounts})

amounts_df["amount_original"] = amounts_df["amount"]

print(amounts_df.head())
```

**Expected output**

```text
A DataFrame with the amount and a preserved original-value column
```

**Explanation**

Keeping the original makes it possible to compare the treatment and audit which records changed.



### 11.2 Cap values at the IQR upper fence

### 11.2. Create a capped column

```python
amounts_df["amount_capped"] = amounts_df["amount"].clip(
    upper=upper_fence
)

print(amounts_df[["amount", "amount_capped"]])
```

**Expected output**

```text
A table where 900 is capped at 178.125 and other values remain unchanged
```

**Explanation**

The clip operation limits values above the chosen upper boundary. The boundary must be learned from training data only during model evaluation.



### 11.3 Compare the distributions

### 11.3. Compare original and capped summaries

```python
print(amounts_df[["amount", "amount_capped"]].describe())
```

**Expected output**

```text
Summary statistics for both columns
```

**Explanation**

Compare mean, maximum and spread to understand how capping changes the feature. Do not overwrite the original column until the treatment is approved.



## 12. Log transformation example

Log transformation can reduce the effect of large positive values when the data and modelling assumptions make it appropriate.

### 12.1 Check for non-positive values

### 12.1. Check the minimum value

```python
print(amounts.min())
```

**Expected output**

```text
100
```

**Explanation**

The natural logarithm is not defined for zero or negative real values. Check this before applying log transformation.



### 12.2 Apply log1p

### 12.2. Create a log-transformed feature

```python
amount_log = np.log1p(amounts)

print(amount_log.round(2))
```

**Expected output**

```text
A Series of transformed values, with the large value compressed
```

**Explanation**

`log1p` applies a log transformation after adding one, so it can handle zero. It still cannot handle negative values in the real-valued case.



### 12.3 Compare original and transformed ranges

### 12.3. Compare maximum values

```python
print("Original maximum:", amounts.max())
print("Transformed maximum:", amount_log.max())
```

**Expected output**

```text
Original maximum: 900, followed by the log-transformed maximum
```

**Explanation**

The transformed values are on a different scale. Explain that interpretation and ensure the same transformation is applied during inference.



## 13. Avoid leakage when learning thresholds

If the outlier boundary is estimated from the complete dataset before evaluation, information from validation or test observations influences preprocessing.

For a fair evaluation:
1. Split the data using the correct strategy.
2. Estimate learned thresholds from training data only.
3. Apply the training-derived rule to validation and test data.
4. Put preprocessing inside a pipeline where supported.
5. Evaluate the impact on the target and important segments.

### 13.1 Separate a small training and test sample

### 13.1. Split data for demonstration

```python
from sklearn.model_selection import train_test_split

train_values, test_values = train_test_split(
    amounts,
    test_size=0.3,
    random_state=42
)

print(len(train_values), len(test_values))
```

**Expected output**

```text
7 3
```

**Explanation**

This is a small demonstration. For real datasets, choose random, group-based or chronological splitting according to the prediction task.



### 13.2 Learn an upper fence from training data only

### 13.2. Calculate training-only IQR boundary

```python
train_q1 = train_values.quantile(0.25)
train_q3 = train_values.quantile(0.75)
train_iqr = train_q3 - train_q1
train_upper = train_q3 + 1.5 * train_iqr

print(train_upper)
```

**Expected output**

```text
A threshold calculated only from train_values
```

**Explanation**

Do not calculate this boundary from the full dataset before splitting. The test set should remain unseen during preprocessing decisions.



### 13.3 Apply the training threshold to test values

### 13.3. Flag test values using the learned threshold

```python
test_flags = test_values > train_upper

print(pd.DataFrame({
    "test_value": test_values,
    "flagged": test_flags
}))
```

**Expected output**

```text
Test values and their Boolean flags
```

**Explanation**

The test observations are checked against the training-derived threshold. Do not refit the threshold on the test values.



## 14. Common beginner mistakes

| Mistake | Why it is wrong | Better approach |
|---|---|---|
| Delete every flagged value | Valid rare cases may be removed | Investigate before treatment |
| Assume IQR proves an error | It is a statistical screening rule | Combine with domain validation |
| Use one threshold for every segment | Groups may have different distributions | Compare segments and business rules |
| Calculate thresholds on all data before split | Test information leaks into preprocessing | Learn thresholds from training data |
| Ignore negative or impossible values | Numeric type does not prove validity | Apply domain checks |
| Cap without preserving original values | Audit trail is lost | Keep original and transformed fields |
| Log-transform without checking zero/negative values | Transformation may fail or be undefined | Inspect domain and choose appropriate handling |
| Report only number of outliers | Does not show business impact | Report count, rate, segments and effect |
| Use a box plot as proof of invalidity | Whiskers are statistical conventions | Verify the source record |

## 15. Interview-ready explanations

**What is an outlier?**  
An outlier is an observation unusually far from the general pattern. It may be a data error, a valid rare event or a different segment. It should be investigated before treatment.

**What is the IQR method?**  
It uses the first and third quartiles to describe the middle half of the data. Values outside commonly used fences are flagged for review. These flags are not automatic deletion instructions.

**What is the difference between IQR and z-score?**  
IQR uses quartiles and is often more robust to extreme values. Z-score uses the mean and standard deviation and can be sensitive to skewness and extreme observations.

**How do you handle outliers in a production ML pipeline?**  
Define the rule using domain knowledge and training data, preserve original values for audit, implement the same transformation in the pipeline, validate impact across segments and monitor new incoming data.

## 16. Scenario questions

1. A transaction is 100 times larger than the usual amount. Do you delete it?  
   **Answer:** No. Verify whether it is valid, a bulk order, a different currency or a source error.

2. A negative age is found. What do you do?  
   **Answer:** Check the source and schema. If age in years cannot be negative, flag it as invalid and apply an approved correction or missing-data rule.

3. A model performs poorly on high-value customers. What would you investigate?  
   **Answer:** Compare data quality, sample size, feature distributions and error patterns for that segment. High-value customers may represent a distinct population.

4. The IQR method flags 8% of transactions. Is that automatically too many?  
   **Answer:** No. The appropriate rate depends on the distribution and business process. Inspect the flagged observations and compare with domain expectations.

5. Your test set contains a value above the training-derived cap. What do you do?  
   **Answer:** Apply the documented training-derived transformation consistently and monitor how often production values exceed the training range. Do not recalculate the threshold using the test set.

## 17. Lesson completion checklist

- [ ] I can explain an outlier.
- [ ] I can inspect extreme values.
- [ ] I can create domain-rule flags.
- [ ] I can calculate IQR boundaries.
- [ ] I can explain z-score screening.
- [ ] I can use histograms and box plots.
- [ ] I can compare treatment options.
- [ ] I can create a capped or transformed feature.
- [ ] I understand training-only threshold fitting.
- [ ] I can explain why a flagged value is not automatically an error.

**Next lesson:** Day 12 — Encoding Categorical Variables and Scaling Numerical Data.
