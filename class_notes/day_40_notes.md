# Day 40: Anomaly Detection Concepts

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain anomaly detection in simple language.
- Distinguish an anomaly from a data error.
- Understand supervised and unsupervised anomaly detection.
- Recognize common anomaly detection use cases.
- Train Isolation Forest using scikit-learn.
- Understand contamination and anomaly labels.
- Train Local Outlier Factor for novelty or outlier detection.
- Compare anomaly scores and flagged observations.
- Evaluate results using available labels or domain review.
- Explain operational risks of false alerts and missed anomalies.

## 2. What is anomaly detection?

**Anomaly detection** identifies observations that differ significantly from expected or common patterns in a dataset.

Examples:
- Unusual banking transactions.
- Unexpected network activity.
- Abnormal sensor readings.
- Sudden changes in machine vibration.
- Unusual customer account behaviour.
- Unexpected spikes in system latency.

An anomaly is not automatically an error or fraud. It is an observation that deserves investigation.

## 3. Anomaly, outlier and error

| Term | Meaning |
|---|---|
| Anomaly | Observation that differs from expected behaviour |
| Outlier | Observation unusually far from the general distribution |
| Data error | Incorrect value caused by collection, entry, parsing or system problems |
| Fraud | Deliberately deceptive activity, which may or may not appear statistically unusual |

A genuine high-value transaction may be unusual but valid. A fraud event may look normal in the available features. Always combine model outputs with business investigation.

## 4. Types of anomaly detection

### Point anomalies
One observation is unusual compared with the rest.

Example: A transaction much larger than typical customer transactions.

### Contextual anomalies
An observation is unusual in a particular context.

Example: Electricity usage that is normal in summer but unusual in winter.

### Collective anomalies
A group or sequence of observations is unusual together.

Example: Many small transactions occurring rapidly in a pattern that is unusual for an account.

## 5. Supervised and unsupervised anomaly detection

| Approach | Data requirement | Example |
|---|---|---|
| Supervised | Historical normal/anomaly labels | Fraud classifier |
| Unsupervised | No anomaly labels required | Isolation Forest |
| Semi-supervised / one-class | Mostly normal examples or a defined normal population | One-Class SVM |

Unsupervised anomaly detection is useful when labels are rare or unavailable, but its alerts need review.

## 6. Create a transaction dataset with unusual observations

### 6.1 Import libraries

### 6.1. Import NumPy, Pandas and plotting tools

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

These libraries create and inspect transaction-like numerical observations.



### 6.2 Generate normal transactions

### 6.2. Create typical transaction values

```python
rng = np.random.default_rng(42)

normal_amount = rng.normal(
    loc=1500,
    scale=300,
    size=250
)

normal_frequency = rng.normal(
    loc=4,
    scale=1,
    size=250
)

print(normal_amount[:5])
print(normal_frequency[:5])
```

**Expected output**

```text
Two arrays showing example transaction amounts and frequencies
```

**Explanation**

The values are simulated. Normal behaviour in a real system must be defined from appropriate historical data and business context.



### 6.3 Create unusual observations

### 6.3. Generate a few unusual transactions

```python
anomaly_amount = np.array([5000, 6500, 7200, 8000, 9000])
anomaly_frequency = np.array([12, 15, 18, 20, 25])

print(anomaly_amount)
print(anomaly_frequency)
```

**Expected output**

```text
Five large amounts and high frequencies
```

**Explanation**

These observations are deliberately far from the simulated normal pattern for demonstration.



### 6.4 Combine normal and unusual observations

### 6.4. Build the full feature matrix

```python
amount = np.concatenate([normal_amount, anomaly_amount])
frequency = np.concatenate([normal_frequency, anomaly_frequency])

df = pd.DataFrame({
    "transaction_amount": amount,
    "daily_transaction_count": frequency
})

print(df.tail())
```

**Expected output**

```text
The final five rows containing the unusual examples
```

**Explanation**

The anomaly examples are placed at the end for inspection, but the model will not be given their labels during unsupervised fitting.



### 6.5 Create teaching labels for later evaluation

### 6.5. Store hidden labels separately

```python
known_labels = np.array(
    [0] * len(normal_amount) + [1] * len(anomaly_amount)
)

print(pd.Series(known_labels).value_counts())
```

**Expected output**

```text
250 normal labels and five anomaly labels
```

**Explanation**

These labels are available only because this is a synthetic teaching example. Do not pass them to an unsupervised anomaly detector during fitting.



### 6.6 Visualize the observations

### 6.6. Plot transaction amount and frequency

```python
plt.scatter(
    df["transaction_amount"],
    df["daily_transaction_count"]
)
plt.xlabel("Transaction amount")
plt.ylabel("Daily transaction count")
plt.title("Transaction observations")
plt.show()
```

**Expected output**

```text
A scatter plot showing most observations in one region and a few far away
```

**Explanation**

The chart helps illustrate point anomalies. Real anomalies may be subtler or depend on customer, time and transaction context.



## 7. Prepare features for anomaly detection

### 7.1 Inspect feature ranges

### 7.1. Review feature summary

```python
print(df.describe().T)
```

**Expected output**

```text
Summary statistics for amount and frequency
```

**Explanation**

Feature scales differ. Scaling may be important for distance-based anomaly methods. Isolation Forest is generally less sensitive to feature scaling than distance-based methods.



### 7.2 Create the feature table

### 7.2. Select numerical inputs

```python
X = df[[
    "transaction_amount",
    "daily_transaction_count"
]]

print(X.head())
```

**Expected output**

```text
A two-column feature table
```

**Explanation**

The model receives the features only. known_labels remains separate for evaluation after fitting.



### 7.3 Scale the features for distance-based methods

### 7.3. Create standardized features

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])
```

**Expected output**

```text
The first five standardized observations
```

**Explanation**

This complete-data fit is for an exploratory unsupervised example. In a held-out evaluation, fit scaling on development data and transform held-out data with the fitted scaler.



## 8. Isolation Forest

**Isolation Forest** identifies unusual observations by isolating them through randomized feature splits.

Intuition:
- Common observations tend to require more splits to isolate.
- Unusual observations can often be isolated with fewer splits.
- The algorithm uses many randomized trees and combines their results.

Isolation Forest is a common starting point for tabular anomaly detection.

### 8.1 Import IsolationForest

### 8.1. Import the estimator

```python
from sklearn.ensemble import IsolationForest

print("IsolationForest imported")
```

**Expected output**

```text
IsolationForest imported
```

**Explanation**

The estimator is available from scikit-learn's ensemble module.



### 8.2 Initialize Isolation Forest

### 8.2. Create the anomaly detector

```python
isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

print(isolation_forest)
```

**Expected output**

```text
An IsolationForest configuration
```

**Explanation**

n_estimators controls the number of trees. contamination is an estimate of the expected anomaly proportion used to set the decision threshold; it is not a guarantee of the true anomaly rate.



### 8.3 Fit the detector

### 8.3. Train Isolation Forest

```python
isolation_forest.fit(X)

print("Isolation Forest fitted")
```

**Expected output**

```text
Isolation Forest fitted
```

**Explanation**

The detector learns the structure of the input observations without using the hidden teaching labels.



### 8.4 Predict normal and anomaly labels

### 8.4. Generate anomaly predictions

```python
isolation_predictions = isolation_forest.predict(X)

print(pd.Series(isolation_predictions).value_counts())
```

**Expected output**

```text
Counts for labels 1 and -1
```

**Explanation**

In scikit-learn, 1 means inlier and -1 means outlier. The label convention differs from the teaching label array, where 1 means anomaly.



### 8.5 Convert labels into a readable flag

### 8.5. Create an anomaly flag

```python
is_anomaly = isolation_predictions == -1

print(is_anomaly.sum())
```

**Expected output**

```text
The number of observations flagged as anomalies
```

**Explanation**

The Boolean flag makes the detector output easier to use in a report.



### 8.6 Add predictions to the data

### 8.6. Create an anomaly result table

```python
result_df = df.copy()
result_df["is_anomaly"] = is_anomaly

print(result_df.tail(10))
```

**Expected output**

```text
The final ten rows with anomaly flags
```

**Explanation**

Review flagged observations in their original feature units.



## 9. Understand Isolation Forest scores

Isolation Forest provides scores that can be used to rank observations by unusualness.

### 9.1 Get decision function scores

### 9.1. Read decision_function values

```python
decision_scores = isolation_forest.decision_function(X)

print(decision_scores[:10])
```

**Expected output**

```text
Decision scores for the first ten observations
```

**Explanation**

Lower decision-function values are more abnormal. The zero boundary is used by the estimator's prediction rule.



### 9.2 Get anomaly scores

### 9.2. Use score_samples()

```python
sample_scores = isolation_forest.score_samples(X)

print(sample_scores[:10])
```

**Expected output**

```text
Sample scores for the first ten observations
```

**Explanation**

Lower score_samples values indicate more abnormal observations in scikit-learn's Isolation Forest implementation.



### 9.3 Add scores to the result table

### 9.3. Create a ranked anomaly report

```python
result_df["anomaly_score"] = sample_scores

print(
    result_df.sort_values(
        "anomaly_score"
    ).head(10)
)
```

**Expected output**

```text
The ten observations with lowest anomaly scores
```

**Explanation**

Sorting helps prioritize unusual observations for review. A score is not a fraud probability.



### 9.4 Visualize flagged observations

### 9.4. Plot normal and flagged observations

```python
sns.scatterplot(
    data=result_df,
    x="transaction_amount",
    y="daily_transaction_count",
    hue="is_anomaly"
)
plt.title("Isolation Forest anomaly flags")
plt.show()
```

**Expected output**

```text
A scatter plot distinguishing flagged and unflagged observations
```

**Explanation**

The visualization helps inspect whether the detector flags the deliberately unusual points.



## 10. Understand contamination

contamination is used to estimate the proportion of outliers in the dataset and set a decision threshold.

It should not be interpreted as:
- The exact real-world fraud rate.
- A calibrated anomaly probability.
- Proof that every flagged observation is genuinely abnormal.

### 10.1 Compare contamination settings

### 10.1. Evaluate different contamination values

```python
contamination_results = []

for contamination in [0.01, 0.02, 0.05, 0.10]:
    candidate = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    labels = candidate.fit_predict(X)

    contamination_results.append({
        "contamination": contamination,
        "flagged_count": int(np.sum(labels == -1))
    })

print(pd.DataFrame(contamination_results))
```

**Expected output**

```text
A table showing how many observations are flagged for each setting
```

**Explanation**

Changing contamination changes the decision threshold and number of flagged observations. It should be chosen with domain context and review capacity.



## 11. Local Outlier Factor (LOF)

**Local Outlier Factor (LOF)** compares the local density around an observation with the density around its neighbours.

An observation may be unusual if it is much less dense than nearby observations.

LOF is useful when local density matters, but it can be sensitive to neighbourhood settings and feature scale.

### 11.1 Import LocalOutlierFactor

### 11.1. Import LOF

```python
from sklearn.neighbors import LocalOutlierFactor

print("LOF imported")
```

**Expected output**

```text
LOF imported
```

**Explanation**

The estimator is available from scikit-learn's neighbours module.



### 11.2 Create an LOF detector

### 11.2. Initialize LOF for outlier detection

```python
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02
)

print(lof)
```

**Expected output**

```text
A LocalOutlierFactor configuration
```

**Explanation**

n_neighbors controls the local neighbourhood size. contamination controls the expected outlier proportion used by the prediction threshold.



### 11.3 Fit and predict outliers

### 11.3. Run LOF on the dataset

```python
lof_predictions = lof.fit_predict(X_scaled)

print(pd.Series(lof_predictions).value_counts())
```

**Expected output**

```text
Counts for inlier label 1 and outlier label -1
```

**Explanation**

In default outlier-detection mode, fit_predict analyses the supplied dataset and returns labels for those observations.



### 11.4 Inspect LOF scores

### 11.4. Read negative outlier factors

```python
lof_scores = lof.negative_outlier_factor_

print(lof_scores[:10])
```

**Expected output**

```text
Negative outlier factor values
```

**Explanation**

More negative values generally indicate stronger local outlier behaviour. These are not probabilities.



### 11.5 Create an LOF result table

### 11.5. Compare Isolation Forest and LOF flags

```python
comparison_df = df.copy()
comparison_df["isolation_flag"] = isolation_predictions == -1
comparison_df["lof_flag"] = lof_predictions == -1

print(comparison_df.tail(10))
```

**Expected output**

```text
A table showing the two detector outputs
```

**Explanation**

Different methods can flag different observations because they model unusualness differently.



## 12. Local Outlier Factor for new observations

LOF's default outlier-detection mode evaluates the dataset used during fitting. For predicting whether new observations are unusual relative to a reference dataset, use novelty=True.

### 12.1 Create a novelty detector

### 12.1. Initialize LOF with novelty detection

```python
lof_novelty = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02,
    novelty=True
)

print(lof_novelty)
```

**Expected output**

```text
A LocalOutlierFactor configuration with novelty=True
```

**Explanation**

This setting supports scoring new observations after fitting. Its use differs from the default outlier-detection mode.



### 12.2 Fit novelty LOF on reference observations

### 12.2. Fit the reference population

```python
lof_novelty.fit(X_scaled)

print("Novelty detector fitted")
```

**Expected output**

```text
Novelty detector fitted
```

**Explanation**

The detector learns local density structure from the reference data.



### 12.3 Create new observations

### 12.3. Prepare new transaction examples

```python
new_transactions = pd.DataFrame({
    "transaction_amount": [1400, 1800, 8500],
    "daily_transaction_count": [4, 5, 22]
})

new_transactions_scaled = scaler.transform(new_transactions)

print(new_transactions)
```

**Expected output**

```text
Three new transaction examples
```

**Explanation**

The new records use the same feature names and are transformed using the fitted scaler.



### 12.4 Predict new observations

### 12.4. Score new transactions with LOF

```python
new_lof_predictions = lof_novelty.predict(
    new_transactions_scaled
)

print(new_lof_predictions)
```

**Expected output**

```text
Labels 1 for inlier or -1 for outlier
```

**Explanation**

The detector compares the new observations with the fitted reference population.



## 13. Compare anomaly detection methods

| Method | Main idea | Strength | Limitation |
|---|---|---|---|
| Isolation Forest | Isolates unusual points through randomized splits | Works well as a general tabular starting point | Threshold and feature representation matter |
| Local Outlier Factor | Compares local density | Can detect local anomalies | Sensitive to neighbourhood settings and density variation |
| One-Class SVM | Learns a boundary around a normal region | Can model some complex boundaries | Sensitive to scaling and parameter settings |
| Statistical rules | Flags values outside defined ranges | Simple and explainable | May miss complex or contextual anomalies |

## 14. Evaluate anomaly detection

If reliable anomaly labels exist:
- Compare flagged observations with known labels.
- Inspect precision and recall.
- Review false positives and false negatives.
- Evaluate performance by customer, region, time or product segment.

If labels do not exist:
- Review samples with domain experts.
- Inspect score distributions.
- Check whether alerts are actionable.
- Monitor alert volume.
- Investigate whether the detector flags known operational changes.

Do not treat the detector's own labels as ground truth.

### 14.1 Evaluate using synthetic teaching labels

### 14.1. Compare Isolation Forest with known synthetic labels

```python
from sklearn.metrics import classification_report

isolation_as_binary = (isolation_predictions == -1).astype(int)

print(classification_report(
    known_labels,
    isolation_as_binary,
    zero_division=0
))
```

**Expected output**

```text
Precision, recall and F1 for the synthetic anomaly labels
```

**Explanation**

The labels are available only because this is a simulated dataset. Real unsupervised anomaly detection often has incomplete or delayed labels.



### 14.2 Inspect false positives and false negatives

### 14.2. Create an evaluation table

```python
evaluation_df = df.copy()
evaluation_df["known_anomaly"] = known_labels
evaluation_df["predicted_anomaly"] = isolation_as_binary

print(pd.crosstab(
    evaluation_df["known_anomaly"],
    evaluation_df["predicted_anomaly"],
    rownames=["Actual"],
    colnames=["Predicted"]
))
```

**Expected output**

```text
A table comparing known synthetic anomalies with detector flags
```

**Explanation**

This shows which normal records were flagged and which synthetic anomalies were missed.



## 15. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Treat every anomaly as fraud | Investigate business context |
| Assume contamination is true anomaly rate | Treat it as a threshold-setting estimate |
| Interpret anomaly score as probability | Explain it as a ranking or model score |
| Ignore feature scaling for LOF | Scale features where distance matters |
| Use LOF default mode for unseen records | Use novelty=True for new observations |
| Evaluate only flagged count | Inspect false positives, false negatives and alert usefulness |
| Ignore contextual patterns | Include time, customer or segment context |
| Automatically delete flagged observations | Review before changing or removing data |

## 16. Interview-ready explanations

**What is anomaly detection?**  
It identifies observations that differ from expected or common patterns and may require investigation.

**What is Isolation Forest?**  
It isolates observations through randomized splits. Unusual observations are often isolated more easily than common observations.

**What is contamination?**  
It is an estimate of the expected outlier proportion used to set the detector's threshold. It is not a guarantee of the true anomaly rate.

**What is Local Outlier Factor?**  
It compares the local density around an observation with the density of its neighbours to identify unusual local patterns.

**Does an anomaly mean fraud or error?**  
No. It means unusual under the model or rule. It requires investigation and context.

## 17. Scenario questions

1. A detector flags a large transaction from a premium customer. What do you do?  
   **Answer:** Review the customer's history, transaction context, source data and business rules before deciding whether it is suspicious.

2. Isolation Forest flags 20% of transactions but the review team can investigate only 2%. What do you consider?  
   **Answer:** Revisit threshold settings, feature quality, ranking strategy and review capacity. Prioritize by risk and business impact.

3. A new fraud pattern is not detected because it looks similar to normal transactions. What is the limitation?  
   **Answer:** Unsupervised anomaly detection cannot identify every harmful event. Feature quality, labels and business rules remain important.

4. LOF flags observations that are normal globally but unusual within a customer segment. Is that possible?  
   **Answer:** Yes. LOF focuses on local density, so local context can affect which observations are considered unusual.

5. The detector's alert count suddenly doubles after a system update. What do you investigate?  
   **Answer:** Input schema, units, missing values, feature distribution, source-system changes and threshold configuration before assuming behaviour changed.

## 18. Lesson completion checklist

- [ ] I can explain anomaly detection.
- [ ] I can distinguish anomaly, outlier and data error.
- [ ] I understand point, contextual and collective anomalies.
- [ ] I can train Isolation Forest.
- [ ] I understand contamination and score direction.
- [ ] I can train LOF for outlier detection.
- [ ] I understand LOF novelty detection.
- [ ] I can compare detector outputs.
- [ ] I understand false alerts and missed anomalies.
- [ ] I can explain why flagged records require investigation.

**Next lesson:** Day 41 — Unsupervised Model Interpretation.
