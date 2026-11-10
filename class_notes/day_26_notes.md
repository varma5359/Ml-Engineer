# Day 26: Decision Boundaries and Classification Thresholds

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain a decision boundary in plain language.
- Understand how a classifier assigns classes in feature space.
- Explain what a classification threshold does.
- Use predicted probabilities to create class labels.
- Compare high and low thresholds.
- Understand the precision-recall trade-off.
- Select a threshold using validation data.
- Visualize a two-feature decision boundary.
- Consider business costs and operational review capacity.
- Avoid selecting a final threshold using the test set.

## 2. What is a decision boundary?

A **decision boundary** is the separation between regions of feature space that a classifier assigns to different classes.

Imagine a model that classifies transactions as:
- 0: normal.
- 1: potentially fraudulent.

Each transaction has feature values, such as amount and number of failed attempts. The classifier separates the feature space into areas where it predicts class 0 or class 1.

The boundary depends on:
- The model type.
- The learned model parameters.
- The input features.
- The classification threshold.

A straight-line boundary may be produced by a linear classifier. Other algorithms can create more complex boundaries.

## 3. What is a classification threshold?

A classifier may estimate a probability for the positive class. The **threshold** is the cut-off used to convert that probability into a final class label.

For example, with a threshold of 0.50:
- Probability below 0.50 → predict class 0.
- Probability at or above 0.50 → predict class 1.

The default threshold is not always the best business choice. A different threshold may be appropriate when false negatives and false positives have different costs.

## 4. Create a classification dataset

### 4.1 Import libraries

### 4.1. Import NumPy, Pandas and plotting libraries

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

NumPy, Pandas and plotting libraries support the demonstration.



### 4.2 Generate two-feature classification data

### 4.2. Create a synthetic two-class dataset

```python
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.2,
    random_state=42
)

print(X_values.shape)
print(pd.Series(y).value_counts())
```

**Expected output**

```text
(300, 2) and class counts for 0 and 1
```

**Explanation**

The dataset contains two numerical features so the decision boundary can be visualized on a two-dimensional chart.



### 4.3 Create a DataFrame

### 4.3. Add feature names

```python
df = pd.DataFrame(
    X_values,
    columns=["feature_1", "feature_2"]
)

df["target"] = y

print(df.head())
```

**Expected output**

```text
A table with two feature columns and one target column
```

**Explanation**

Feature names are added to make the dataset easier to inspect.



### 4.4 Plot the two classes

### 4.4. Visualize class distribution in feature space

```python
sns.scatterplot(
    data=df,
    x="feature_1",
    y="feature_2",
    hue="target"
)
plt.title("Two-class feature space")
plt.show()
```

**Expected output**

```text
A scatter plot with points colored by target class
```

**Explanation**

The chart shows how the two classes are distributed across the feature space. Some overlap may remain.



## 5. Split the data

### 5.1 Create features and target

### 5.1. Separate X and y

```python
X = df[["feature_1", "feature_2"]]
y = df["target"]

print(X.shape, y.shape)
```

**Expected output**

```text
(300, 2) and (300,)
```

**Explanation**

The target is kept separate from the two input features.



### 5.2 Create training, validation and test partitions

### 5.2. Hold out test observations

```python
from sklearn.model_selection import train_test_split

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_dev.shape, X_test.shape)
```

**Expected output**

```text
(240, 2) and (60, 2)
```

**Explanation**

The final test set is held back. The development data is used for training and threshold selection.



### 5.3 Create training and validation subsets

### 5.3. Split development data

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

print(X_train.shape, X_valid.shape, X_test.shape)
```

**Expected output**

```text
(180, 2), (60, 2), (60, 2)
```

**Explanation**

The overall proportions are 60% training, 20% validation and 20% test.



## 6. Train a logistic regression classifier

### 6.1 Create a preprocessing and model pipeline

### 6.1. Build a scaled logistic regression pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

print(model)
```

**Expected output**

```text
A pipeline containing StandardScaler and LogisticRegression
```

**Explanation**

The scaler is fitted on training data only and reused for validation and test data.



### 6.2 Fit the model

### 6.2. Train using training observations

```python
model.fit(X_train, y_train)

print("Classifier fitted")
```

**Expected output**

```text
Classifier fitted
```

**Explanation**

The model learns its decision function from the training subset.



### 6.3 Generate validation probabilities

### 6.3. Get validation probabilities

```python
valid_probabilities = model.predict_proba(X_valid)[:, 1]

print(valid_probabilities[:10])
```

**Expected output**

```text
The first ten estimated positive-class probabilities
```

**Explanation**

The second probability column corresponds to class 1 for this binary example. Check classes_ if class ordering is uncertain.



### 6.4 Inspect class order

### 6.4. Read classifier classes

```python
print(model.named_steps["classifier"].classes_)
```

**Expected output**

```text
[0 1]
```

**Explanation**

This confirms that probability column zero is class 0 and column one is class 1.



## 7. Apply a threshold manually

### 7.1 Use the default 0.50 threshold

### 7.1. Convert probabilities to labels

```python
threshold = 0.50

valid_pred_50 = (valid_probabilities >= threshold).astype(int)

print(valid_pred_50[:10])
```

**Expected output**

```text
The first ten predicted class labels
```

**Explanation**

The comparison creates True/False values and astype(int) converts them to zero and one.



### 7.2 Create a reusable threshold function

### 7.2. Define a threshold prediction function

```python
def apply_threshold(probabilities, threshold):
    return (probabilities >= threshold).astype(int)

print(apply_threshold(np.array([0.2, 0.5, 0.8]), 0.5))
```

**Expected output**

```text
[0 1 1]
```

**Explanation**

The function converts positive-class probabilities into class labels using the supplied threshold.



### 7.3 Apply a lower threshold

### 7.3. Use threshold 0.30

```python
valid_pred_30 = apply_threshold(valid_probabilities, 0.30)

print(valid_pred_30[:10])
```

**Expected output**

```text
The first ten predicted labels at threshold 0.30
```

**Explanation**

A lower threshold usually classifies more observations as positive.



### 7.4 Apply a higher threshold

### 7.4. Use threshold 0.70

```python
valid_pred_70 = apply_threshold(valid_probabilities, 0.70)

print(valid_pred_70[:10])
```

**Expected output**

```text
The first ten predicted labels at threshold 0.70
```

**Explanation**

A higher threshold usually requires stronger model confidence before assigning the positive class.



## 8. Understand the threshold trade-off

### Lower threshold

- More observations are predicted positive.
- The model may catch more actual positive cases.
- False positives may increase.
- Review teams may receive more alerts.

### Higher threshold

- Fewer observations are predicted positive.
- False positives may decrease.
- Some true positive cases may be missed.
- The number of alerts may fall.

The direction of precision and recall changes depends on the data, but lowering a threshold commonly increases recall while reducing precision.

## 9. Compare precision and recall across thresholds

### 9.1 Import metrics

### 9.1. Import precision and recall

```python
from sklearn.metrics import precision_score, recall_score, f1_score

print("Metrics imported")
```

**Expected output**

```text
Metrics imported
```

**Explanation**

Precision and recall describe different kinds of classification performance.



### 9.2 Calculate metrics at threshold 0.30

### 9.2. Evaluate lower threshold

```python
print("Precision:", precision_score(y_valid, valid_pred_30, zero_division=0))
print("Recall:", recall_score(y_valid, valid_pred_30, zero_division=0))
print("F1:", f1_score(y_valid, valid_pred_30, zero_division=0))
```

**Expected output**

```text
Three metric values
```

**Explanation**

This describes the validation performance at threshold 0.30.



### 9.3 Calculate metrics at threshold 0.50

### 9.3. Evaluate default threshold

```python
print("Precision:", precision_score(y_valid, valid_pred_50, zero_division=0))
print("Recall:", recall_score(y_valid, valid_pred_50, zero_division=0))
print("F1:", f1_score(y_valid, valid_pred_50, zero_division=0))
```

**Expected output**

```text
Three metric values
```

**Explanation**

This provides a reference for comparison with other thresholds.



### 9.4 Calculate metrics at threshold 0.70

### 9.4. Evaluate higher threshold

```python
print("Precision:", precision_score(y_valid, valid_pred_70, zero_division=0))
print("Recall:", recall_score(y_valid, valid_pred_70, zero_division=0))
print("F1:", f1_score(y_valid, valid_pred_70, zero_division=0))
```

**Expected output**

```text
Three metric values
```

**Explanation**

The threshold changes the final class labels and therefore changes precision and recall.



### 9.5 Build a threshold comparison table

### 9.5. Compare three thresholds

```python
threshold_rows = []

for threshold in [0.30, 0.50, 0.70]:
    predictions = apply_threshold(valid_probabilities, threshold)

    threshold_rows.append({
        "threshold": threshold,
        "precision": precision_score(y_valid, predictions, zero_division=0),
        "recall": recall_score(y_valid, predictions, zero_division=0),
        "f1": f1_score(y_valid, predictions, zero_division=0),
        "positive_predictions": int(predictions.sum())
    })

threshold_table = pd.DataFrame(threshold_rows)

print(threshold_table)
```

**Expected output**

```text
A table comparing precision, recall, F1 and positive prediction count
```

**Explanation**

The positive prediction count helps connect threshold choice with operational alert volume.



## 10. Decision boundary visualization

A decision boundary shows where the model changes its predicted class across feature space.

### 10.1 Find feature ranges

### 10.1. Inspect minimum and maximum feature values

```python
print(X_train.min())
print(X_train.max())
```

**Expected output**

```text
Minimum and maximum values for each feature
```

**Explanation**

The plotting grid should cover the observed training feature range.



### 10.2 Create a mesh grid

### 10.2. Generate a grid of feature combinations

```python
feature_1_values = np.linspace(
    X_train["feature_1"].min() - 0.5,
    X_train["feature_1"].max() + 0.5,
    150
)

feature_2_values = np.linspace(
    X_train["feature_2"].min() - 0.5,
    X_train["feature_2"].max() + 0.5,
    150
)

xx, yy = np.meshgrid(feature_1_values, feature_2_values)

print(xx.shape, yy.shape)
```

**Expected output**

```text
Two grid arrays with matching two-dimensional shapes
```

**Explanation**

The mesh grid creates many points across the feature space so the classifier can be evaluated at each location.



### 10.3 Convert the grid into a DataFrame

### 10.3. Prepare grid points for the model

```python
grid_points = pd.DataFrame({
    "feature_1": xx.ravel(),
    "feature_2": yy.ravel()
})

print(grid_points.head())
```

**Expected output**

```text
A DataFrame containing feature combinations
```

**Explanation**

The flattened grid values are arranged into rows matching the model's expected feature names.



### 10.4 Predict probabilities across the grid

### 10.4. Get grid probabilities

```python
grid_probabilities = model.predict_proba(grid_points)[:, 1]

print(grid_probabilities[:5])
```

**Expected output**

```text
The first five positive-class probabilities across the grid
```

**Explanation**

The model estimates a probability for every location in feature space.



### 10.5 Reshape probabilities to the grid

### 10.5. Restore the grid shape

```python
probability_surface = grid_probabilities.reshape(xx.shape)

print(probability_surface.shape)
```

**Expected output**

```text
A two-dimensional probability surface
```

**Explanation**

The reshaped values correspond to the grid positions for contour plotting.



### 10.6 Plot the decision boundary at 0.50

### 10.6. Draw the probability contour

```python
plt.contourf(
    xx,
    yy,
    probability_surface,
    levels=[0, 0.5, 1],
    alpha=0.35
)

sns.scatterplot(
    data=df,
    x="feature_1",
    y="feature_2",
    hue="target",
    edgecolor="black"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic regression decision boundary")
plt.show()
```

**Expected output**

```text
A feature-space chart with class observations and a shaded decision region
```

**Explanation**

The contour separates locations where the model probability is below or above 0.50. The boundary is a model decision rule, not a natural law.



### 10.7 Plot a different threshold boundary

### 10.7. Draw a contour for threshold 0.70

```python
plt.contourf(
    xx,
    yy,
    probability_surface,
    levels=[0, 0.7, 1],
    alpha=0.35
)

sns.scatterplot(
    data=df,
    x="feature_1",
    y="feature_2",
    hue="target",
    edgecolor="black"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision region at threshold 0.70")
plt.show()
```

**Expected output**

```text
A probability-region chart with the 0.70 contour
```

**Explanation**

Changing the threshold changes which feature-space locations are assigned to the positive class.



## 11. Select a threshold using validation data

A threshold should reflect the business objective:
- Cost of missing a positive case.
- Cost of a false alert.
- Number of cases the team can review.
- Regulatory or safety requirements.
- Desired precision or recall.

Do not select a threshold only because it produces the highest accuracy.

### 11.1 Evaluate a range of thresholds

### 11.1. Create a threshold search table

```python
candidate_thresholds = np.arange(0.10, 0.91, 0.05)

threshold_results = []

for threshold in candidate_thresholds:
    predictions = apply_threshold(valid_probabilities, threshold)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(y_valid, predictions, zero_division=0),
        "recall": recall_score(y_valid, predictions, zero_division=0),
        "f1": f1_score(y_valid, predictions, zero_division=0),
        "alerts": int(predictions.sum())
    })

threshold_results_df = pd.DataFrame(threshold_results)

print(threshold_results_df.head())
```

**Expected output**

```text
A table showing validation metrics across candidate thresholds
```

**Explanation**

The threshold search is performed on validation probabilities, not test probabilities.



### 11.2 Find threshold with highest validation F1

### 11.2. Select a threshold using validation F1

```python
best_row = threshold_results_df.loc[
    threshold_results_df["f1"].idxmax()
]

print(best_row)
```

**Expected output**

```text
The row with the highest validation F1 score
```

**Explanation**

This is one possible selection rule. The best threshold for a business may instead depend on recall, precision, alert volume or explicit cost constraints.



### 11.3 Record the selected threshold

### 11.3. Store the threshold

```python
selected_threshold = float(best_row["threshold"])

print(selected_threshold)
```

**Expected output**

```text
A selected threshold value
```

**Explanation**

Record the selection rule and validation data version. The threshold is part of the deployed decision policy.



## 12. Evaluate once on the test set

After selecting the threshold using validation data, evaluate it on the final test set.

### 12.1 Generate test probabilities

### 12.1. Predict test probabilities

```python
test_probabilities = model.predict_proba(X_test)[:, 1]

print(test_probabilities[:10])
```

**Expected output**

```text
The first ten test positive-class probabilities
```

**Explanation**

The test set has not been used to choose the threshold.



### 12.2 Apply the selected threshold

### 12.2. Create test class predictions

```python
test_predictions = apply_threshold(
    test_probabilities,
    selected_threshold
)

print(test_predictions)
```

**Expected output**

```text
Test class labels generated using the selected threshold
```

**Explanation**

This applies the validation-selected threshold to held-out test probabilities.



### 12.3 Evaluate test precision and recall

### 12.3. Report final test metrics

```python
print("Precision:", precision_score(y_test, test_predictions, zero_division=0))
print("Recall:", recall_score(y_test, test_predictions, zero_division=0))
print("F1:", f1_score(y_test, test_predictions, zero_division=0))
```

**Expected output**

```text
Three final test metric values
```

**Explanation**

This is the final held-out evaluation for the chosen model and threshold. Do not keep adjusting the threshold based on these results.



## 13. Business threshold scenarios

### Fraud detection

Missing fraud can be expensive. A lower threshold may be considered to catch more suspicious cases, but false alerts increase investigation workload.

### Medical screening

A screening tool may prioritize high recall so fewer positive cases are missed. A follow-up diagnostic process may be required to confirm cases.

### Marketing campaign

A team may prefer higher precision if contacting customers is expensive and the campaign has limited capacity.

### Customer retention

The team may have a fixed number of customer calls per day. The threshold should consider the number of customers flagged and the value of successful interventions.

## 14. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Assume 0.50 is always best | Business costs differ | Select using validation and costs |
| Lower threshold without checking workload | Alert volume may become too large | Report positive prediction count |
| Maximize accuracy only | Minority class may be missed | Inspect precision, recall and costs |
| Choose threshold using test data | Test set becomes part of model selection | Use validation data |
| Treat probability as certainty | Outcomes remain uncertain | Explain probability as estimate |
| Ignore calibration | Thresholds may not behave as expected | Assess probability calibration |
| Change threshold without documenting it | Production decisions become inconsistent | Version the threshold and selection rule |

## 15. Interview-ready explanations

**What is a decision boundary?**  
It is the separation in feature space between regions that a classifier assigns to different classes.

**What is a classification threshold?**  
It is the cut-off applied to a score or probability to convert it into a final class label.

**What happens when the threshold is lowered?**  
The classifier generally predicts more positive cases. Recall may increase, but false positives can also increase and precision may fall.

**Why should threshold selection use validation data?**  
The threshold is a model-development decision. The test set should remain independent for final evaluation.

**How do business costs affect threshold selection?**  
The cost of false negatives, false positives and the operational capacity for reviewing alerts should guide the decision.

## 16. Scenario questions

1. Fraud is expensive to miss. What should threshold selection consider?  
   **Answer:** Recall, false-negative cost, precision and investigation capacity.

2. A review team can investigate only 100 alerts per day. What should the model team consider?  
   **Answer:** Alert volume at different thresholds, ranking quality and the business value of investigated cases.

3. The default threshold gives high accuracy but low recall. What might you try?  
   **Answer:** Evaluate lower thresholds using validation data and compare precision, recall and workload.

4. Can you select the final threshold on the test set?  
   **Answer:** No. Use validation data and keep the test set for final evaluation.

5. A model's probability estimates are poorly calibrated. What is the concern?  
   **Answer:** A threshold may not correspond to the expected real-world event rate. Assess calibration and consider a suitable calibration approach.

## 17. Lesson completion checklist

- [ ] I can explain a decision boundary.
- [ ] I understand how a threshold converts probabilities into classes.
- [ ] I can apply a custom threshold.
- [ ] I can compare precision and recall at different thresholds.
- [ ] I can visualize a two-feature decision boundary.
- [ ] I can select a threshold using validation data.
- [ ] I can evaluate the selected threshold once on test data.
- [ ] I can explain business threshold trade-offs.

**Next lesson:** Day 27 — K-Nearest Neighbors and Distance Metrics.
