# Day 32: Supervised Model Comparison and Model Selection

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain why multiple supervised algorithms should be compared.
- Establish a baseline before model selection.
- Choose classification or regression models based on target type.
- Compare models using the same data splits.
- Use cross-validation for a more stable comparison.
- Select metrics based on business costs.
- Consider training time, prediction time and interpretability.
- Recognize overfitting and underfitting.
- Separate model selection from final test evaluation.
- Document the final model-selection decision.

## 2. What is model selection?

**Model selection** is the process of choosing a suitable model for a particular prediction task.

It is not simply choosing the algorithm with the highest accuracy.

Consider:
- Does it predict the correct target?
- Does it perform well on unseen data?
- Does it handle the important classes or segments?
- Is it fast enough for production?
- Can the team explain and maintain it?
- Does it fit memory and infrastructure limits?
- Are the predictions stable and reliable?
- Does it improve on a simple baseline?

A model with slightly lower validation performance may be preferable if it is easier to explain, faster and more stable.

## 3. Define the task before comparing algorithms

| Target | Task | Example algorithms |
|---|---|---|
| Numerical | Regression | Linear Regression, Ridge, Random Forest Regressor |
| Binary category | Classification | Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, SVM |
| Multiple categories | Classification | Logistic Regression variants, Decision Tree, Random Forest, SVM, Naive Bayes |

Do not compare a regression estimator with a classification estimator for the same target without first defining the task correctly.

## 4. Create a classification dataset

### 4.1 Import libraries

### 4.1. Import Pandas, NumPy and plotting tools

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

These libraries prepare and inspect the example data.



### 4.2 Generate classification observations

### 4.2. Create a binary classification dataset

```python
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=600,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    weights=[0.7, 0.3],
    class_sep=1.0,
    random_state=42
)

print(X_values.shape)
print(pd.Series(y).value_counts())
```

**Expected output**

```text
(600, 8) and target class counts
```

**Explanation**

The dataset contains eight numerical features and a binary target. It is synthetic and designed for model-comparison practice.



### 4.3 Create a DataFrame

### 4.3. Add feature names

```python
feature_names = [f"feature_{i}" for i in range(1, 9)]

df = pd.DataFrame(X_values, columns=feature_names)
df["target"] = y

print(df.head())
```

**Expected output**

```text
A five-row preview with eight features and target
```

**Explanation**

Named columns make the feature schema easier to inspect.



### 4.4 Inspect target balance

### 4.4. Check class proportions

```python
print(df["target"].value_counts(normalize=True))
```

**Expected output**

```text
The proportions for class 0 and class 1
```

**Explanation**

The positive class is smaller in this example. Accuracy alone may not be enough to compare classifiers.



## 5. Split the data once for final evaluation

### 5.1 Separate X and y

### 5.1. Create features and target

```python
X = df.drop(columns=["target"])
y = df["target"]

print(X.shape, y.shape)
```

**Expected output**

```text
(600, 8) and (600,)
```

**Explanation**

The target is excluded from the feature matrix.



### 5.2 Create training and test sets

### 5.2. Hold out the final test set

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)
```

**Expected output**

```text
(480, 8) and (120, 8)
```

**Explanation**

The test set is reserved for final evaluation. Model comparison and tuning will use training data only.



## 6. Establish a baseline

A baseline provides a simple reference. A model should add value beyond this starting point.

### 6.1 Import DummyClassifier

### 6.1. Import the baseline estimator

```python
from sklearn.dummy import DummyClassifier

print("DummyClassifier ready")
```

**Expected output**

```text
DummyClassifier ready
```

**Explanation**

The baseline does not learn complex relationships between features and target.



### 6.2 Fit the majority-class baseline

### 6.2. Train the baseline

```python
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

print("Baseline fitted")
```

**Expected output**

```text
Baseline fitted
```

**Explanation**

The baseline learns the most frequent class in the training labels.



### 6.3 Predict test labels

### 6.3. Generate baseline predictions

```python
baseline_predictions = baseline.predict(X_test)

print(baseline_predictions[:15])
```

**Expected output**

```text
The first fifteen baseline predictions
```

**Explanation**

The baseline predicts the most frequent class for each test observation.



### 6.4 Inspect baseline metrics

### 6.4. Calculate baseline classification metrics

```python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, baseline_predictions))
print(classification_report(y_test, baseline_predictions, zero_division=0))
```

**Expected output**

```text
Baseline accuracy and a class-level report
```

**Explanation**

This shows what a trivial strategy can achieve. A candidate model should be evaluated against this reference.



## 7. Prepare candidate models

We will compare several algorithms:
- Logistic Regression.
- K-Nearest Neighbors.
- Decision Tree.
- Random Forest.
- Support Vector Machine.
- Gaussian Naive Bayes.

Each model has different assumptions, strengths and operational trade-offs.

### 7.1 Import candidate estimators

### 7.1. Import supervised models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

print("Candidate models imported")
```

**Expected output**

```text
Candidate models imported
```

**Explanation**

These estimators represent different approaches to classification.



### 7.2 Build model pipelines

### 7.2. Create candidate model configurations

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "KNN": Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42
    ),
    "SVM": Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ]),
    "GaussianNB": GaussianNB()
}

print(list(models.keys()))
```

**Expected output**

```text
A list of six candidate model names
```

**Explanation**

Scaling is included for algorithms that are sensitive to feature magnitude. Tree models and GaussianNB are shown without scaling for this numeric example.



## 8. Compare models using cross-validation

Cross-validation evaluates each model across multiple training/validation folds. It helps reduce dependence on one random split.

### 8.1 Import cross-validation tools

### 8.1. Import StratifiedKFold and cross_validate

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

print("Cross-validation tools ready")
```

**Expected output**

```text
Cross-validation tools ready
```

**Explanation**

StratifiedKFold keeps class proportions approximately consistent across folds.



### 8.2 Configure the folds

### 8.2. Create a stratified cross-validation strategy

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print(cv)
```

**Expected output**

```text
A five-fold StratifiedKFold configuration
```

**Explanation**

The training set is divided into five folds. Each fold is used for validation while the remaining folds are used for fitting.



### 8.3 Define multiple metrics

### 8.3. Select comparison metrics

```python
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

print(scoring)
```

**Expected output**

```text
A dictionary of four classification metrics
```

**Explanation**

The metrics provide a broader view than accuracy alone. Choose metrics that reflect the business objective.



### 8.4 Run cross-validation for each model

### 8.4. Evaluate all candidate models

```python
cv_rows = []

for name, estimator in models.items():
    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    cv_rows.append({
        "model": name,
        "train_accuracy": scores["train_accuracy"].mean(),
        "validation_accuracy": scores["test_accuracy"].mean(),
        "validation_precision": scores["test_precision"].mean(),
        "validation_recall": scores["test_recall"].mean(),
        "validation_f1": scores["test_f1"].mean(),
        "validation_f1_std": scores["test_f1"].std()
    })

cv_results = pd.DataFrame(cv_rows)

print(cv_results.sort_values("validation_f1", ascending=False))
```

**Expected output**

```text
A comparison table of mean training and validation metrics
```

**Explanation**

Each model is evaluated using the same folds. Pipeline preprocessing is fitted separately inside each fold, reducing preprocessing leakage.



### 8.5 Compare validation F1

### 8.5. Sort models by validation F1

```python
print(
    cv_results[
        ["model", "validation_f1", "validation_f1_std"]
    ].sort_values("validation_f1", ascending=False)
)
```

**Expected output**

```text
Models ranked by mean validation F1 and fold variation
```

**Explanation**

A model with a slightly lower mean but much lower variation may be worth investigating. Do not treat small score differences as certain improvements.



### 8.6 Compare training and validation performance

### 8.6. Inspect possible overfitting

```python
print(
    cv_results[
        ["model", "train_accuracy", "validation_accuracy"]
    ].sort_values("validation_accuracy", ascending=False)
)
```

**Expected output**

```text
Training and validation accuracy for each candidate
```

**Explanation**

A large training-validation gap can indicate overfitting. Consider the data size, model complexity and metric choice.



## 9. Select a candidate model

Model selection should consider:
- Validation performance.
- Variation across folds.
- Error costs.
- Class and segment performance.
- Interpretability.
- Training and inference latency.
- Memory and deployment requirements.
- Monitoring and maintenance.

### 9.1 Identify the top validation F1 candidate

### 9.1. Read the highest mean validation F1 row

```python
top_candidate = cv_results.loc[
    cv_results["validation_f1"].idxmax()
]

print(top_candidate)
```

**Expected output**

```text
The row with the highest mean validation F1
```

**Explanation**

This is an initial selection rule for the example. A real selection decision should also consider the other criteria and uncertainty.



### 9.2 Record the selected model name

### 9.2. Store the selected candidate

```python
selected_model_name = top_candidate["model"]

print(selected_model_name)
```

**Expected output**

```text
The selected model name
```

**Explanation**

Record the model name, settings, metric and validation procedure.



### 9.3 Fit the selected model on the full training set

### 9.3. Fit the selected estimator

```python
selected_model = models[selected_model_name]

selected_model.fit(X_train, y_train)

print("Selected model fitted")
```

**Expected output**

```text
Selected model fitted
```

**Explanation**

After model selection, refit the chosen estimator on the complete training partition. The final test set remains untouched.



## 10. Evaluate the selected model once on test data

### 10.1 Predict final test labels

### 10.1. Generate final predictions

```python
final_predictions = selected_model.predict(X_test)

print(final_predictions[:15])
```

**Expected output**

```text
The first fifteen final test predictions
```

**Explanation**

The model uses the training-fitted estimator and preprocessing to predict held-out observations.



### 10.2 Calculate final metrics

### 10.2. Calculate final test metrics

```python
from sklearn.metrics import precision_score, recall_score, f1_score

final_metrics = {
    "accuracy": accuracy_score(y_test, final_predictions),
    "precision": precision_score(y_test, final_predictions, zero_division=0),
    "recall": recall_score(y_test, final_predictions, zero_division=0),
    "f1": f1_score(y_test, final_predictions, zero_division=0)
}

print(final_metrics)
```

**Expected output**

```text
A dictionary of final test metrics
```

**Explanation**

This is the final held-out evaluation for the selected model. Do not keep changing model settings based on this result.



### 10.3 Compare final model with baseline

### 10.3. Create a final baseline comparison

```python
final_comparison = pd.DataFrame({
    "model": ["Majority baseline", selected_model_name],
    "accuracy": [
        accuracy_score(y_test, baseline_predictions),
        final_metrics["accuracy"]
    ],
    "f1": [
        f1_score(y_test, baseline_predictions, zero_division=0),
        final_metrics["f1"]
    ]
})

print(final_comparison)
```

**Expected output**

```text
A comparison table for baseline and selected candidate
```

**Explanation**

The table shows whether the selected model adds value beyond the simple reference on the held-out test data.



## 11. Understand underfitting and overfitting

### Underfitting

The model is too simple to capture useful patterns. It may perform poorly on both training and validation data.

### Overfitting

The model fits training observations closely but does not generalize well to validation or test observations.

### Good generalization

The model performs consistently on training and unseen observations, with acceptable errors for the business task.

### 11.1 Inspect training and validation scores

### 11.1. Compare train and validation score averages

```python
cv_results["accuracy_gap"] = (
    cv_results["train_accuracy"] -
    cv_results["validation_accuracy"]
)

print(
    cv_results[["model", "accuracy_gap"]]
    .sort_values("accuracy_gap", ascending=False)
)
```

**Expected output**

```text
A table showing the train-validation accuracy gap
```

**Explanation**

A large positive gap can indicate overfitting, though it must be interpreted with sample size and model behaviour.



## 12. Model selection beyond metrics

A production model must meet practical constraints.

| Criterion | Question |
|---|---|
| Predictive performance | Does it meet the agreed metric target? |
| Generalization | Are fold scores stable? |
| Interpretability | Can users understand or audit decisions? |
| Latency | Does inference meet response-time requirements? |
| Memory | Can the model fit the deployment environment? |
| Data requirements | Are all features available at prediction time? |
| Reliability | How does it handle missing or unknown values? |
| Maintenance | Can the team retrain, monitor and version it? |
| Risk | Are errors acceptable for the use case? |

A model should be selected based on the complete requirement, not only one score.

## 13. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Compare models on different splits | Comparison is unfair | Use the same folds and evaluation data |
| Select by training accuracy | Rewards memorization | Use validation or cross-validation |
| Use accuracy alone | Hides minority-class errors | Compare precision, recall and F1 |
| Ignore fold variation | Averages can hide instability | Inspect standard deviation and per-fold scores |
| Tune repeatedly on test data | Test set becomes part of development | Use validation or cross-validation |
| Ignore baseline | No reference for added value | Establish a simple baseline |
| Select only by score | Ignores deployment constraints | Consider latency, memory, interpretability and risk |
| Assume a small score difference is meaningful | Results vary by sample | Check stability and practical significance |

## 14. Interview-ready explanations

**What is model selection?**  
It is choosing a suitable model by comparing predictive performance, stability, error costs, interpretability and operational constraints.

**Why establish a baseline?**  
It provides a simple reference so the team can determine whether a complex model adds meaningful value.

**Why use cross-validation?**  
It evaluates models across multiple partitions and reduces reliance on one split. It also supports more stable model comparison.

**Why use the same folds for all candidates?**  
It makes the comparison fair because each model is evaluated on the same training and validation partitions.

**Why should the test set be used only at the end?**  
Repeatedly using test results to select models or settings makes the test set part of development and weakens the final evaluation.

## 15. Scenario questions

1. Model A has slightly higher mean F1 but much larger fold variation than Model B. What do you consider?  
   **Answer:** Compare stability, sample size, business impact and practical difference. A more stable model may be preferable.

2. A complex model improves F1 by a very small amount but doubles inference latency. What do you do?  
   **Answer:** Decide whether the performance gain justifies the operational cost and business requirements.

3. One model has high accuracy but low recall for fraud. Should it be selected?  
   **Answer:** Not without reviewing the cost of missed fraud and appropriate class-level metrics.

4. The team chooses the best model using test accuracy after trying many candidates. What is wrong?  
   **Answer:** The test set has influenced model selection. Use validation or cross-validation and reserve the test set for final evaluation.

5. Two models perform similarly. How do you choose?  
   **Answer:** Consider stability, interpretability, latency, memory, reliability and maintenance requirements.

## 16. Lesson completion checklist

- [ ] I can explain model selection.
- [ ] I can create a baseline.
- [ ] I can compare classification algorithms.
- [ ] I can use StratifiedKFold and cross_validate.
- [ ] I can compare multiple metrics.
- [ ] I can inspect fold variation.
- [ ] I understand underfitting and overfitting.
- [ ] I can evaluate the selected model once on test data.
- [ ] I can include deployment constraints in the selection decision.

**Next lesson:** Day 33 — Clustering and Unsupervised Learning.
