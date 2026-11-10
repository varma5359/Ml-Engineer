# Day 51: Imbalanced Datasets and Resampling Concepts

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives
- Explain class imbalance and why it matters.
- Identify imbalance using target counts and proportions.
- Understand why accuracy can be misleading.
- Compare class weights, random oversampling and undersampling.
- Understand SMOTE and its limitations.
- Apply resampling only to training data.
- Use an imbalanced-learn Pipeline safely.
- Evaluate with precision, recall, F1 and PR analysis.
- Explain resampling choices in interviews.

## 2. What is an imbalanced dataset?

A classification dataset is imbalanced when one class has many more observations than another.

Examples:
- Fraud detection: normal transactions are much more common than fraud.
- Defect detection: most products are good.
- Disease screening: positive cases may be rare.
- Churn prediction: most customers may stay.

The minority class may be the most important class. A model can achieve high accuracy by mostly predicting the majority class while failing to identify minority examples.

## 3. Identify class imbalance

### 3.1 Import libraries

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Libraries ready")
~~~

### 3.2 Create an imbalanced target

~~~python
y = np.array([0] * 900 + [1] * 100)

print(pd.Series(y).value_counts())
~~~

**Expected output**
~~~text
0    900
1    100
~~~

**Explanation**
Class 0 is the majority class and class 1 is the minority class.

### 3.3 Calculate class proportions

~~~python
class_proportions = pd.Series(y).value_counts(normalize=True)

print(class_proportions)
~~~

**Expected output**
Class 0: 90%; class 1: 10%.

### 3.4 Visualize class distribution

~~~python
pd.Series(y).value_counts().sort_index().plot(kind="bar")
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.show()
~~~

**Explanation**
The chart makes the class imbalance visible. Always inspect the target distribution before selecting evaluation and training strategies.

## 4. Why imbalance can cause problems

A model may optimize an objective that is dominated by the majority class.

Possible outcomes:
- High overall accuracy.
- Low recall for the minority class.
- Few predicted positive cases.
- Poor detection of rare but important events.

Imbalance does not automatically mean the model is wrong. The correct approach depends on the business objective, data volume, label quality and model behaviour.

## 5. Establish a baseline before resampling

Before applying sampling methods:
- Split the data appropriately.
- Train a baseline model.
- Inspect the confusion matrix.
- Check minority-class precision and recall.
- Record PR-AUC or Average Precision where appropriate.
- Understand the operational cost of false alerts and missed cases.

Do not assume that resampling always improves performance.

## 6. Create a classification dataset

### 6.1 Generate features and target

~~~python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=12,
    n_informative=5,
    n_redundant=2,
    weights=[0.90, 0.10],
    random_state=42
)

print("X shape:", X.shape)
print(pd.Series(y).value_counts())
~~~

### 6.2 Split before resampling

~~~python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
~~~

**Explanation**
- stratify preserves class proportions as closely as possible in both splits.
- The test set must retain a realistic distribution.
- Do not resample before this split.

## 7. Train a baseline classifier

### 7.1 Import LogisticRegression

~~~python
from sklearn.linear_model import LogisticRegression

baseline_model = LogisticRegression(max_iter=1000)

print("Baseline model ready")
~~~

### 7.2 Fit the baseline

~~~python
baseline_model.fit(X_train, y_train)

print("Baseline fitted")
~~~

### 7.3 Predict test labels

~~~python
baseline_predictions = baseline_model.predict(X_test)

print(pd.Series(baseline_predictions).value_counts())
~~~

### 7.4 Evaluate minority-class performance

~~~python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        baseline_predictions,
        zero_division=0
    )
)
~~~

**Explanation**
Inspect the class 1 row. The minority class may have weaker recall or precision than the overall accuracy suggests.

## 8. Strategy 1 — Class weights

Class weighting changes the importance assigned to classes during model training.

For many scikit-learn classifiers, class_weight="balanced" automatically adjusts weights inversely according to class frequency.

Benefits:
- Does not create synthetic observations.
- Does not discard majority observations.
- Simple to test for supported estimators.

Limitations:
- Not supported by every estimator.
- Does not guarantee better minority-class performance.
- Can increase false positives.
- The resulting threshold still needs evaluation.

### 8.1 Create a class-weighted model

~~~python
weighted_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

print(weighted_model)
~~~

### 8.2 Fit the weighted model

~~~python
weighted_model.fit(X_train, y_train)

print("Weighted model fitted")
~~~

### 8.3 Predict and evaluate

~~~python
weighted_predictions = weighted_model.predict(X_test)

print(
    classification_report(
        y_test,
        weighted_predictions,
        zero_division=0
    )
)
~~~

**Explanation**
Compare the minority-class precision and recall with the baseline. Do not judge the result by accuracy alone.

## 9. Strategy 2 — Random oversampling

Random oversampling increases the minority class by duplicating existing minority observations.

Benefits:
- Simple method.
- Keeps all majority observations.
- Can help the model see minority examples more often.

Limitations:
- Duplicated records can increase overfitting.
- Does not create new information.
- Sampling must happen only inside training data.

### 9.1 Import RandomOverSampler

~~~python
from imblearn.over_sampling import RandomOverSampler

print("RandomOverSampler imported")
~~~

### 9.2 Create the sampler

~~~python
over_sampler = RandomOverSampler(
    random_state=42
)

print(over_sampler)
~~~

### 9.3 Resample training data only

~~~python
X_train_over, y_train_over = over_sampler.fit_resample(
    X_train,
    y_train
)

print(pd.Series(y_train_over).value_counts())
~~~

**Expected output**
A more balanced class count in the resampled training target.

**Explanation**
The sampler duplicates minority observations. The original X_test and y_test remain unchanged.

### 9.4 Train a model on oversampled data

~~~python
over_model = LogisticRegression(max_iter=1000)

over_model.fit(X_train_over, y_train_over)

print("Oversampled model fitted")
~~~

### 9.5 Evaluate on the original test set

~~~python
over_predictions = over_model.predict(X_test)

print(
    classification_report(
        y_test,
        over_predictions,
        zero_division=0
    )
)
~~~

**Explanation**
Always evaluate on the untouched test distribution. Do not evaluate on the resampled training data and present it as generalization performance.

## 10. Strategy 3 — Random undersampling

Random undersampling reduces the majority class by removing some majority observations.

Benefits:
- Reduces training size and compute cost.
- Can give the minority class more influence.
- Useful when the majority dataset is extremely large.

Limitations:
- Discards potentially useful majority information.
- Results can vary depending on which records are removed.
- It can make the training data less representative.

### 10.1 Import RandomUnderSampler

~~~python
from imblearn.under_sampling import RandomUnderSampler

print("RandomUnderSampler imported")
~~~

### 10.2 Create the sampler

~~~python
under_sampler = RandomUnderSampler(
    random_state=42
)

print(under_sampler)
~~~

### 10.3 Resample training data

~~~python
X_train_under, y_train_under = under_sampler.fit_resample(
    X_train,
    y_train
)

print(pd.Series(y_train_under).value_counts())
~~~

### 10.4 Fit an undersampled model

~~~python
under_model = LogisticRegression(max_iter=1000)

under_model.fit(X_train_under, y_train_under)

print("Undersampled model fitted")
~~~

### 10.5 Evaluate on original test data

~~~python
under_predictions = under_model.predict(X_test)

print(
    classification_report(
        y_test,
        under_predictions,
        zero_division=0
    )
)
~~~

## 11. Strategy 4 — SMOTE

**SMOTE** means Synthetic Minority Over-sampling Technique.

Instead of only duplicating minority observations, SMOTE creates synthetic minority samples using neighbouring minority observations.

Benefits:
- Can increase minority representation without exact duplication.
- May help some models learn a broader minority region.

Limitations:
- Synthetic observations may not represent valid real-world cases.
- It can create overlapping or unrealistic samples.
- It is not appropriate for every feature type or data structure.
- It can be sensitive to noisy or outlier minority observations.
- It must be applied only to training data.

### 11.1 Import SMOTE

~~~python
from imblearn.over_sampling import SMOTE

print("SMOTE imported")
~~~

### 11.2 Create SMOTE sampler

~~~python
smote = SMOTE(
    random_state=42,
    k_neighbors=5
)

print(smote)
~~~

**Explanation**
k_neighbors controls the number of nearby minority observations used when generating synthetic samples. The value must be suitable for the number of minority observations available in the training fold.

### 11.3 Apply SMOTE to training data

~~~python
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print(pd.Series(y_train_smote).value_counts())
~~~

### 11.4 Fit a model using SMOTE data

~~~python
smote_model = LogisticRegression(max_iter=1000)

smote_model.fit(X_train_smote, y_train_smote)

print("SMOTE model fitted")
~~~

### 11.5 Evaluate on the original test data

~~~python
smote_predictions = smote_model.predict(X_test)

print(
    classification_report(
        y_test,
        smote_predictions,
        zero_division=0
    )
)
~~~

**Explanation**
The test data was not resampled. This provides a more realistic estimate of how the model performs on the original class distribution.

## 12. Use an imbalanced-learn Pipeline

When cross-validating, resampling must occur separately inside each training fold. Applying SMOTE to the entire training dataset before cross-validation can allow synthetic samples to be influenced by observations later used for validation.

Use imbalanced-learn's Pipeline to keep resampling inside each fold.

### 12.1 Import Pipeline

~~~python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

print("Imbalanced-learn Pipeline ready")
~~~

### 12.2 Build the pipeline

~~~python
smote_pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])

print(smote_pipeline)
~~~

**Explanation**
- Scaling is fitted on the training fold.
- SMOTE resamples that training fold.
- The classifier learns from the resampled fold.
- The validation fold is transformed but not resampled.

### 12.3 Evaluate using stratified cross-validation

~~~python
from sklearn.model_selection import StratifiedKFold, cross_val_score

stratified_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    smote_pipeline,
    X_train,
    y_train,
    cv=stratified_cv,
    scoring="average_precision"
)

print("Fold Average Precision:", cv_scores)
print("Mean Average Precision:", cv_scores.mean())
~~~

**Explanation**
The pipeline applies scaling and SMOTE independently inside each training fold. Average Precision is useful to inspect positive-class ranking behaviour.

## 13. Compare strategies fairly

Compare:
- Baseline model.
- Class-weighted model.
- Random oversampling.
- Random undersampling.
- SMOTE.

Use:
- Same original test observations.
- Same positive-class definition.
- Precision, recall and F1 for the minority class.
- PR curve or Average Precision.
- False-positive and false-negative counts.
- Business review capacity and error costs.

Do not choose the strategy that produces the most balanced training data. Choose the strategy that performs best for the real objective on representative unseen data.

## 14. Threshold tuning after resampling

Resampling changes the class distribution seen during training. Predicted probability calibration may therefore differ from the original population.

Important:
- Do not assume a default threshold of 0.50 remains suitable.
- Inspect scores on validation data that retains the original class distribution.
- Choose the threshold based on the cost of errors and operational constraints.
- Evaluate the locked threshold on the untouched test set.
- Consider probability calibration when probability values drive decisions.

## 15. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Resample before train-test split | Split first; resample training data only |
| Resample the test set | Keep test data at original prevalence |
| Assume SMOTE always improves performance | Compare with baseline and class weights |
| Duplicate minority records without validation | Monitor overfitting |
| Undersample without considering information loss | Review whether removed majority data is important |
| Apply SMOTE before cross-validation | Put it inside an imbalanced-learn Pipeline |
| Report accuracy only | Report minority-class precision, recall, F1 and AP |
| Assume balanced training means balanced production | Evaluate using realistic production prevalence |
| Use SMOTE blindly on mixed categorical data | Choose a method suitable for the feature types |

## 16. Interview-ready explanations

**What is class imbalance?**
- One target class has many more observations than another.
- It can cause a model to favour the majority class.
- Accuracy may hide poor minority-class detection.

**What is class weighting?**
- Giving different training importance to different classes.
- It can increase the learning impact of minority examples without creating synthetic records.

**Random oversampling versus undersampling?**
- Oversampling duplicates minority observations.
- Undersampling removes majority observations.
- Oversampling can increase overfitting; undersampling can discard useful information.

**What is SMOTE?**
- A method that creates synthetic minority samples using neighbouring minority observations.
- It must be applied only to training data.
- Synthetic records may not always be realistic.

## 17. Scenario-based interview practice

### Scenario 1: Fraud dataset has 99% normal and 1% fraud
**Expected response:**
- Inspect the class distribution and establish a baseline.
- Evaluate fraud precision, recall and Average Precision.
- Compare class weighting and resampling.
- Keep the test set at realistic prevalence.
- Select threshold based on fraud loss and review capacity.

### Scenario 2: SMOTE gives excellent training results but weak test results
**Expected response:**
- Check overfitting and synthetic sample quality.
- Confirm resampling was performed only inside training folds.
- Review whether minority examples contain noise or outliers.
- Compare with class weighting and the baseline.

### Scenario 3: Random undersampling improves recall but reduces precision
**Expected response:**
- The model catches more positives but creates more false alerts.
- Quantify both changes using the confusion matrix.
- Check review capacity and false-negative cost.
- Decide whether the trade-off is acceptable for the business.

### Scenario 4: A team balances the entire dataset before splitting
**Expected response:**
- This can leak information and create an unrealistic test distribution.
- Split first.
- Apply sampling only within training data or within a training-fold Pipeline.
- Evaluate on untouched original test data.

## 18. Lesson completion checklist

- [ ] I can identify class imbalance.
- [ ] I can explain why accuracy can be misleading.
- [ ] I can train a baseline classifier.
- [ ] I understand class weights.
- [ ] I can apply random oversampling and undersampling.
- [ ] I can explain SMOTE and its limitations.
- [ ] I can use an imbalanced-learn Pipeline.
- [ ] I can evaluate on the original test distribution.
- [ ] I can select a strategy based on business costs.

**Next lesson:** Day 52 — Model Bias, Variance and Overfitting.
