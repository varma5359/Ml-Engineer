# Day 25: Logistic Regression and Sigmoid Function

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain why logistic regression is used for classification.
- Distinguish classification from regression.
- Understand a model score and sigmoid probability in plain language.
- Train a binary logistic regression classifier.
- Use predict() and predict_proba().
- Interpret predicted probabilities carefully.
- Understand the default classification threshold.
- Recognize class imbalance and feature scaling considerations.
- Evaluate a classifier using a confusion matrix and suitable metrics.
- Explain why a probability is not a guarantee.

## 2. What is logistic regression?

Despite the word "regression" in its name, **logistic regression** is commonly used for classification.

It estimates the probability that an observation belongs to a class. For binary classification, the classes are often represented as:
- 0: negative class.
- 1: positive class.

Examples:
- Customer will churn / will not churn.
- Email is spam / not spam.
- Transaction is fraudulent / not fraudulent.
- Loan application is approved / not approved.

The model produces a score, converts it into a value between zero and one using the logistic or sigmoid function, and uses a decision threshold to assign a class.

## 3. Understand the sigmoid function

The **sigmoid function** converts a real-valued model score into a value between zero and one.

In simple terms:
- A value close to zero indicates a lower estimated probability for the positive class.
- A value close to one indicates a higher estimated probability for the positive class.
- A value around the middle indicates uncertainty under the model's current estimate.

The output can be interpreted as a probability only when the model is appropriate and its probability estimates are sufficiently calibrated.

**Important:** A probability of 0.80 does not guarantee that one specific customer will churn.

## 4. Create a customer churn dataset

### 4.1 Import libraries

### 4.1. Import Pandas, NumPy and plotting tools

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

Pandas and NumPy prepare the data. Matplotlib and Seaborn help inspect distributions and class separation.



### 4.2 Create customer observations

### 4.2. Build a small churn dataset

```python
df = pd.DataFrame({
    "tenure_months": [2, 4, 6, 8, 10, 12, 15, 18, 20, 24,
                      28, 30, 34, 36, 40, 42, 45, 48, 50, 54,
                      3, 7, 11, 16, 22, 27, 33, 39, 46, 52],
    "monthly_spend": [1200, 1500, 1800, 2000, 2200, 2500, 3000, 3200, 3500, 3800,
                      4200, 4500, 4800, 5000, 5300, 5500, 5800, 6000, 6200, 6500,
                      1300, 1900, 2400, 3100, 3900, 4300, 4700, 5200, 5900, 6400],
    "support_calls": [5, 4, 4, 3, 4, 3, 2, 2, 1, 2,
                      1, 1, 0, 1, 0, 1, 0, 0, 0, 0,
                      5, 4, 3, 3, 2, 2, 1, 1, 0, 0],
    "contract_type": ["Monthly", "Monthly", "Monthly", "Monthly", "Monthly",
                      "Annual", "Monthly", "Annual", "Annual", "Annual",
                      "Annual", "Annual", "Annual", "Annual", "Annual",
                      "Annual", "Annual", "Annual", "Annual", "Annual",
                      "Monthly", "Monthly", "Monthly", "Annual", "Monthly",
                      "Annual", "Annual", "Annual", "Annual", "Annual"],
    "churn": [1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
})

print(df.head())
```

**Expected output**

```text
The first five customer records
```

**Explanation**

The dataset contains numerical and categorical features and a binary churn target. It is constructed for learning, not for business conclusions.



### 4.3 Inspect target classes

### 4.3. Count churn labels

```python
print(df["churn"].value_counts())
```

**Expected output**

```text
Counts for class 0 and class 1
```

**Explanation**

Check the number of examples in each class before splitting and fitting a classifier.



### 4.4 Inspect target proportions

### 4.4. Calculate class proportions

```python
print(df["churn"].value_counts(normalize=True))
```

**Expected output**

```text
Proportions for churn and non-churn
```

**Explanation**

Class proportions help identify imbalance. The default accuracy metric may be misleading when classes are highly imbalanced.



### 4.5 Visualize target distribution

### 4.5. Plot churn counts

```python
sns.countplot(data=df, x="churn")
plt.xlabel("Churn label")
plt.ylabel("Customer count")
plt.title("Churn class distribution")
plt.show()
```

**Expected output**

```text
A bar chart showing the count of each churn class
```

**Explanation**

This gives a quick visual summary of the target distribution.



## 5. Separate features and target

### 5.1 Select numerical and categorical features

### 5.1. Create feature table X

```python
feature_columns = [
    "tenure_months",
    "monthly_spend",
    "support_calls",
    "contract_type"
]

X = df[feature_columns]
y = df["churn"]

print(X.head())
```

**Expected output**

```text
A feature table containing four input columns
```

**Explanation**

The target is excluded from X. Contract type is categorical and requires encoding.



### 5.2 Check feature and target shapes

### 5.2. Inspect dimensions

```python
print(X.shape)
print(y.shape)
```

**Expected output**

```text
(30, 4) and (30,)
```

**Explanation**

There are 30 observations and four feature columns, with one target per observation.



## 6. Split the data

### 6.1 Import train_test_split

### 6.1. Import the split function

```python
from sklearn.model_selection import train_test_split

print("Split tool ready")
```

**Expected output**

```text
Split tool ready
```

**Explanation**

The function separates training and test observations while keeping features and labels aligned.



### 6.2 Create a stratified split

### 6.2. Split features and target

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)
print(y_train.value_counts())
print(y_test.value_counts())
```

**Expected output**

```text
Training and test shapes plus class counts
```

**Explanation**

Stratification preserves approximate class proportions. It is feasible here because each class has enough examples for the split.



## 7. Build preprocessing for mixed features

Logistic regression needs numerical inputs. We will scale numerical columns and one-hot encode the contract category.

### 7.1 Identify feature groups

### 7.1. Create numeric and categorical column lists

```python
numeric_features = [
    "tenure_months",
    "monthly_spend",
    "support_calls"
]

categorical_features = ["contract_type"]

print(numeric_features)
print(categorical_features)
```

**Expected output**

```text
Two lists of feature names
```

**Explanation**

The lists define which columns will be scaled and which will be encoded.



### 7.2 Create numerical preprocessing

### 7.2. Build a numeric scaler

```python
from sklearn.preprocessing import StandardScaler

numeric_transformer = StandardScaler()

print(numeric_transformer)
```

**Expected output**

```text
StandardScaler()
```

**Explanation**

Scaling helps numerical features use comparable magnitudes for logistic regression optimization.



### 7.3 Create categorical preprocessing

### 7.3. Build a one-hot encoder

```python
from sklearn.preprocessing import OneHotEncoder

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

print(categorical_transformer)
```

**Expected output**

```text
OneHotEncoder(handle_unknown='ignore')
```

**Explanation**

The encoder learns categories during fit. The unknown-category setting avoids errors for unseen categories at inference time.



### 7.4 Combine using ColumnTransformer

### 7.4. Create the preprocessor

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

print(preprocessor)
```

**Expected output**

```text
A ColumnTransformer with numeric and categorical branches
```

**Explanation**

Each branch transforms only its selected columns.



## 8. Train logistic regression

### 8.1 Import LogisticRegression

### 8.1. Import the classifier

```python
from sklearn.linear_model import LogisticRegression

print("LogisticRegression imported")
```

**Expected output**

```text
LogisticRegression imported
```

**Explanation**

The estimator is used for classification. The max_iter setting can be increased if the optimization process needs more iterations.



### 8.2 Build the full pipeline

### 8.2. Connect preprocessing and classifier

```python
from sklearn.pipeline import Pipeline

logistic_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print(logistic_pipeline)
```

**Expected output**

```text
A Pipeline containing preprocessing and logistic regression
```

**Explanation**

The pipeline ensures preprocessing is fitted on training data and reused for prediction.



### 8.3 Fit the pipeline

### 8.3. Train the classifier

```python
logistic_pipeline.fit(X_train, y_train)

print("Logistic regression fitted")
```

**Expected output**

```text
Logistic regression fitted
```

**Explanation**

The pipeline learns scaling, category encoding and classifier parameters using training observations only.



### 8.4 Predict class labels

### 8.4. Use predict()

```python
predicted_classes = logistic_pipeline.predict(X_test)

print(predicted_classes)
```

**Expected output**

```text
Predicted class labels containing 0 or 1
```

**Explanation**

predict() returns the final class labels using the estimator's default decision rule.



### 8.5 Predict probabilities

### 8.5. Use predict_proba()

```python
probabilities = logistic_pipeline.predict_proba(X_test)

print(probabilities)
```

**Expected output**

```text
A two-column probability array
```

**Explanation**

Each row contains probabilities for the two classes. The columns follow the order in classifier.classes_.



### 8.6 Inspect class order

### 8.6. Read class labels

```python
print(logistic_pipeline.named_steps["classifier"].classes_)
```

**Expected output**

```text
[0 1]
```

**Explanation**

This confirms which probability column corresponds to each class. Do not assume column one always represents the positive class without checking classes_.



### 8.7 Select the positive-class probability

### 8.7. Extract probability for churn class 1

```python
positive_probabilities = probabilities[:, 1]

print(positive_probabilities)
```

**Expected output**

```text
A one-dimensional array of churn probabilities
```

**Explanation**

The second column corresponds to class 1 in this dataset because classes_ is [0, 1].



## 9. Understand the sigmoid output

A sigmoid transformation maps the model's internal score to a value between zero and one.

For example:
- 0.10 means a low estimated probability for the positive class.
- 0.50 is the default decision boundary in many binary classifiers.
- 0.90 means a high estimated probability for the positive class.

The probability is an estimate. It should be checked for calibration and evaluated on representative data.

### 9.1 Display probability with predicted class

### 9.1. Create a prediction report

```python
prediction_report = pd.DataFrame({
    "actual_churn": y_test.values,
    "predicted_churn": predicted_classes,
    "churn_probability": positive_probabilities
})

print(prediction_report)
```

**Expected output**

```text
A table of actual class, predicted class and positive-class probability
```

**Explanation**

The report helps explain how probability estimates are converted into class labels.



### 9.2 Sort by predicted probability

### 9.2. Rank test observations by churn probability

```python
print(
    prediction_report.sort_values(
        "churn_probability",
        ascending=False
    )
)
```

**Expected output**

```text
Rows ordered from highest to lowest estimated churn probability
```

**Explanation**

Ranking can help prioritize cases, but an operational threshold must be selected based on business needs.



## 10. Evaluate classification predictions

### 10.1 Calculate accuracy

### 10.1. Use accuracy_score()

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predicted_classes)

print(accuracy)
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

Accuracy is the fraction of correct class predictions. It can hide minority-class failures on imbalanced data.



### 10.2 Create a confusion matrix

### 10.2. Calculate the confusion matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predicted_classes)

print(cm)
```

**Expected output**

```text
A two-by-two confusion matrix
```

**Explanation**

The matrix compares actual and predicted classes. The row and column order follows the labels used by the metric.



### 10.3 Plot the confusion matrix

### 10.3. Visualize confusion matrix

```python
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[0, 1],
    yticklabels=[0, 1]
)
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title("Churn confusion matrix")
plt.show()
```

**Expected output**

```text
A heatmap of true and incorrect class predictions
```

**Explanation**

This helps identify false positives and false negatives. Later lessons cover the metrics in detail.



### 10.4 Print classification report

### 10.4. Inspect precision, recall and F1

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    predicted_classes,
    zero_division=0
))
```

**Expected output**

```text
A class-wise classification report
```

**Explanation**

Precision and recall provide more information than accuracy alone. Interpret them based on the cost of different errors.



## 11. Interpret logistic regression coefficients

The classifier learns a coefficient for each transformed feature. Coefficients influence the model's internal score, not directly the final class label.

After encoding, the model may have several indicator columns for a categorical feature. Keep feature names aligned with coefficients.

### 11.1 Inspect transformed feature names

### 11.1. Get feature names from preprocessing

```python
transformed_names = logistic_pipeline.named_steps[
    "preprocess"
].get_feature_names_out()

print(transformed_names)
```

**Expected output**

```text
Names for scaled numeric and encoded categorical features
```

**Explanation**

The feature names identify the representation used by the classifier.



### 11.2 Create a coefficient table

### 11.2. Pair names with coefficients

```python
coefficients = logistic_pipeline.named_steps[
    "classifier"
].coef_[0]

coefficient_table = pd.DataFrame({
    "feature": transformed_names,
    "coefficient": coefficients
})

print(coefficient_table)
```

**Expected output**

```text
A table containing each transformed feature and its coefficient
```

**Explanation**

Coefficients describe changes in the model's internal score. Do not treat them as direct probability changes or causal effects.



## 12. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Assume logistic regression is a regression prediction task | It is commonly used for classification | Identify whether target is numerical or categorical |
| Treat probability as certainty | Individual outcomes are uncertain | Explain probability as an estimate |
| Assume predict_proba column one is positive class | Class ordering can vary | Inspect classes_ |
| Ignore class imbalance | Accuracy can hide poor minority recall | Inspect confusion matrix and class metrics |
| Skip scaling where it matters | Optimization may be less stable | Use a fitted preprocessing pipeline |
| Encode categories inconsistently | Train and production columns may differ | Use a fitted encoder |
| Tune on test data repeatedly | Test becomes part of development | Use validation or cross-validation |
| Ignore calibration | Probability estimates may not match observed frequencies | Assess calibration where probabilities drive decisions |

## 13. Interview-ready explanations

**Why is logistic regression used for classification?**  
It estimates class probabilities by transforming a linear model score into a value between zero and one, then assigns a class using a decision threshold.

**What is the sigmoid function?**  
It maps a real-valued score into a range between zero and one. In binary logistic regression, the output is used as the estimated probability of the positive class.

**What is the difference between predict() and predict_proba()?**  
predict() returns class labels. predict_proba() returns probability estimates for each class.

**Does a probability of 0.90 guarantee the customer will churn?**  
No. It is an estimate. Its usefulness depends on model performance and probability calibration.

**Why inspect classes_?**  
It identifies the order of class labels used by predict_proba(), so the correct probability column is selected.

## 14. Scenario questions

1. The model predicts a churn probability of 0.82. Does that guarantee churn?  
   **Answer:** No. It is an estimated probability, not a guaranteed outcome.

2. The model predicts every customer as non-churn. What do you inspect?  
   **Answer:** Class balance, threshold, confusion matrix, recall and whether the model learned useful patterns.

3. Why might max_iter be increased?  
   **Answer:** To allow the optimizer more iterations. Also inspect scaling, convergence warnings and data quality.

4. A new contract category appears in production. What should happen?  
   **Answer:** Use a documented unknown-category policy in the fitted encoder and monitor the new category.

5. A probability model ranks customers well but its probabilities are overconfident. What should you assess?  
   **Answer:** Probability calibration and whether calibration methods are needed using validation data.

## 15. Lesson completion checklist

- [ ] I can explain logistic regression as a classifier.
- [ ] I understand the sigmoid output in plain language.
- [ ] I can create a mixed-feature preprocessing pipeline.
- [ ] I can fit LogisticRegression.
- [ ] I can use predict() and predict_proba().
- [ ] I know how to inspect classes_.
- [ ] I can create a confusion matrix.
- [ ] I can explain why probability is not certainty.
- [ ] I can identify class imbalance and calibration concerns.

**Next lesson:** Day 26 — Decision Boundaries and Classification Thresholds.
