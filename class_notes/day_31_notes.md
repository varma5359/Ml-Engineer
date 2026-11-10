# Day 31: Support Vector Machines (SVM)

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain Support Vector Machines in simple language.
- Understand a decision boundary and margin.
- Explain what support vectors are.
- Distinguish linear and kernel-based SVM.
- Understand the role of C and gamma.
- Scale numerical features before SVM training.
- Train SVC using scikit-learn.
- Compare linear and RBF kernels.
- Tune parameters using validation or cross-validation.
- Recognize SVM strengths and limitations.

## 2. What is a Support Vector Machine?

A **Support Vector Machine (SVM)** is a supervised learning algorithm used for classification and regression. For classification, it tries to find a decision boundary that separates classes while maintaining a useful margin between the boundary and nearby training observations.

Beginner explanation:
- The model looks at observations from different classes.
- It searches for a separating boundary.
- Important nearby observations help define that boundary.
- The model uses the learned boundary to classify new observations.

SVM can use different kernels to represent more complex boundaries.

## 3. What are support vectors?

**Support vectors** are training observations that are important in defining the decision boundary. They are typically the observations closest to the separating boundary.

If the support vectors change, the learned boundary can change.

## 4. What is a margin?

The **margin** is the separation around the decision boundary. SVM aims to find a boundary with a useful margin while managing classification errors according to its settings.

A larger margin can help generalization, but the model must balance margin size against training errors.

## 5. Linear SVM and kernel SVM

### Linear SVM

Uses a linear decision boundary in the original feature representation. It can work well when classes are approximately separable with a straight boundary.

### Kernel SVM

Uses a kernel function to model nonlinear relationships. A common option is the Radial Basis Function (RBF) kernel.

The kernel approach can represent curved boundaries without manually creating every nonlinear feature.

## 6. Create a classification dataset

### 6.1 Import libraries

### 6.1. Import Pandas, NumPy and plotting tools

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

These libraries prepare and visualize the sample classification data.



### 6.2 Generate a two-class dataset

### 6.2. Create synthetic classification observations

```python
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=400,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=42
)

print(X_values.shape)
```

**Expected output**

```text
(400, 2)
```

**Explanation**

Two features make it possible to visualize the decision boundary.



### 6.3 Create a DataFrame

### 6.3. Name the feature columns

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
A five-row preview with two features and target
```

**Explanation**

The target is the class label. The features remain separate from the target during model fitting.



### 6.4 Visualize the classes

### 6.4. Plot the two classes

```python
sns.scatterplot(
    data=df,
    x="feature_1",
    y="feature_2",
    hue="target"
)
plt.title("SVM classification data")
plt.show()
```

**Expected output**

```text
A scatter plot with observations colored by class
```

**Explanation**

The plot gives an initial view of how easily the classes can be separated.



## 7. Split the data

### 7.1 Create X and y

### 7.1. Separate features and target

```python
X = df[["feature_1", "feature_2"]]
y = df["target"]

print(X.shape, y.shape)
```

**Expected output**

```text
(400, 2) and (400,)
```

**Explanation**

The model receives two features and learns to predict one class label.



### 7.2 Create training and test sets

### 7.2. Split the data

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
(320, 2) and (80, 2)
```

**Explanation**

The test data is held back. Stratification maintains approximate class proportions.



## 8. Why feature scaling is important for SVM

SVM depends on feature geometry and margins. A feature with a much larger scale can dominate the model's distance or margin calculations.

Scaling is usually an important preprocessing step for SVM.

### 8.1 Inspect feature ranges

### 8.1. Review training feature ranges

```python
print(X_train.describe())
```

**Expected output**

```text
Summary statistics for both features
```

**Explanation**

Check units and ranges before training.



### 8.2 Create a StandardScaler pipeline

### 8.2. Initialize scaling and SVM

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

linear_svm = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", SVC(kernel="linear"))
])

print(linear_svm)
```

**Expected output**

```text
A Pipeline with StandardScaler and linear SVC
```

**Explanation**

The scaler is fitted on training data and applied consistently to validation/test data.



## 9. Train a linear SVM classifier

### 9.1 Fit the linear SVM

### 9.1. Train the linear SVM pipeline

```python
linear_svm.fit(X_train, y_train)

print("Linear SVM fitted")
```

**Expected output**

```text
Linear SVM fitted
```

**Explanation**

The pipeline scales the training features and fits the linear SVM classifier.



### 9.2 Predict test labels

### 9.2. Generate linear SVM predictions

```python
linear_predictions = linear_svm.predict(X_test)

print(linear_predictions[:15])
```

**Expected output**

```text
The first fifteen predicted class labels
```

**Explanation**

The classifier uses its learned linear decision boundary.



### 9.3 Evaluate accuracy

### 9.3. Calculate accuracy

```python
from sklearn.metrics import accuracy_score

linear_accuracy = accuracy_score(y_test, linear_predictions)

print(linear_accuracy)
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

Accuracy is a basic summary. Inspect precision, recall and the confusion matrix where class-specific errors matter.



### 9.4 Print classification report

### 9.4. Inspect class-level performance

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    linear_predictions,
    zero_division=0
))
```

**Expected output**

```text
Precision, recall, F1 and support for both classes
```

**Explanation**

The report shows whether the model performs differently across classes.



## 10. Train an RBF kernel SVM

The RBF kernel can represent nonlinear decision boundaries.

### 10.1 Create an RBF SVM pipeline

### 10.1. Initialize RBF SVC

```python
rbf_svm = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", SVC(kernel="rbf"))
])

print(rbf_svm)
```

**Expected output**

```text
A pipeline with StandardScaler and RBF SVC
```

**Explanation**

The RBF kernel can represent nonlinear boundaries. Scaling is important because kernel calculations depend on feature distances.



### 10.2 Fit RBF SVM

### 10.2. Train the RBF model

```python
rbf_svm.fit(X_train, y_train)

print("RBF SVM fitted")
```

**Expected output**

```text
RBF SVM fitted
```

**Explanation**

The model learns a nonlinear decision boundary based on the scaled training data.



### 10.3 Predict test labels

### 10.3. Generate RBF predictions

```python
rbf_predictions = rbf_svm.predict(X_test)

print(rbf_predictions[:15])
```

**Expected output**

```text
The first fifteen predicted labels
```

**Explanation**

The RBF classifier applies its learned nonlinear decision function.



### 10.4 Evaluate RBF accuracy

### 10.4. Calculate RBF accuracy

```python
rbf_accuracy = accuracy_score(y_test, rbf_predictions)

print(rbf_accuracy)
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

Compare with the linear SVM using the same held-out observations.



### 10.5 Compare linear and RBF SVM

### 10.5. Create a kernel comparison

```python
kernel_comparison = pd.DataFrame({
    "kernel": ["linear", "rbf"],
    "accuracy": [linear_accuracy, rbf_accuracy]
})

print(kernel_comparison)
```

**Expected output**

```text
A two-row comparison table
```

**Explanation**

A nonlinear kernel is not automatically better. Compare validation results, complexity and operational requirements.



## 11. Understand C

**C** controls how strongly the SVM penalizes training classification errors.

Beginner interpretation:
- Smaller C allows a softer margin and may tolerate more training errors.
- Larger C penalizes training errors more strongly and may fit training observations more closely.
- Very large C can increase overfitting risk.
- Very small C can underfit.

### 11.1 Train a lower-C linear SVM

### 11.1. Create a model with smaller C

```python
low_c_svm = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", SVC(kernel="linear", C=0.1))
])

low_c_svm.fit(X_train, y_train)

print("Low-C model fitted")
```

**Expected output**

```text
Low-C model fitted
```

**Explanation**

A smaller C allows a softer margin and more tolerance for training errors.



### 11.2 Train a higher-C linear SVM

### 11.2. Create a model with larger C

```python
high_c_svm = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", SVC(kernel="linear", C=10))
])

high_c_svm.fit(X_train, y_train)

print("High-C model fitted")
```

**Expected output**

```text
High-C model fitted
```

**Explanation**

A larger C penalizes training errors more strongly. It may fit training data more closely.



### 11.3 Compare C settings

### 11.3. Evaluate different C values

```python
for name, estimator in [
    ("C=0.1", low_c_svm),
    ("C=1", linear_svm),
    ("C=10", high_c_svm)
]:
    print(name, estimator.score(X_test, y_test))
```

**Expected output**

```text
Accuracy values for three C settings
```

**Explanation**

Use validation or cross-validation for parameter selection. This test comparison is illustrative only; do not tune repeatedly on the final test set.



## 12. Understand gamma for RBF kernel

**Gamma** controls how far the influence of an individual training observation reaches in an RBF kernel.

Beginner interpretation:
- Low gamma gives each observation a broader influence and can create smoother boundaries.
- High gamma gives observations more local influence and can create more complex boundaries.
- Very high gamma may overfit.
- Very low gamma may underfit.

### 12.1 Train with low gamma

### 12.1. Create a low-gamma model

```python
low_gamma_svm = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", SVC(kernel="rbf", gamma=0.01, C=1))
])

low_gamma_svm.fit(X_train, y_train)

print("Low-gamma model fitted")
```

**Expected output**

```text
Low-gamma model fitted
```

**Explanation**

A low gamma creates a broader influence for each training observation.



### 12.2 Train with high gamma

### 12.2. Create a high-gamma model

```python
high_gamma_svm = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", SVC(kernel="rbf", gamma=5, C=1))
])

high_gamma_svm.fit(X_train, y_train)

print("High-gamma model fitted")
```

**Expected output**

```text
High-gamma model fitted
```

**Explanation**

A high gamma can create very local and complex decision regions.



### 12.3 Compare gamma settings

### 12.3. Evaluate gamma choices

```python
for name, estimator in [
    ("gamma=0.01", low_gamma_svm),
    ("gamma=scale", rbf_svm),
    ("gamma=5", high_gamma_svm)
]:
    print(name, estimator.score(X_test, y_test))
```

**Expected output**

```text
Accuracy values for three gamma settings
```

**Explanation**

Compare on validation data for actual tuning. The test set should remain a final independent check.



## 13. Visualize a decision boundary

### 13.1 Create a plotting grid

### 13.1. Find scaled training ranges

```python
scaled_train = linear_svm.named_steps["scale"].transform(X_train)

x1 = np.linspace(scaled_train[:, 0].min() - 0.5, scaled_train[:, 0].max() + 0.5, 150)
x2 = np.linspace(scaled_train[:, 1].min() - 0.5, scaled_train[:, 1].max() + 0.5, 150)

xx, yy = np.meshgrid(x1, x2)

print(xx.shape)
```

**Expected output**

```text
A two-dimensional mesh shape
```

**Explanation**

The grid covers the scaled feature space represented by training observations.



### 13.2 Create grid points

### 13.2. Combine coordinates

```python
grid_scaled = np.c_[xx.ravel(), yy.ravel()]

print(grid_scaled.shape)
```

**Expected output**

```text
A two-column array of grid points
```

**Explanation**

Each row represents a location in scaled feature space.



### 13.3 Predict linear SVM regions

### 13.3. Predict grid classes

```python
linear_region = linear_svm.named_steps["classifier"].predict(grid_scaled)

linear_region = linear_region.reshape(xx.shape)

print(linear_region.shape)
```

**Expected output**

```text
A two-dimensional class region
```

**Explanation**

The classifier step receives already scaled grid points, because the grid was created in scaled feature space.



### 13.4 Plot linear decision boundary

### 13.4. Visualize linear SVM

```python
plt.contourf(xx, yy, linear_region, alpha=0.3)
plt.scatter(
    scaled_train[:, 0],
    scaled_train[:, 1],
    c=y_train,
    edgecolor="black"
)
plt.xlabel("Scaled feature 1")
plt.ylabel("Scaled feature 2")
plt.title("Linear SVM decision boundary")
plt.show()
```

**Expected output**

```text
A plot with a linear separation region
```

**Explanation**

The linear kernel produces a straight boundary in this two-feature representation.



## 14. Support vectors

### 14.1 Inspect support vector count

### 14.1. Read number of support vectors

```python
print(linear_svm.named_steps["classifier"].n_support_)
```

**Expected output**

```text
A count of support vectors for each class
```

**Explanation**

Support vectors are observations that help define the separating boundary.



### 14.2 Inspect support vector indices

### 14.2. Read support vector positions

```python
print(linear_svm.named_steps["classifier"].support_[:10])
```

**Expected output**

```text
The first ten support-vector indices
```

**Explanation**

These indices refer to positions in the training data used to fit the classifier.



### 14.3 Understand support vector importance

The support vectors are not necessarily the most important business records. They are important to the mathematical boundary learned by SVM. Do not interpret them as causal drivers or automatically as high-risk customers.

## 15. Probability estimates in SVM

SVC does not enable probability estimates by default. If probability estimates are needed, the estimator can be configured to estimate them, but this adds fitting overhead and probabilities should be assessed for calibration.

### 15.1 Create SVC with probability enabled

### 15.1. Initialize probability-enabled SVC

```python
probability_svm = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

print(probability_svm)
```

**Expected output**

```text
A pipeline with probability-enabled SVC
```

**Explanation**

The probability option enables probability estimates. It may increase training time.



### 15.2 Fit probability-enabled SVM

### 15.2. Train the model

```python
probability_svm.fit(X_train, y_train)

print("Probability SVM fitted")
```

**Expected output**

```text
Probability SVM fitted
```

**Explanation**

The model is trained using the training subset.



### 15.3 Generate class probabilities

### 15.3. Predict probabilities

```python
svm_probabilities = probability_svm.predict_proba(X_test)

print(svm_probabilities[:5])
```

**Expected output**

```text
A probability matrix for the first five test observations
```

**Explanation**

Probability estimates should not be assumed to be calibrated automatically. Assess calibration if they drive decisions.



## 16. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Train SVM without considering feature scale | Use scaling inside a pipeline |
| Assume linear kernel always works | Compare suitable kernels using validation |
| Choose high C automatically | Tune C based on generalization |
| Choose high gamma for best training fit | Inspect validation and overfitting |
| Tune on test data | Use validation or cross-validation |
| Assume support vectors are business-important records | Interpret them as boundary-defining observations |
| Expect SVC probabilities by default | Enable probability estimates explicitly when needed |
| Use SVM on very large data without benchmarking | Measure training and inference cost |

## 17. Interview-ready explanations

**What is SVM?**  
A supervised algorithm that finds a decision boundary with a useful margin between classes. Kernel methods allow it to represent nonlinear boundaries.

**What are support vectors?**  
Training observations that are important in defining the decision boundary, typically those closest to it.

**What does C control?**  
It controls how strongly the model penalizes training classification errors. Larger C fits training observations more strictly; smaller C allows a softer margin.

**What does gamma control in RBF SVM?**  
It controls the locality of each training observation's influence. High gamma can create complex boundaries; low gamma creates smoother boundaries.

**Why scale features for SVM?**  
The margin and kernel calculations depend on feature geometry, so large feature ranges can dominate if inputs are not scaled.

## 18. Scenario questions

1. SVM performs poorly when income has a much larger range than age. What do you do?  
   **Answer:** Scale features using a pipeline fitted on training data.

2. Training accuracy is high but validation accuracy is low with a high gamma. What might be happening?  
   **Answer:** The model may be overfitting with overly local decision regions.

3. A linear SVM underfits but RBF performs better on validation. What might that indicate?  
   **Answer:** The class relationship may be nonlinear and the RBF kernel captures it better.

4. Why should C and gamma be selected using validation data?  
   **Answer:** They control model complexity. Choosing them using test performance would make the test set part of development.

5. SVM training is slow on a very large dataset. What do you consider?  
   **Answer:** Benchmark computational cost, simplify the model, use an appropriate implementation or compare with algorithms that scale better for the task.

## 19. Lesson completion checklist

- [ ] I can explain SVM and its margin.
- [ ] I understand support vectors.
- [ ] I can distinguish linear and RBF kernels.
- [ ] I understand C and gamma.
- [ ] I can build a scaled SVM pipeline.
- [ ] I can compare kernels and settings.
- [ ] I can visualize a decision boundary.
- [ ] I understand SVM probability settings and limitations.

**Next lesson:** Day 32 — Supervised Model Comparison and Model Selection.
