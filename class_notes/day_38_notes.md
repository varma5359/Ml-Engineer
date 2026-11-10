# Day 38: Principal Component Analysis and Dimensionality Reduction

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain dimensionality reduction in simple language.
- Understand why datasets with many features can be difficult to work with.
- Explain Principal Component Analysis (PCA) conceptually.
- Understand principal components as new combined features.
- Scale numerical features before PCA when appropriate.
- Fit PCA and inspect explained variance.
- Select a number of components using cumulative explained variance.
- Visualize high-dimensional data in two dimensions.
- Understand information loss and interpretability trade-offs.
- Avoid fitting PCA on held-out data during model evaluation.

## 2. What is dimensionality reduction?

**Dimensionality reduction** reduces the number of input features while attempting to preserve useful information.

A dataset may contain:
- Hundreds of numerical measurements.
- Many correlated sensor readings.
- Large sets of image pixel features.
- Customer behaviour indicators.
- Text features with thousands of dimensions.

Too many features can increase computation, make visualization difficult and create redundant information.

Dimensionality reduction creates a smaller representation of the original data.

## 3. What is PCA?

**Principal Component Analysis (PCA)** is a dimensionality-reduction technique that creates new features called principal components.

Beginner explanation:
1. PCA examines how numerical features vary together.
2. It identifies directions that capture large amounts of variation.
3. It creates new combined features along those directions.
4. The first component captures the largest share of variation.
5. Later components capture additional variation while representing different directions.
6. We can keep a smaller number of components for analysis or modelling.

PCA does not select original columns. It creates new combinations of the original numerical features.

## 4. Important PCA vocabulary

| Term | Meaning |
|---|---|
| Feature | Original input column |
| Principal component | New combined feature created by PCA |
| Explained variance | Amount of variation represented by a component |
| Explained variance ratio | Share of total variation represented by a component |
| Cumulative explained variance | Total share represented by the first several components |
| Loading | Contribution of an original feature to a component |
| Projection | Representing observations in component space |
| Reconstruction | Approximate recovery of original features from retained components |

## 5. Create a high-dimensional dataset

### 5.1 Import libraries

### 5.1. Import Pandas, NumPy and plotting libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Ready")
```

**Expected output**

```text
Ready```

**Explanation**

These libraries prepare, transform and visualize the dataset.



### 5.2 Generate numerical features

### 5.2. Create a dataset with ten features

```python
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_repeated=0,
    random_state=42
)

print(X_values.shape)
```

**Expected output**

```text
(500, 10)```

**Explanation**

The dataset contains 500 observations and ten numerical features. Some features are redundant by construction.



### 5.3 Create a DataFrame

### 5.3. Add feature names

```python
feature_names = [f"feature_{i}" for i in range(1, 11)]

df = pd.DataFrame(X_values, columns=feature_names)
df["target"] = y

print(df.head())
```

**Expected output**

```text
A five-row preview with ten features and one target```

**Explanation**

The target is retained for later visualization but is not used to fit PCA in this example.



### 5.4 Inspect the feature correlation

### 5.4. Calculate feature correlations

```python
correlation = df[feature_names].corr()

print(correlation.round(2))
```

**Expected output**

```text
A ten-by-ten feature correlation matrix```

**Explanation**

Correlated features may contain overlapping information. PCA can create components that represent shared variation.



### 5.5 Visualize the correlation matrix

### 5.5. Plot a heatmap

```python
plt.figure(figsize=(10, 7))
sns.heatmap(correlation, cmap="coolwarm", center=0)
plt.title("Feature correlation heatmap")
plt.show()
```

**Expected output**

```text
A heatmap of pairwise feature correlations```

**Explanation**

The heatmap helps inspect relationships among original features before applying PCA.



## 6. Scale features before PCA

PCA is influenced by feature variance. If features use very different units, high-variance columns can dominate the components.

Scaling is commonly used when numerical features are measured on different scales.

### 6.1 Separate features and target

### 6.1. Create X and y

```python
X = df[feature_names]
y = df["target"]

print(X.shape, y.shape)
```

**Expected output**

```text
(500, 10) and (500,)```

**Explanation**

PCA is fitted to X only. The target is not used to create components.



### 6.2 Split the data

### 6.2. Create train and test subsets

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
(400, 10) and (100, 10)```

**Explanation**

The test data is held back. PCA and scaling will be fitted using training features only.



### 6.3 Create a scaler

### 6.3. Initialize StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print(scaler)
```

**Expected output**

```text
StandardScaler()```

**Explanation**

The scaler learns feature means and standard deviations during fit.



### 6.4 Fit scaler on training features

### 6.4. Scale training data

```python
X_train_scaled = scaler.fit_transform(X_train)

print(X_train_scaled[:3])
```

**Expected output**

```text
The first three standardized training observations```

**Explanation**

Only training features are used to learn scaling statistics.



### 6.5 Transform test features

### 6.5. Scale test data

```python
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled[:3])
```

**Expected output**

```text
The first three transformed test observations```

**Explanation**

The test data uses the training-fitted scaler. Do not fit a new scaler on the test set.



## 7. Fit PCA

### 7.1 Import PCA

### 7.1. Import PCA

```python
from sklearn.decomposition import PCA

print("PCA imported")
```

**Expected output**

```text
PCA imported```

**Explanation**

PCA is available through scikit-learn's decomposition module.



### 7.2 Create PCA with two components

### 7.2. Initialize two-component PCA

```python
pca_2 = PCA(n_components=2)

print(pca_2)
```

**Expected output**

```text
PCA(n_components=2)```

**Explanation**

The model will represent the ten original features using two new principal components.



### 7.3 Fit PCA on training features

### 7.3. Fit and transform training data

```python
X_train_pca = pca_2.fit_transform(X_train_scaled)

print(X_train_pca.shape)
```

**Expected output**

```text
(400, 2)```

**Explanation**

PCA learns component directions from training data and transforms the training observations into two-dimensional component space.



### 7.4 Transform test data

### 7.4. Apply fitted PCA to test data

```python
X_test_pca = pca_2.transform(X_test_scaled)

print(X_test_pca.shape)
```

**Expected output**

```text
(100, 2)```

**Explanation**

The test data is transformed using component directions learned from training data. PCA is not refitted on test observations.



### 7.5 Inspect the transformed observations

### 7.5. Preview principal component values

```python
print(X_train_pca[:5])
```

**Expected output**

```text
Five rows containing two component values each```

**Explanation**

The original ten features are represented by two new numerical component values.



## 8. Explained variance

PCA provides an explained variance ratio for each component. It indicates how much of the original feature variation is represented by that component.

### 8.1 Inspect explained variance ratio

### 8.1. Read the component variance ratios

```python
print(pca_2.explained_variance_ratio_)
```

**Expected output**

```text
Two values between zero and one```

**Explanation**

Each value is the share of variation represented by a component. Their total is the variation represented by the two-component representation.



### 8.2 Calculate total explained variance

### 8.2. Sum the two component ratios

```python
print(pca_2.explained_variance_ratio_.sum())
```

**Expected output**

```text
A value between zero and one```

**Explanation**

This indicates how much total variation is represented by the two retained components.



### 8.3 Inspect all component ratios

### 8.3. Fit PCA without limiting components

```python
pca_all = PCA()

pca_all.fit(X_train_scaled)

print(pca_all.explained_variance_ratio_)
```

**Expected output**

```text
A sequence of variance ratios for all ten components```

**Explanation**

This model learns all component directions so we can inspect how variation is distributed across them.



### 8.4 Calculate cumulative explained variance

### 8.4. Create cumulative variance

```python
cumulative_variance = np.cumsum(
    pca_all.explained_variance_ratio_
)

print(cumulative_variance)
```

**Expected output**

```text
A sequence that increases towards one```

**Explanation**

The cumulative values show how much total variation is represented by the first one, first two, first three components and so on.



### 8.5 Plot cumulative explained variance

### 8.5. Visualize cumulative variance

```python
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative explained variance")
plt.show()
```

**Expected output**

```text
A curve showing cumulative explained variance as components increase```

**Explanation**

The chart helps choose a component count that balances compression and retained variation. There is no universal cutoff for every task.



## 9. Choose a component count

A team may choose a component count based on:
- Cumulative explained variance.
- Downstream model validation performance.
- Computational requirements.
- Visualization needs.
- Interpretability requirements.

Do not select a component count only because it gives a round number of features.

### 9.1 Find the number of components for a chosen variance target

### 9.1. Find components reaching 90 percent

```python
components_for_90 = np.argmax(
    cumulative_variance >= 0.90
) + 1

print(components_for_90)
```

**Expected output**

```text
The number of leading components needed to reach the chosen variance target```

**Explanation**

The 90% target is an illustrative selection rule. A real project should validate the retained representation against the modelling objective.



### 9.2 Fit PCA using the selected count

### 9.2. Create PCA with selected components

```python
pca_selected = PCA(
    n_components=components_for_90
)

X_train_selected = pca_selected.fit_transform(X_train_scaled)
X_test_selected = pca_selected.transform(X_test_scaled)

print(X_train_selected.shape, X_test_selected.shape)
```

**Expected output**

```text
Training and test arrays with the selected component count```

**Explanation**

The PCA model is fitted on training features and reused for test features.



## 10. Visualize observations in two dimensions

### 10.1 Create a PCA visualization table

### 10.1. Combine components and target labels

```python
pca_view = pd.DataFrame(
    X_train_pca,
    columns=["PC1", "PC2"]
)

pca_view["target"] = y_train.reset_index(drop=True)

print(pca_view.head())
```

**Expected output**

```text
A table with PC1, PC2 and target```

**Explanation**

The target is added only for visualization. PCA itself was fitted without target labels.



### 10.2 Plot PCA components

### 10.2. Visualize two components by class

```python
sns.scatterplot(
    data=pca_view,
    x="PC1",
    y="PC2",
    hue="target"
)
plt.title("PCA projection of training observations")
plt.show()
```

**Expected output**

```text
A scatter plot of observations in two-component space```

**Explanation**

This shows whether classes appear separated in the reduced representation. PCA is unsupervised and does not guarantee class separation.



## 11. Understand component loadings

A principal component combines original features. Loadings help inspect which original features contribute strongly to each component.

### 11.1 Create a loading table

### 11.1. Inspect component loadings

```python
loading_table = pd.DataFrame(
    pca_all.components_,
    columns=feature_names,
    index=[f"PC{i}" for i in range(1, 11)]
)

print(loading_table.head())
```

**Expected output**

```text
A table showing the original-feature contributions for the first components```

**Explanation**

The component directions are combinations of original features. A loading's sign and magnitude help interpret the direction, but interpretation should consider all features together.



### 11.2 Inspect the first component

### 11.2. Sort first-component loadings

```python
print(
    loading_table.loc["PC1"].sort_values(
        key=lambda values: values.abs(),
        ascending=False
    )
)
```

**Expected output**

```text
Original features ordered by absolute loading for PC1```

**Explanation**

This helps identify which original features contribute most strongly to the first component.



### 11.3 Avoid treating components as business labels

A component such as PC1 is a mathematical combination of original features. It is not automatically a meaningful business concept such as "customer loyalty". Interpret it from the loading pattern and domain context.

## 12. Reconstruct original features

PCA can approximately reconstruct the original data from retained components. Using fewer components usually loses some information.

### 12.1 Transform training data with two components

### 12.1. Create a two-component projection

```python
pca_two = PCA(n_components=2)

train_two_components = pca_two.fit_transform(X_train_scaled)

print(train_two_components.shape)
```

**Expected output**

```text
(400, 2)```

**Explanation**

The ten-dimensional training representation is reduced to two dimensions.



### 12.2 Reconstruct the scaled features

### 12.2. Use inverse_transform()

```python
reconstructed_scaled = pca_two.inverse_transform(
    train_two_components
)

print(reconstructed_scaled.shape)
```

**Expected output**

```text
(400, 10)```

**Explanation**

The inverse transformation reconstructs an approximation of the original ten scaled features.



### 12.3 Calculate reconstruction error

### 12.3. Compare original and reconstructed data

```python
reconstruction_error = np.mean(
    (X_train_scaled - reconstructed_scaled) ** 2
)

print(reconstruction_error)
```

**Expected output**

```text
A non-negative reconstruction error```

**Explanation**

This measures information lost when representing the data with two components. More retained components generally reduce reconstruction error.



## 13. PCA inside an ML pipeline

When PCA is used before a supervised estimator, fit it inside a Pipeline. This ensures it learns component directions using training data during cross-validation.

### 13.1 Build a PCA and logistic regression pipeline

### 13.1. Create the pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pca_classifier = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=5)),
    ("model", LogisticRegression(max_iter=2000))
])

print(pca_classifier)
```

**Expected output**

```text
A Pipeline with scaler, PCA and logistic regression```

**Explanation**

The scaler and PCA are fitted within the training workflow before the classifier is trained.



### 13.2 Fit the pipeline

### 13.2. Train using original features

```python
pca_classifier.fit(X_train, y_train)

print("PCA classifier fitted")
```

**Expected output**

```text
PCA classifier fitted```

**Explanation**

The pipeline learns scaling and component directions from X_train, then fits logistic regression on the reduced features.



### 13.3 Predict test observations

### 13.3. Generate predictions

```python
pca_predictions = pca_classifier.predict(X_test)

print(pca_predictions[:15])
```

**Expected output**

```text
The first fifteen predicted class labels```

**Explanation**

The pipeline transforms the test features using the fitted scaler and PCA before predicting.



### 13.4 Evaluate the model

### 13.4. Calculate accuracy

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pca_predictions))
```

**Expected output**

```text
An accuracy value between zero and one```

**Explanation**

Compare this with a model using the original features under the same split and validation procedure. PCA does not always improve predictive performance.



## 14. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Fit PCA before train-test split | Fit PCA on training data only |
| Forget scaling when feature units differ | Scale before PCA where appropriate |
| Assume PC1 is a business feature | Inspect loadings and explain the combination |
| Keep only two components automatically | Select based on variance, validation and task needs |
| Assume PCA always improves accuracy | Compare downstream performance |
| Interpret explained variance as target relevance | PCA is unsupervised and does not use the target |
| Use PCA on unprepared mixed text/categories | Encode or select suitable numerical representations first |
| Ignore information loss | Inspect cumulative variance and reconstruction error |

## 15. Interview-ready explanations

**What is PCA?**  
It is an unsupervised dimensionality-reduction method that creates new combined features called principal components, ordered by how much variation they represent.

**Why use PCA?**  
It can reduce the number of numerical dimensions, help visualization, reduce redundancy and lower computational cost.

**Does PCA select original columns?**  
No. It creates new combinations of original features.

**What is explained variance ratio?**  
It describes the share of original feature variation represented by each principal component.

**Why fit PCA on training data only?**  
PCA learns component directions from the data. Fitting on held-out observations allows their information to influence preprocessing and can bias evaluation.

## 16. Scenario questions

1. A dataset has 500 numerical features. What might PCA help with?  
   **Answer:** It can create a smaller representation that retains a chosen share of variation, reducing dimensionality for exploration or modelling.

2. PCA retains 95% of variation but model accuracy decreases. Is that possible?  
   **Answer:** Yes. PCA preserves overall feature variation, not necessarily the information most useful for predicting the target.

3. PC1 has strong positive loadings for spend and purchase frequency. What could it represent?  
   **Answer:** It may represent a combined activity or value pattern, but the interpretation needs domain review.

4. Should PCA be fitted before the train-test split?  
   **Answer:** No. Fit it using training data only and transform held-out data.

5. A business team cannot interpret principal components. What could you do?  
   **Answer:** Show the original-feature loadings, compare with original features, and decide whether interpretability requirements justify using PCA.

## 17. Lesson completion checklist

- [ ] I can explain dimensionality reduction.
- [ ] I understand PCA and principal components.
- [ ] I can scale features and fit PCA.
- [ ] I can inspect explained variance and cumulative variance.
- [ ] I can choose a component count.
- [ ] I can visualize data in two components.
- [ ] I can inspect loadings.
- [ ] I understand reconstruction error.
- [ ] I can place PCA inside a supervised learning pipeline.
- [ ] I understand PCA's limitations.

**Next lesson:** Day 39 — Association Rules and Market Basket Analysis.
