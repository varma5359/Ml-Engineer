# Day 27: K-Nearest Neighbors and Distance Metrics

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain K-Nearest Neighbors (KNN) in simple language.
- Understand how KNN uses nearby observations to predict.
- Distinguish classification and regression with KNN.
- Explain the role of the number of neighbors.
- Understand why feature scaling is important.
- Compare common distance measures conceptually.
- Train KNeighborsClassifier using scikit-learn.
- Tune the number of neighbors using validation or cross-validation.
- Recognize the impact of noise, class imbalance and high-dimensional data.
- Explain KNN limitations in real-world ML systems.

## 2. What is K-Nearest Neighbors?

**K-Nearest Neighbors (KNN)** is a supervised learning algorithm that predicts an observation using the labels or values of nearby training observations.

For classification:
- Find the K nearest training observations.
- Inspect their class labels.
- Predict the class that receives the most votes.

For regression:
- Find the K nearest training observations.
- Combine their target values, commonly using an average.
- Return the combined numerical estimate.

KNN is often called a lazy learning method because it stores training observations and performs much of its work when a new prediction is requested.

## 3. Understand K

The parameter **K** controls how many nearby observations are considered.

- Small K: predictions depend on fewer neighbours and may be sensitive to noise.
- Large K: predictions use more neighbours and may become smoother, but can miss local patterns.
- The suitable K depends on the data and must be selected using validation.

There is no universally best K.

## 4. Why distance matters

KNN needs a way to measure how close two observations are.

Common distance choices:
- Euclidean distance: straight-line distance in numerical feature space.
- Manhattan distance: sum of absolute differences across dimensions.
- Minkowski distance: a general family that includes common distance choices.
- Hamming distance: counts differing positions and is used for some categorical or binary representations.

Distance only makes sense when feature representation and scaling are appropriate.

## 5. Create a classification dataset

### 5.1 Import libraries

### 5.1. Import NumPy, Pandas and plotting libraries

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

These libraries prepare and visualize the classification data.



### 5.2 Generate two-class observations

### 5.2. Create a synthetic dataset

```python
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.3,
    random_state=42
)

print(X_values.shape)
print(pd.Series(y).value_counts())
```

**Expected output**

```text
(300, 2) and class counts
```

**Explanation**

Two numerical features make it possible to visualize the neighbourhoods and decision regions.



### 5.3 Create a labelled DataFrame

### 5.3. Add feature names and target

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

The target is stored separately when we create X and y for modelling.



### 5.4 Visualize the class distribution

### 5.4. Plot observations by class

```python
sns.scatterplot(
    data=df,
    x="feature_1",
    y="feature_2",
    hue="target"
)
plt.title("KNN training data")
plt.show()
```

**Expected output**

```text
A two-dimensional scatter plot with class labels
```

**Explanation**

The chart shows whether classes form visible neighbourhoods or overlap.



## 6. Prepare training and test data

### 6.1 Create X and y

### 6.1. Select feature and target columns

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

X contains two features and y contains one class label for each observation.



### 6.2 Split the dataset

### 6.2. Create stratified train-test split

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
(240, 2) and (60, 2)
```

**Explanation**

The test observations are held out. Stratification keeps class proportions approximately similar.



## 7. Why scale features for KNN?

KNN compares feature distances. If one feature has a much larger numerical range than another, it can dominate the distance.

Example:
- Age ranges from 18 to 80.
- Annual income ranges from 10,000 to 2,000,000.

Without scaling, income differences can dominate the distance even if age is also important.

### 7.1 Inspect feature ranges

### 7.1. Check feature summary statistics

```python
print(X_train.describe())
```

**Expected output**

```text
Summary statistics for the two features
```

**Explanation**

Inspect ranges before applying distance-based algorithms.



### 7.2 Create a StandardScaler

### 7.2. Initialize the scaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print(scaler)
```

**Expected output**

```text
StandardScaler()
```

**Explanation**

The scaler learns training feature statistics during fit.



### 7.3 Fit and transform training data

### 7.3. Scale training features

```python
X_train_scaled = scaler.fit_transform(X_train)

print(X_train_scaled[:5])
```

**Expected output**

```text
The first five standardized training rows
```

**Explanation**

The scaler is fitted using training data only.



### 7.4 Transform test data

### 7.4. Scale test features using training statistics

```python
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled[:5])
```

**Expected output**

```text
The first five transformed test rows
```

**Explanation**

Use transform(), not fit_transform(), on the test set.



## 8. Train a KNN classifier

### 8.1 Import KNeighborsClassifier

### 8.1. Import the classifier

```python
from sklearn.neighbors import KNeighborsClassifier

print("KNN classifier imported")
```

**Expected output**

```text
KNN classifier imported
```

**Explanation**

The estimator performs classification using nearby training observations.



### 8.2 Create a KNN model

### 8.2. Initialize KNN with five neighbors

```python
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2
)

print(knn)
```

**Expected output**

```text
KNeighborsClassifier(n_neighbors=5) with Minkowski distance settings
```

**Explanation**

n_neighbors specifies K. With p=2, Minkowski distance corresponds to Euclidean distance.



### 8.3 Fit the classifier

### 8.3. Fit KNN on scaled training data

```python
knn.fit(X_train_scaled, y_train)

print("KNN fitted")
```

**Expected output**

```text
KNN fitted
```

**Explanation**

KNN stores the training observations and labels. Unlike many models, fitting does not learn a compact set of coefficients.



### 8.4 Predict test classes

### 8.4. Generate KNN predictions

```python
knn_predictions = knn.predict(X_test_scaled)

print(knn_predictions[:15])
```

**Expected output**

```text
The first fifteen predicted class labels
```

**Explanation**

For each test observation, KNN identifies nearby training observations and uses their labels to vote.



### 8.5 Predict class probabilities

### 8.5. Generate KNN probabilities

```python
knn_probabilities = knn.predict_proba(X_test_scaled)

print(knn_probabilities[:5])
```

**Expected output**

```text
A probability array for the first five test observations
```

**Explanation**

The estimated class probabilities are based on neighbour voting or weights, depending on model settings.



## 9. Evaluate KNN

### 9.1 Calculate accuracy

### 9.1. Evaluate accuracy

```python
from sklearn.metrics import accuracy_score

knn_accuracy = accuracy_score(y_test, knn_predictions)

print(knn_accuracy)
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

Accuracy reports the fraction of correct predictions. Inspect class-level metrics when classes are imbalanced.



### 9.2 Print classification report

### 9.2. Review class metrics

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    knn_predictions,
    zero_division=0
))
```

**Expected output**

```text
Precision, recall, F1 and support for each class
```

**Explanation**

This shows which class the model handles well or poorly.



### 9.3 Plot the confusion matrix

### 9.3. Create confusion matrix chart

```python
from sklearn.metrics import confusion_matrix

knn_cm = confusion_matrix(y_test, knn_predictions)

sns.heatmap(knn_cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN confusion matrix")
plt.show()
```

**Expected output**

```text
A confusion-matrix heatmap
```

**Explanation**

The chart shows correct and incorrect predictions by class.



## 10. Compare different K values

### 10.1 Create a validation split

### 10.1. Split training data into fit and validation subsets

```python
X_fit, X_valid, y_fit, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

print(X_fit.shape, X_valid.shape)
```

**Expected output**

```text
Training-fit and validation shapes
```

**Explanation**

The final test set remains untouched while K is selected.



### 10.2 Fit scaler on the fit subset

### 10.2. Fit scaling on fit subset

```python
validation_scaler = StandardScaler()

X_fit_scaled = validation_scaler.fit_transform(X_fit)
X_valid_scaled = validation_scaler.transform(X_valid)

print(X_fit_scaled.shape, X_valid_scaled.shape)
```

**Expected output**

```text
Matching fit and validation feature shapes
```

**Explanation**

The scaler learns only from X_fit and transforms validation data using those learned statistics.



### 10.3 Compare odd K values

### 10.3. Evaluate several neighbour counts

```python
from sklearn.metrics import accuracy_score

k_results = []

for k in [1, 3, 5, 7, 9, 11, 15]:
    candidate = KNeighborsClassifier(n_neighbors=k)
    candidate.fit(X_fit_scaled, y_fit)

    valid_pred = candidate.predict(X_valid_scaled)

    k_results.append({
        "k": k,
        "validation_accuracy": accuracy_score(y_valid, valid_pred)
    })

k_table = pd.DataFrame(k_results)

print(k_table)
```

**Expected output**

```text
A table comparing validation accuracy for each K
```

**Explanation**

The comparison uses the same validation observations. An odd K can reduce ties in binary voting, but it is not a universal requirement.



### 10.4 Plot K against validation accuracy

### 10.4. Visualize K selection

```python
plt.plot(k_table["k"], k_table["validation_accuracy"], marker="o")
plt.xlabel("Number of neighbors (K)")
plt.ylabel("Validation accuracy")
plt.title("KNN K-value comparison")
plt.show()
```

**Expected output**

```text
A line chart showing validation accuracy across K values
```

**Explanation**

The chart helps inspect how model flexibility changes as more neighbours are included.



### 10.5 Select a K using validation data

### 10.5. Find the best validation K

```python
best_k = int(
    k_table.loc[
        k_table["validation_accuracy"].idxmax(),
        "k"
    ]
)

print(best_k)
```

**Expected output**

```text
The K value with the highest validation accuracy
```

**Explanation**

This is a simple selection rule. In practice, also consider metric choice, score variation and business costs.



## 11. Compare distance metrics

### 11.1 Train with Euclidean distance

### 11.1. Fit KNN with Euclidean distance

```python
euclidean_knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2
)

euclidean_knn.fit(X_fit_scaled, y_fit)

print("Euclidean KNN fitted")
```

**Expected output**

```text
Euclidean KNN fitted
```

**Explanation**

Minkowski with p=2 is equivalent to Euclidean distance.



### 11.2 Train with Manhattan distance

### 11.2. Fit KNN with Manhattan distance

```python
manhattan_knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="manhattan"
)

manhattan_knn.fit(X_fit_scaled, y_fit)

print("Manhattan KNN fitted")
```

**Expected output**

```text
Manhattan KNN fitted
```

**Explanation**

Manhattan distance sums absolute feature differences. It can behave differently from Euclidean distance.



### 11.3 Compare validation predictions

### 11.3. Compare distance choices

```python
euclidean_pred = euclidean_knn.predict(X_valid_scaled)
manhattan_pred = manhattan_knn.predict(X_valid_scaled)

print("Euclidean:", accuracy_score(y_valid, euclidean_pred))
print("Manhattan:", accuracy_score(y_valid, manhattan_pred))
```

**Expected output**

```text
Two validation accuracy values
```

**Explanation**

Use the same fit and validation data for a fair comparison.



## 12. Distance weighting

KNN can give each neighbour equal voting power or assign greater influence to closer neighbours.

### 12.1 Train uniform-weight KNN

### 12.1. Use uniform neighbour weights

```python
uniform_knn = KNeighborsClassifier(
    n_neighbors=7,
    weights="uniform"
)

uniform_knn.fit(X_fit_scaled, y_fit)

print("Uniform KNN fitted")
```

**Expected output**

```text
Uniform KNN fitted
```

**Explanation**

Each selected neighbour contributes equally to the vote.



### 12.2 Train distance-weighted KNN

### 12.2. Use distance-based weights

```python
distance_knn = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance"
)

distance_knn.fit(X_fit_scaled, y_fit)

print("Distance-weighted KNN fitted")
```

**Expected output**

```text
Distance-weighted KNN fitted
```

**Explanation**

Closer neighbours receive more influence than farther neighbours.



### 12.3 Compare both weighting approaches

### 12.3. Evaluate weighting choices

```python
uniform_pred = uniform_knn.predict(X_valid_scaled)
distance_pred = distance_knn.predict(X_valid_scaled)

print("Uniform:", accuracy_score(y_valid, uniform_pred))
print("Distance:", accuracy_score(y_valid, distance_pred))
```

**Expected output**

```text
Two validation accuracy values
```

**Explanation**

The best weighting strategy depends on the distribution and noise in the data.



## 13. Visualize a KNN decision boundary

### 13.1 Create a grid across feature space

### 13.1. Generate plotting coordinates

```python
x1 = np.linspace(X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5, 150)
x2 = np.linspace(X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5, 150)

xx, yy = np.meshgrid(x1, x2)

print(xx.shape)
```

**Expected output**

```text
A two-dimensional grid shape
```

**Explanation**

The grid contains points at which the classifier's prediction will be calculated.



### 13.2 Prepare grid points

### 13.2. Combine grid coordinates

```python
grid = np.c_[xx.ravel(), yy.ravel()]

print(grid.shape)
```

**Expected output**

```text
A two-column array of grid points
```

**Explanation**

Each row represents one location in the scaled feature space.



### 13.3 Predict the class across the grid

### 13.3. Predict decision regions

```python
grid_predictions = knn.predict(grid)

region = grid_predictions.reshape(xx.shape)

print(region.shape)
```

**Expected output**

```text
A two-dimensional prediction region
```

**Explanation**

The predicted classes are reshaped to match the plotting grid.



### 13.4 Plot KNN decision regions

### 13.4. Visualize the decision boundary

```python
plt.contourf(xx, yy, region, alpha=0.3)
plt.scatter(
    X_train_scaled[:, 0],
    X_train_scaled[:, 1],
    c=y_train,
    edgecolor="black"
)
plt.xlabel("Scaled feature 1")
plt.ylabel("Scaled feature 2")
plt.title("KNN decision regions")
plt.show()
```

**Expected output**

```text
A shaded decision-region plot with training observations
```

**Explanation**

The boundary can be irregular because KNN uses local neighbourhoods. The chart is based on the scaled features.



## 14. KNN for regression

KNN can also predict numerical targets. It combines the target values of nearby training observations, commonly using an average.

### 14.1 Create a simple regression dataset

### 14.1. Prepare house area and price

```python
reg_df = pd.DataFrame({
    "area": [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
    "price": [20, 22, 25, 28, 31, 33, 36, 39, 42, 45]
})

print(reg_df.head())
```

**Expected output**

```text
A house area and price table
```

**Explanation**

This small example illustrates KNN regression.



### 14.2 Separate feature and target

### 14.2. Create regression X and y

```python
X_reg = reg_df[["area"]]
y_reg = reg_df["price"]

print(X_reg.shape, y_reg.shape)
```

**Expected output**

```text
(10, 1) and (10,)
```

**Explanation**

Area is the feature and price is the numerical target.



### 14.3 Scale the feature

### 14.3. Scale area values

```python
reg_scaler = StandardScaler()
X_reg_scaled = reg_scaler.fit_transform(X_reg)

print(X_reg_scaled[:5])
```

**Expected output**

```text
The first five standardized area values
```

**Explanation**

This demonstration scales one complete dataset for learning the transformation. For model evaluation, split first and fit the scaler on training data only.



### 14.4 Train KNN regression

### 14.4. Fit KNeighborsRegressor

```python
from sklearn.neighbors import KNeighborsRegressor

knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_reg_scaled, y_reg)

print("KNN regressor fitted")
```

**Expected output**

```text
KNN regressor fitted
```

**Explanation**

The model stores nearby training observations and combines their numerical target values during prediction.



### 14.5 Predict a new house price

### 14.5. Transform and predict a new area

```python
new_area = pd.DataFrame({"area": [950]})
new_area_scaled = reg_scaler.transform(new_area)

print(knn_regressor.predict(new_area_scaled))
```

**Expected output**

```text
A numerical predicted price
```

**Explanation**

The new input is transformed using the fitted scaler before prediction.



## 15. KNN limitations

- Prediction can be slow when the training dataset is very large.
- It stores training observations, which can require substantial memory.
- It is sensitive to feature scaling.
- Irrelevant features can distort distance.
- High-dimensional data can make distances less informative.
- Noise and class imbalance can affect neighbour voting.
- A new observation outside the training range may have unreliable neighbours.

## 16. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Use KNN without scaling | Scale numeric features in a pipeline |
| Choose K using test data | Select K using validation or cross-validation |
| Assume larger K is always better | Compare suitable K values |
| Ignore irrelevant features | Inspect feature quality and validation |
| Use only accuracy for imbalanced data | Inspect precision, recall and class counts |
| Forget the feature order at prediction | Validate input schema |
| Assume KNN training is computationally expensive | Understand that prediction can be the costly part |
| Ignore high dimensionality | Consider feature selection or other models |

## 17. Interview-ready explanations

**What is KNN?**  
It is a supervised learning algorithm that predicts a new observation using the labels or target values of nearby training observations.

**Why is scaling important for KNN?**  
KNN uses distances. Features with larger numerical ranges can dominate distance calculations if they are not scaled appropriately.

**What does K control?**  
It determines how many neighbours contribute to the prediction. Small K is sensitive to local noise; large K can smooth over local patterns.

**What is the difference between uniform and distance weighting?**  
Uniform weighting gives each selected neighbour equal influence. Distance weighting gives closer neighbours more influence.

**What is one limitation of KNN?**  
Prediction can be slow and memory-intensive for large datasets because it needs access to training observations.

## 18. Scenario questions

1. KNN gives poor results when income is much larger in scale than age. What do you do?  
   **Answer:** Scale numerical features within a fitted training pipeline.

2. K=1 performs very well on training data but poorly on validation. What may be happening?  
   **Answer:** The model may be too sensitive to individual training observations and overfit noise.

3. The dataset has hundreds of features. What is the concern?  
   **Answer:** High-dimensional distance can become less informative. Consider feature selection, dimensionality reduction or another algorithm.

4. KNN prediction is slow after deployment. What might explain it?  
   **Answer:** KNN may need to compare new observations with many stored training rows. Consider dataset size, indexing strategy or a different model.

5. A new observation has values far outside the training range. What should you do?  
   **Answer:** Validate the input and recognize that nearest-neighbour predictions may be unreliable outside the training distribution.

## 19. Lesson completion checklist

- [ ] I can explain KNN classification and regression.
- [ ] I understand K and neighbour voting.
- [ ] I can explain common distance metrics.
- [ ] I understand why scaling is important.
- [ ] I can train KNeighborsClassifier.
- [ ] I can compare K values and distance choices.
- [ ] I can visualize a decision boundary.
- [ ] I can explain KNN limitations.
- [ ] I can describe how KNN regression works.

**Next lesson:** Day 28 — Naive Bayes and Conditional Probability.
