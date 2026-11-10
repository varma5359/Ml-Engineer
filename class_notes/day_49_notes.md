# Day 49: Feature Selection and Feature Importance

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives
- Explain why feature selection is used.
- Distinguish feature selection from feature extraction.
- Understand filter, wrapper and embedded methods.
- Use VarianceThreshold and SelectKBest.
- Apply Recursive Feature Elimination.
- Read tree-based feature importance.
- Use permutation importance.
- Recognize leakage, redundancy and misleading importance.
- Explain feature selection decisions in interviews.

## 2. What is a feature?

A feature is an input variable used by a machine learning model to make predictions.

Examples for predicting customer churn:
- Customer tenure.
- Monthly bill.
- Number of support calls.
- Payment delay count.
- Subscription type.

A dataset may contain many features, but not all of them are useful. Some may be duplicated, irrelevant, noisy, unstable or unavailable at prediction time.

## 3. What is feature selection?

**Feature selection** means choosing a useful subset of the original input features for a model.

Why use it?
- Remove irrelevant or redundant inputs.
- Reduce training and inference cost.
- Make models easier to interpret.
- Reduce noise and sometimes improve generalization.
- Simplify data collection and maintenance.

Feature selection does not guarantee better model performance. Always compare the selected-feature model with a sensible baseline using the same validation strategy.

## 4. Feature selection versus feature extraction

| Feature selection | Feature extraction |
|---|---|
| Chooses some original features | Creates new representations from original features |
| Original column meaning is retained | New features may be harder to interpret |
| Example: select tenure and monthly bill | Example: PCA components |
| Can simplify data collection | Can reduce dimensionality in a transformed space |

PCA is a feature extraction method, not a method that simply selects original columns.

## 5. Three major feature selection families

### Filter methods
- Evaluate features using statistical or data-based criteria before fitting a particular model.
- Often fast and useful for an initial screening.
- Examples: variance threshold, correlation checks, mutual information and statistical tests.

### Wrapper methods
- Train and evaluate a model using different feature subsets.
- Can capture model-specific interactions.
- Often more computationally expensive.
- Example: Recursive Feature Elimination (RFE).

### Embedded methods
- Perform feature selection or importance estimation as part of model fitting.
- Examples: Lasso regularization and tree-based importance.
- Results depend on the model and data.

## 6. Create a feature selection dataset

### 6.1 Import libraries

~~~python
import numpy as np
import pandas as pd

print("Libraries ready")
~~~

### 6.2 Create a classification dataset

~~~python
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    n_repeated=0,
    random_state=42
)

feature_names = [f"feature_{i}" for i in range(1, 11)]

X = pd.DataFrame(
    X_values,
    columns=feature_names
)

print(X.shape)
print(X.head())
~~~

**Expected output**
A feature table with 500 rows and 10 columns.

**Explanation**
The synthetic dataset includes informative, redundant and less useful features. This helps demonstrate different selection methods.

### 6.3 Check the target distribution

~~~python
print(pd.Series(y).value_counts())
~~~

**Expected output**
Counts for the two target classes.

**Explanation**
Always inspect class balance before choosing classification evaluation and feature-selection methods.

## 7. Split data before supervised feature selection

Feature selection that uses target labels must be performed using training data only.

Why?
- The target labels contain information about the outcome.
- Selecting features using the full dataset allows test labels to influence the chosen inputs.
- This creates data leakage and optimistic evaluation.

### 7.1 Split the data

~~~python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(X_train.shape, X_test.shape)
~~~

### 7.2 Keep the test set untouched

Use X_test and y_test only for final evaluation after feature selection and model choices are complete. For robust selection, perform the selection within cross-validation or a Pipeline.

## 8. Filter method — VarianceThreshold

VarianceThreshold removes features whose values do not vary enough.

A feature with the same value for every observation usually cannot distinguish observations.

Important:
- Low variance does not always mean a feature is useless.
- A rare but important binary event may have low variance.
- Choose a threshold based on the data and business meaning.

### 8.1 Import VarianceThreshold

~~~python
from sklearn.feature_selection import VarianceThreshold

print("VarianceThreshold imported")
~~~

### 8.2 Fit the selector on training features

~~~python
variance_selector = VarianceThreshold(
    threshold=0.01
)

X_train_variance = variance_selector.fit_transform(
    X_train
)

print(X_train.shape)
print(X_train_variance.shape)
~~~

**Explanation**
The selector learns which features meet the variance requirement using training data.

### 8.3 Inspect selected feature names

~~~python
selected_by_variance = X_train.columns[
    variance_selector.get_support()
]

print(list(selected_by_variance))
~~~

**Expected output**
A list of retained feature names.

**Explanation**
get_support returns a Boolean selection mask. Use it to identify which original columns remain.

## 9. Filter method — SelectKBest

SelectKBest ranks features using a scoring function and keeps the requested number of top-scoring features.

For classification, possible scoring functions include:
- f_classif: ANOVA F-value based feature scoring.
- mutual_info_classif: estimates dependency between a feature and the target.

The selected features depend on the scoring function and dataset.

### 9.1 Import SelectKBest and f_classif

~~~python
from sklearn.feature_selection import SelectKBest, f_classif

print("SelectKBest ready")
~~~

### 9.2 Select the top five features

~~~python
kbest = SelectKBest(
    score_func=f_classif,
    k=5
)

X_train_kbest = kbest.fit_transform(
    X_train,
    y_train
)

print(X_train_kbest.shape)
~~~

**Explanation**
- score_func defines how features are scored.
- k is the number of features to keep.
- Fit the selector using training features and training labels only.

### 9.3 Display feature scores

~~~python
kbest_results = pd.DataFrame({
    "feature": X_train.columns,
    "score": kbest.scores_,
    "selected": kbest.get_support()
}).sort_values("score", ascending=False)

print(kbest_results)
~~~

**Explanation**
A larger score means stronger association under this scoring method. It does not prove that a feature causes the target.

### 9.4 Use mutual information

~~~python
from sklearn.feature_selection import mutual_info_classif

mutual_selector = SelectKBest(
    score_func=mutual_info_classif,
    k=5
)

mutual_selector.fit(
    X_train,
    y_train
)

print(
    list(X_train.columns[mutual_selector.get_support()])
)
~~~

**Explanation**
Mutual information can identify nonlinear statistical dependencies, but estimates can be affected by sample size and data characteristics.

## 10. Wrapper method — Recursive Feature Elimination

RFE repeatedly fits an estimator and removes features considered least important until the requested number remains.

Use it when:
- You want model-specific feature selection.
- The feature count is manageable.
- You can afford repeated model fitting.

Limitations:
- Can be computationally expensive.
- Depends on the estimator and its importance mechanism.
- Correlated features can make selected subsets unstable.

### 10.1 Import RFE and a model

~~~python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

print("RFE and Random Forest ready")
~~~

### 10.2 Configure RFE

~~~python
rfe = RFE(
    estimator=RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    n_features_to_select=5,
    step=1
)

print("RFE configured")
~~~

### 10.3 Fit RFE on training data

~~~python
rfe.fit(X_train, y_train)

print("RFE complete")
~~~

### 10.4 Inspect selected features and ranking

~~~python
rfe_results = pd.DataFrame({
    "feature": X_train.columns,
    "selected": rfe.support_,
    "ranking": rfe.ranking_
}).sort_values("ranking")

print(rfe_results)
~~~

**Explanation**
- selected=True indicates a retained feature.
- ranking=1 identifies selected features.
- Higher ranks indicate features removed earlier.
- The ranking is specific to this estimator and dataset.

## 11. Embedded method — tree-based feature importance

Tree models can provide feature importance based on how features contribute to splits.

### 11.1 Fit a Random Forest

~~~python
forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

forest.fit(X_train, y_train)

print("Random Forest fitted")
~~~

### 11.2 Read feature importances

~~~python
importance_table = pd.DataFrame({
    "feature": X_train.columns,
    "importance": forest.feature_importances_
}).sort_values("importance", ascending=False)

print(importance_table)
~~~

**Explanation**
The importance values summarize how the fitted forest used features under its built-in importance definition.

Cautions:
- Importance is not causation.
- Correlated features may share or distort importance.
- Impurity-based importance can favour features with many possible split points.
- Importance can change across datasets and random seeds.

### 11.3 Plot feature importances

~~~python
importance_table.sort_values("importance").plot(
    x="feature",
    y="importance",
    kind="barh",
    legend=False
)

plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()
~~~

**Expected output**
A horizontal bar chart ranking features by the fitted forest's importance values.

## 12. Permutation importance

Permutation importance measures how much a model's evaluation score changes when one feature's values are shuffled.

Intuition:
- Keep the fitted model fixed.
- Shuffle one feature in the evaluation data.
- Measure the performance change.
- A larger performance drop suggests the model relied more on that feature for the chosen score.

### 12.1 Import permutation_importance

~~~python
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

print("Permutation importance ready")
~~~

### 12.2 Fit an evaluation model

~~~python
forest.fit(X_train, y_train)

baseline_accuracy = accuracy_score(
    y_test,
    forest.predict(X_test)
)

print("Baseline accuracy:", baseline_accuracy)
~~~

### 12.3 Calculate permutation importance on held-out data

~~~python
permutation_result = permutation_importance(
    forest,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

print(permutation_result.importances_mean)
~~~

**Explanation**
The feature values are shuffled repeatedly. The output estimates how much the selected evaluation score changes.

### 12.4 Create a readable importance table

~~~python
permutation_table = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": permutation_result.importances_mean,
    "importance_std": permutation_result.importances_std
}).sort_values("importance_mean", ascending=False)

print(permutation_table)
~~~

**Explanation**
- importance_mean is the average score decrease after permutation.
- importance_std shows variation across repeats.
- Negative or near-zero importance may mean the feature adds little useful signal under this model and evaluation setup.
- Correlated features can substitute for one another, so a single feature's permutation impact may look small.

## 13. Compare feature selection methods

| Method | Family | Main idea | Limitation |
|---|---|---|---|
| VarianceThreshold | Filter | Remove near-constant features | Low variance can still be meaningful |
| SelectKBest | Filter | Rank features with a statistical score | May miss interactions |
| RFE | Wrapper | Repeatedly remove less useful features | Repeated fitting can be expensive |
| Lasso | Embedded | Regularization can shrink some coefficients to zero | Selection depends on scaling and correlated features |
| Tree importance | Embedded | Use fitted tree split contribution | Can be biased and unstable |
| Permutation importance | Model inspection | Measure score change after shuffling | Correlated features can hide individual contribution |

## 14. Feature selection inside a Pipeline

For reliable model evaluation, place supervised feature selection inside the Pipeline so it is refitted within each cross-validation training fold.

~~~python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score

selection_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selection", SelectKBest(f_classif, k=5)),
    ("model", LogisticRegression(max_iter=1000))
])

scores = cross_val_score(
    selection_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
)

print("Fold F1:", scores)
print("Average F1:", scores.mean())
~~~

**Explanation**
The scaler and selector are fitted separately inside each training fold. This avoids allowing validation-fold information to influence feature selection.

## 15. How to decide which features to keep

Consider:
- Predictive contribution under a suitable validation strategy.
- Redundancy with other features.
- Data quality and missingness.
- Availability at prediction time.
- Collection and maintenance cost.
- Stability across folds and time periods.
- Fairness, privacy and business restrictions.
- Whether the model becomes easier to explain.

Do not remove a feature solely because one importance chart ranks it low. Test the reduced feature set and compare results.

## 16. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Select features using the full dataset | Fit selection on training data inside validation |
| Assume importance means causation | Describe it as model reliance or association |
| Remove every low-variance feature | Review business meaning and rare-event value |
| Trust one feature importance method | Compare methods and stability |
| Ignore correlated features | Inspect redundancy and grouped importance |
| Keep the top five without validation | Compare model performance with a baseline |
| Use target-derived fields unavailable at prediction time | Check feature availability and leakage |
| Ignore feature maintenance cost | Include operational requirements in selection |

## 17. Interview-ready explanations

**What is feature selection?**
- Choosing a subset of original input variables.
- It can reduce noise, cost and complexity.
- It does not guarantee better performance; validate the selected subset.

**Filter versus wrapper versus embedded methods?**
- Filter methods score features independently of a specific fitted model.
- Wrapper methods repeatedly evaluate feature subsets with a model.
- Embedded methods perform selection or importance estimation during model fitting.

**What is permutation importance?**
- Shuffle one feature and measure how much the evaluation score changes.
- A larger score drop suggests the fitted model relied on that feature.
- It is not a causal explanation.

**Why can feature importance be misleading?**
- Correlated features can share information.
- Some built-in importance methods favour certain feature types.
- Importance depends on the fitted model and evaluation data.

## 18. Scenario-based interview practice

### Scenario 1: A dataset has 500 features
**Expected response:**
- Inspect data quality, feature types, redundancy and prediction-time availability.
- Establish a baseline.
- Apply filter methods for initial screening where suitable.
- Compare model-based selection and permutation importance.
- Validate selected subsets through cross-validation.

### Scenario 2: A feature has high importance but is unavailable at prediction time
**Expected response:**
- It cannot be used in the production feature set.
- Check whether it is a target leak or post-outcome field.
- Remove it and retrain using features available at the real decision time.

### Scenario 3: Two correlated features have low individual permutation importance
**Expected response:**
- The model may use one feature as a substitute for the other.
- Evaluate the features together or inspect correlation and grouped importance.
- Do not conclude that both are useless based only on individual permutation scores.

### Scenario 4: Feature selection improves training performance but hurts validation
**Expected response:**
- The selection may be overfitting or unstable.
- Recheck the validation design and leakage.
- Compare fold-level results and a baseline.
- Avoid choosing features from training performance alone.

## 19. Lesson completion checklist

- [ ] I can explain feature selection and feature extraction.
- [ ] I can distinguish filter, wrapper and embedded methods.
- [ ] I can use VarianceThreshold and SelectKBest.
- [ ] I can run RFE.
- [ ] I can inspect tree-based importance.
- [ ] I can calculate permutation importance.
- [ ] I understand correlated feature limitations.
- [ ] I can prevent leakage by selecting features inside a Pipeline.
- [ ] I can justify feature choices in an interview.

**Next lesson:** Day 50 — Feature Engineering and Transformation.
