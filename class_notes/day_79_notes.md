# Day 79: Capstone — Preprocessing and Feature Engineering

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Reusable preprocessing pipeline

## 1. Learning objectives
- Separate features and target without leakage.
- Split data before fitting preprocessing transformations.
- Handle numeric and categorical features with scikit-learn.
- Build a ColumnTransformer and Pipeline.
- Save and reuse the complete preprocessing/model workflow.

## 2. Why a reusable preprocessing pipeline?
Training and serving must apply the same transformations. If training imputes and encodes features one way but the API handles them differently, predictions can become inconsistent.

A scikit-learn pipeline bundles transformations and an estimator so the same fitted steps are applied during training and inference.

## 3. Prepare features and target
Use the actual column names from Day 78. Example:
```python
import pandas as pd

df = pd.read_csv("data/raw/customer_churn.csv")

target_col = "Churn"
id_columns = ["customerID"]

X = df.drop(columns=[target_col] + [
    col for col in id_columns if col in df.columns
])
y = df[target_col].map({"No": 0, "Yes": 1})

print(X.shape, y.value_counts(dropna=False))
```

Check for unmapped target values before continuing:
```python
if y.isna().any():
    raise ValueError("Target contains missing or unexpected labels")
```

Do not copy this label mapping blindly if your dataset uses a different encoding.

## 4. Split before fitting transformations
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X_train.shape, X_test.shape)
```

For time-based or repeated-customer data, use a split strategy that respects time and customer grouping instead of a random split.

## 5. Identify numeric and categorical columns
```python
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)
```

Review the lists. Unexpected types should be corrected based on the data dictionary.

## 6. Build preprocessing transformers
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
```

Explanation:
- Numeric imputer fills missing numeric values using a training-set statistic.
- StandardScaler standardizes numeric features.
- Categorical imputer fills missing categories.
- OneHotEncoder converts categories to indicator columns.
- `handle_unknown="ignore"` prevents a new category from crashing inference.
- ColumnTransformer applies the correct pipeline to each feature group.

Select imputation and scaling strategies based on feature meaning and model requirements.

## 7. Combine preprocessing with an estimator
```python
from sklearn.linear_model import LogisticRegression

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model_pipeline.fit(X_train, y_train)
print("Pipeline fitted")
```

All fitted preprocessing statistics are learned from X_train only. The test set is transformed using those fitted steps when evaluated.

## 8. Generate predictions
```python
predictions = model_pipeline.predict(X_test)
probabilities = model_pipeline.predict_proba(X_test)[:, 1]

print(predictions[:10])
print(probabilities[:10].round(3))
```

The positive class must correspond to churn. Confirm the label mapping and class order before interpreting probability column 1.

## 9. Inspect transformed feature names
```python
feature_names = model_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

print(feature_names[:20])
print("Transformed feature count:", len(feature_names))
```

This is useful for interpretation and debugging. Feature names may vary with scikit-learn version and category levels.

## 10. Save the complete pipeline
```python
import joblib
from pathlib import Path

Path("artifacts").mkdir(exist_ok=True)
joblib.dump(model_pipeline, "artifacts/churn_pipeline.joblib")
print("Saved complete pipeline")
```

Load only artifacts from trusted sources. Record library versions and the feature schema with the artifact.

## 11. Approved deliverable — Reusable preprocessing pipeline

**Objective:** Build a single reusable pipeline that can transform raw customer features and generate predictions consistently.

### Tasks
1. Define target and remove identifiers from candidate features.
2. Split the data using an appropriate strategy.
3. Identify numeric and categorical columns.
4. Create numeric and categorical preprocessing pipelines.
5. Combine them with ColumnTransformer.
6. Add a baseline estimator inside Pipeline.
7. Fit on training data only.
8. Generate predictions for held-out data.
9. Save the full fitted pipeline and feature schema.
10. Test inference using a new input row with a previously unseen category.

**Suggested structure**
```text
customer_churn_capstone/
├── src/
│   └── preprocessing.py
├── artifacts/
├── data/raw/
├── reports/
└── README.md
```

**Validation:** Confirm no preprocessing is fitted on test data; unknown categories do not crash; input columns match the expected schema; target labels are correct.

## 12. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for separating numeric and categorical columns." |
| Plan | "Break a ColumnTransformer pipeline into small implementation steps." |
| Example | "Show a tiny example of OneHotEncoder with an unseen category." |
| Guided implementation | "Guide me through the numeric transformer first; do not write the entire pipeline." |
| Explanation | "Explain how Pipeline prevents preprocessing mismatch." |
| Debugging | "Inspect my feature names and error; help me find the mismatch." |
| Review | "Review my pipeline for leakage, target mapping and inference compatibility." |

## 13. Daily quiz

**Q1. Why split data before fitting imputers and scalers?**  
A. To make the file smaller  
B. To prevent test information leaking into training transformations  
C. To improve UI design  
D. To remove the target  
**Correct: B.** Preprocessing statistics must be learned from training data only.

**Q2. Which tool applies different transformations to selected column groups?**  
A. ColumnTransformer  
B. LabelEncoder only  
C. train_test_split  
D. accuracy_score  
**Correct: A.** ColumnTransformer routes columns to their designated transformers.

**Q3. What does handle_unknown="ignore" do in OneHotEncoder?**  
A. Deletes all rows  
B. Allows unseen categories without raising an error  
C. Converts target to float  
D. Fits on test data  
**Correct: B.** Unknown categories are handled without failing transformation.

**Q4. What should be saved for consistent inference?**  
A. Only the scaler  
B. Only the notebook  
C. The complete fitted preprocessing and model pipeline  
D. Only the target column  
**Correct: C.** This preserves the fitted transformations and estimator together.

**Q5. Which split may be more appropriate for future churn prediction from monthly snapshots?**  
A. Random split in every situation  
B. Time-aware split when the deployment task predicts future periods  
C. No split  
D. Split only the target  
**Correct: B.** Time-aware evaluation better reflects future deployment.

## 14. Interview preparation

### Q1. What is the benefit of a scikit-learn Pipeline?
**Answer:** It chains preprocessing and estimation into one object. It ensures the same fitted transformations are used during training and inference, supports cross-validation without preprocessing leakage, and simplifies artifact saving.

**Follow-up:** How does Pipeline work with ColumnTransformer?

### Q2. How do you prevent preprocessing leakage?
**Answer:** Split the data first, fit all preprocessing steps only on the training fold, and use a pipeline during cross-validation. Never calculate imputation values, scaling parameters or encodings using the held-out test set.

**Follow-up:** How would you handle leakage in time-series or repeated-customer data?

## 15. Completion checklist
- [ ] Separate features, target and identifiers.
- [ ] Split using an appropriate strategy.
- [ ] Build numeric and categorical transformers.
- [ ] Combine them into a fitted pipeline.
- [ ] Test unseen categories and schema compatibility.
- [ ] Save the complete pipeline.
- [ ] Complete Reusable preprocessing pipeline.

**Next lesson:** Day 80 — Capstone: Baseline model and initial experiments.
