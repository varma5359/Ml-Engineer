# Day 12: Encoding Categorical Variables and Scaling Numerical Data

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Distinguish numerical and categorical features.
- Explain why many ML estimators require numerical inputs.
- Apply one-hot encoding to nominal categories.
- Explain when ordinal encoding is appropriate.
- Recognize why arbitrary numeric codes can mislead a model.
- Explain feature scaling in plain language.
- Apply standardization and min-max scaling.
- Understand why scaling matters for distance-based and gradient-based algorithms.
- Fit preprocessing only on training data.
- Use scikit-learn transformers and a ColumnTransformer.
- Validate transformed feature names and output shapes.

## 2. Why do we need encoding and scaling?

A dataset can contain both numerical and categorical information.

Example:

| age | monthly_spend | city | contract_type |
|---:|---:|---|---|
| 25 | 4500 | Hyderabad | Monthly |
| 34 | 5200 | Chennai | Annual |
| 29 | 6100 | Vizag | Monthly |

Age and monthly spend are numerical. City and contract type are categories.

Many estimators work with numerical feature representations. Encoding converts categories into suitable numerical columns. Scaling adjusts the magnitude of numerical features so that a large unit range does not dominate certain algorithms merely because of its scale.

**Important:** Encoding and scaling solve different problems. Encoding represents categories; scaling changes numerical magnitude.

## 3. Create a small dataset

### 3.1. Import Pandas and NumPy

```python
import pandas as pd
import numpy as np

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

Pandas creates the table and NumPy supplies numerical values.



### 3.2. Create a mixed-type feature table

```python
df = pd.DataFrame({
    "age": [25, 34, 29, 42, 31],
    "monthly_spend": [4500, 5200, 6100, 7200, 4900],
    "city": ["Hyderabad", "Chennai", "Vizag", "Hyderabad", "Chennai"],
    "contract_type": ["Monthly", "Annual", "Monthly", "Annual", "Monthly"],
    "support_level": ["Low", "Medium", "High", "Medium", "Low"]
})

print(df)
```

**Expected output**

```text
A five-row DataFrame with two numerical and three categorical columns
```

**Explanation**

This dataset is intentionally small for demonstration. A real project should inspect the full category list and data quality first.



### 3.3. Inspect column types

```python
print(df.dtypes)
```

**Expected output**

```text
Numeric dtypes for age and monthly_spend; object/string-like dtypes for the category columns
```

**Explanation**

The technical dtype helps identify candidate columns, but you must also understand each column's meaning.



## 4. Nominal and ordinal categories

### Nominal category

A nominal category has labels with no natural ranking.

Examples:
- City: Hyderabad, Chennai, Vizag.
- Payment type: cash, card, UPI.
- Product category: electronics, clothing, home.

### Ordinal category

An ordinal category has a meaningful order.

Examples:
- Low, Medium, High.
- Bronze, Silver, Gold.
- Beginner, Intermediate, Advanced.

Do not invent an order for nominal labels. For example, assigning Hyderabad=1, Chennai=2 and Vizag=3 does not mean Vizag is greater than Chennai.

## 5. One-hot encoding

**One-hot encoding** creates a separate indicator column for each category. A row receives an active value in the column representing its category.

### 5.1 Inspect city categories

### 5.1. Find distinct city values

```python
print(df["city"].unique())
```

**Expected output**

```text
['Hyderabad' 'Chennai' 'Vizag']
```

**Explanation**

Check the category list before encoding. Inconsistent spelling can create unnecessary extra columns.



### 5.2 Count city categories

### 5.2. Count each city

```python
print(df["city"].value_counts())
```

**Expected output**

```text
Counts for Hyderabad, Chennai and Vizag
```

**Explanation**

Counts help identify rare categories and unexpected labels.



### 5.3 Apply Pandas get_dummies()

### 5.3. One-hot encode the city column

```python
city_encoded = pd.get_dummies(
    df,
    columns=["city"],
    dtype=int
)

print(city_encoded)
```

**Expected output**

```text
The original city column is replaced by city_Hyderabad, city_Chennai and city_Vizag indicator columns
```

**Explanation**

The `columns` argument identifies the columns to encode. `dtype=int` requests integer indicators rather than Boolean values.



### 5.4 Encode two nominal columns

### 5.4. Encode city and contract type

```python
encoded_df = pd.get_dummies(
    df,
    columns=["city", "contract_type"],
    dtype=int
)

print(encoded_df.head())
```

**Expected output**

```text
A DataFrame containing numerical fields, support_level and indicator columns for city and contract_type
```

**Explanation**

Each nominal category receives an indicator column. support_level is still categorical and has not yet been encoded.



### 5.5 Understand one row's encoded values

### 5.5. Inspect the first encoded record

```python
print(encoded_df.iloc[0])
```

**Expected output**

```text
The encoded feature values for the first customer
```

**Explanation**

The active city and contract columns identify the first customer's category. The original text labels have been replaced by indicator values.



### 5.6 Avoid encoding the target as an ordinary feature

When a supervised target is a category, it is handled separately as the label. Do not include it in the feature encoding step. Keep feature preprocessing and target preparation distinct.

## 6. Ordinal encoding

Ordinal encoding maps ordered categories to integer values. Use it only when the category order is meaningful and the mapping is agreed.

### 6.1 Inspect support levels

### 6.1. Check support-level categories

```python
print(df["support_level"].unique())
```

**Expected output**

```text
['Low' 'Medium' 'High']
```

**Explanation**

Confirm that Low, Medium and High are genuinely ordered levels.



### 6.2 Define an explicit mapping

### 6.2. Create an approved order mapping

```python
support_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

support_encoded = df["support_level"].map(support_mapping)

print(support_encoded)
```

**Expected output**

```text
0, 1, 2, 1, 0
```

**Explanation**

The mapping encodes the business-defined order. The integer values are labels for ordered levels; their numeric spacing should not automatically be interpreted as equal real-world distances.



### 6.3 Add the encoded column to a copy

### 6.3. Create an encoded support column

```python
ordinal_df = df.copy()

ordinal_df["support_level_encoded"] = (
    ordinal_df["support_level"].map(support_mapping)
)

print(ordinal_df[["support_level", "support_level_encoded"]])
```

**Expected output**

```text
The original category and its encoded representation
```

**Explanation**

Keeping both columns during inspection helps confirm that the mapping was applied correctly. Remove or retain the original according to the final preprocessing design.



### 6.4 Check for unmapped categories

### 6.4. Detect missing mapping results

```python
print(ordinal_df["support_level_encoded"].isna().sum())
```

**Expected output**

```text
0
```

**Explanation**

An unmapped category becomes missing after map(). A non-zero count means the category mapping is incomplete or the source contains unexpected labels.



## 7. Why arbitrary category numbers can be misleading

Imagine city labels are assigned:
- Hyderabad = 1
- Chennai = 2
- Vizag = 3

A distance-based algorithm may interpret the numbers as ordered and equally spaced. But city labels have no natural numeric distance in this representation.

**Rule:** For unordered categories, use an encoding method that does not invent a false ranking. One-hot encoding is a common starting point for low-cardinality nominal features.

## 8. Unknown categories at inference time

A model may receive a category during prediction that was not present during training. The preprocessing design must define how to handle it.

### 8.1 Create a train and test category example

### 8.1. Create training and inference categories

```python
train_cities = pd.DataFrame({
    "city": ["Hyderabad", "Chennai", "Vizag"]
})

new_customers = pd.DataFrame({
    "city": ["Hyderabad", "Warangal"]
})

print(train_cities)
print(new_customers)
```

**Expected output**

```text
Two small DataFrames, with Warangal appearing only in the new data
```

**Explanation**

This simulates a category that the training data has not seen.



### 8.2 Use OneHotEncoder with unknown handling

### 8.2. Fit an encoder on training categories

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

train_encoded = encoder.fit_transform(train_cities[["city"]])

print(train_encoded)
```

**Expected output**

```text
A three-row indicator matrix
```

**Explanation**

The encoder learns categories from training data. `handle_unknown='ignore'` allows unseen categories during transform without raising an error. Some scikit-learn versions use the older parameter name `sparse`; check the installed version if needed.



### 8.3 Transform new categories

### 8.3. Apply the fitted encoder to new data

```python
new_encoded = encoder.transform(new_customers[["city"]])

print(new_encoded)
```

**Expected output**

```text
An indicator row for Hyderabad and an all-zero encoded row for unseen Warangal
```

**Explanation**

The encoder does not learn new categories during transform. Unknown handling prevents an exception, but the all-zero representation should be understood and monitored.



### 8.4 Inspect learned feature names

### 8.4. Read encoded feature names

```python
print(encoder.get_feature_names_out(["city"]))
```

**Expected output**

```text
Names such as city_Chennai, city_Hyderabad and city_Vizag
```

**Explanation**

Feature names make transformed columns easier to interpret and audit.



## 9. What is feature scaling?

**Feature scaling** changes the numerical range or distribution of a feature. It is useful when the algorithm is sensitive to feature magnitudes.

Example:
- Age may range from 18 to 80.
- Monthly spend may range from 500 to 100,000.

A distance-based algorithm can be dominated by the larger numerical range if features are not appropriately scaled.

Scaling is commonly important for:
- K-nearest neighbors.
- K-means clustering.
- Support vector machines.
- Many gradient-based optimization workflows.
- Regularized linear models.

Tree-based algorithms are generally less sensitive to monotonic feature scaling, though preprocessing decisions should still be validated.

## 10. Standardization

Standardization transforms a numerical feature so that it is centered around zero and measured in standard-deviation units, using statistics learned from the training data.

It does not guarantee every transformed value lies between zero and one.

### 10.1 Create a numerical feature table

### 10.1. Create age and spend values

```python
numeric_df = pd.DataFrame({
    "age": [20, 25, 30, 35, 40],
    "monthly_spend": [1000, 2000, 3000, 4000, 5000]
})

print(numeric_df)
```

**Expected output**

```text
A five-row numerical DataFrame
```

**Explanation**

The two features have very different numerical ranges.



### 10.2 Inspect feature means

### 10.2. Calculate means

```python
print(numeric_df.mean())
```

**Expected output**

```text
The mean age and mean monthly spend
```

**Explanation**

This gives a baseline summary before scaling.



### 10.3 Import StandardScaler

### 10.3. Import the scaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print(type(scaler))
```

**Expected output**

```text
The StandardScaler class instance
```

**Explanation**

The scaler learns transformation statistics when fit is called.



### 10.4 Fit and transform the training data

### 10.4. Standardize the numerical features

```python
scaled_values = scaler.fit_transform(numeric_df)

print(scaled_values)
```

**Expected output**

```text
A numerical array with values centered around zero for each column
```

**Explanation**

fit_transform learns the mean and standard deviation from the supplied training data and transforms those values. In an ML evaluation workflow, supply training data only.



### 10.5 Convert the scaled result back to a DataFrame

### 10.5. Preserve column names

```python
scaled_df = pd.DataFrame(
    scaled_values,
    columns=numeric_df.columns,
    index=numeric_df.index
)

print(scaled_df)
```

**Expected output**

```text
A DataFrame with age and monthly_spend columns in standardized form
```

**Explanation**

Scikit-learn transformers commonly return NumPy arrays. Reattach names and index labels carefully if you need a labelled table for inspection.



### 10.6 Check scaled means

### 10.6. Inspect transformed means

```python
print(scaled_df.mean().round(6))
```

**Expected output**

```text
Values close to zero for both columns
```

**Explanation**

The transformed training columns are centered around zero, subject to floating-point precision.



### 10.7 Check scaled standard deviations

### 10.7. Inspect transformed spread

```python
print(scaled_df.std(ddof=0).round(6))
```

**Expected output**

```text
Values close to one for both columns
```

**Explanation**

The scaler uses its fitted population-style scaling convention. The check uses ddof=0 to match the transformation convention.



## 11. Min-max scaling

Min-max scaling maps a feature into a selected range, commonly zero to one, based on the minimum and maximum values learned from the training data.

### 11.1 Import MinMaxScaler

### 11.1. Create a MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

print(type(minmax))
```

**Expected output**

```text
The MinMaxScaler class instance
```

**Explanation**

The scaler learns training feature minima and maxima when fit is called.



### 11.2 Fit and transform numerical data

### 11.2. Apply min-max scaling

```python
minmax_values = minmax.fit_transform(numeric_df)

print(minmax_values)
```

**Expected output**

```text
Each training column mapped between zero and one
```

**Explanation**

The minimum training value maps to zero and maximum to one for each feature, assuming non-constant values.



### 11.3 Inspect scaled minimum and maximum

### 11.3. Check the transformed range

```python
print(minmax_values.min(axis=0))
print(minmax_values.max(axis=0))
```

**Expected output**

```text
[0. 0.] followed by [1. 1.]
```

**Explanation**

The training data reaches the endpoints for each feature. New observations may fall outside this range unless clipping is explicitly configured.



### 11.4 Compare standardization and min-max scaling

| Method | Typical result | Useful note |
|---|---|---|
| StandardScaler | Centered around zero with unit-like spread | Does not constrain values to 0–1 |
| MinMaxScaler | Maps training minimum and maximum to selected range | Sensitive to extreme min/max values |
| RobustScaler | Uses median and quantile range | Can be useful when outliers affect ordinary scaling |

## 12. Why scaling must be fitted on training data only

A scaler learns statistics from the data. If it is fitted on all rows before splitting, information from validation and test sets influences preprocessing.

Correct workflow:
1. Split the data.
2. Fit the scaler using training features only.
3. Transform training features.
4. Transform validation/test features using the already fitted scaler.
5. Never fit a new scaler on the test set for the same evaluation.

### 12.1 Split a numerical dataset

### 12.1. Create a training and test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(
    numeric_df,
    test_size=0.4,
    random_state=42
)

print(X_train.shape, X_test.shape)
```

**Expected output**

```text
Training and test shapes
```

**Explanation**

This is a simple unsupervised split demonstration. For supervised ML, split X and y together using a strategy appropriate to the data.



### 12.2 Fit the scaler on training values

### 12.2. Fit only on training data

```python
train_scaler = StandardScaler()

X_train_scaled = train_scaler.fit_transform(X_train)

print(X_train_scaled)
```

**Expected output**

```text
The standardized training feature values
```

**Explanation**

The scaler's learned statistics come only from X_train.



### 12.3 Transform test data without refitting

### 12.3. Transform held-out data

```python
X_test_scaled = train_scaler.transform(X_test)

print(X_test_scaled)
```

**Expected output**

```text
The test values transformed using training statistics
```

**Explanation**

Use transform(), not fit_transform(), on the test data. Test values can fall outside the training distribution.



### 12.4 Compare incorrect and correct practice

**Incorrect for evaluation:**
```python
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X)
# Split after fitting the scaler
```

**Correct:**
```python
X_train, X_test = train_test_split(X, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The second workflow prevents the scaler from learning from the held-out test rows.

## 13. Use ColumnTransformer for mixed data

Real datasets commonly have both numeric and categorical columns. A **ColumnTransformer** applies different transformations to different column groups.

### 13.1 Define numerical and categorical columns

### 13.1. Separate feature groups

```python
numeric_features = ["age", "monthly_spend"]
categorical_features = ["city", "contract_type", "support_level"]

print(numeric_features)
print(categorical_features)
```

**Expected output**

```text
Two lists of column names
```

**Explanation**

This makes the preprocessing plan explicit. The lists should match the actual schema.



### 13.2 Create preprocessing transformers

### 13.2. Build a ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

print(type(preprocessor))
```

**Expected output**

```text
A ColumnTransformer object
```

**Explanation**

The num branch scales numerical features. The cat branch one-hot encodes categorical features and ignores unseen categories during transform.



### 13.3 Fit and transform a dataset for demonstration

### 13.3. Apply mixed-column preprocessing

```python
X = df[numeric_features + categorical_features]

X_prepared = preprocessor.fit_transform(X)

print(X_prepared.shape)
```

**Expected output**

```text
A shape showing the rows and expanded encoded feature columns
```

**Explanation**

This demonstrates the transformation on one dataset. For evaluation, fit the preprocessor only on training rows and transform held-out rows with the fitted object.



### 13.4 Inspect transformed feature names

### 13.4. Retrieve output feature names

```python
print(preprocessor.get_feature_names_out())
```

**Expected output**

```text
Names for scaled numerical and one-hot categorical output features
```

**Explanation**

Feature names help document the transformed input schema. The prefix identifies which transformer produced each feature.



## 14. Combine preprocessing with a model pipeline

A **Pipeline** chains transformations and an estimator. It reduces the risk that training and inference use different preprocessing steps.

### 14.1 Define a target and feature table

### 14.1. Create an illustrative target

```python
X = df[numeric_features + categorical_features]
y = pd.Series([0, 1, 0, 1, 0], name="churn")

print(X.shape, y.shape)
```

**Expected output**

```text
(5, 5) and (5,)
```

**Explanation**

The example target is illustrative. In a real project, labels must come from verified outcomes and be aligned with each feature row.



### 14.2 Split the data

### 14.2. Split features and target together

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)
```

**Expected output**

```text
Training and test feature shapes
```

**Explanation**

Stratification is useful for classification when class representation should be maintained and there are enough examples in each class. Very small datasets may not support the requested split.



### 14.3 Build a preprocessing and model pipeline

### 14.3. Create the complete pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

print(model_pipeline)
```

**Expected output**

```text
A Pipeline containing the preprocessor and logistic regression estimator
```

**Explanation**

During fit, the preprocessing component learns from training data and transforms it before the estimator is trained.



### 14.4 Fit the pipeline

### 14.4. Train the pipeline

```python
model_pipeline.fit(X_train, y_train)

print("Pipeline fitted")
```

**Expected output**

```text
Pipeline fitted
```

**Explanation**

The pipeline fits transformations and the estimator using only X_train and y_train. With a tiny teaching dataset, model quality is not meaningful; this demonstrates workflow structure.



### 14.5 Predict on held-out data

### 14.5. Generate test predictions

```python
predictions = model_pipeline.predict(X_test)

print(predictions)
```

**Expected output**

```text
A sequence of predicted class labels
```

**Explanation**

The pipeline applies the already fitted transformations to X_test and then calls the model. It does not refit preprocessing on test data.



## 15. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Assign arbitrary numbers to nominal categories | Creates a false order or distance | Use suitable nominal encoding |
| Ordinal-encode unordered labels | Invents ranking | Use one-hot or another justified encoding |
| Fit scaler before splitting | Leaks held-out information | Split first; fit on training only |
| Fit a new scaler on test data | Test representation differs from training | Use transform() with the fitted scaler |
| Scale every column | Categories and IDs need different treatment | Use column-specific preprocessing |
| Ignore unseen categories | Inference may fail | Define unknown-category handling |
| Forget feature names after transform | Harder to audit model inputs | Retain output feature names |
| Assume scaling always improves performance | Tree models may not need it | Match preprocessing to algorithm and validate |
| Use tiny demonstration data as proof | Results are unstable | Evaluate on suitable representative data |

## 16. Interview-ready explanations

**What is one-hot encoding?**  
It converts an unordered category into indicator columns. Each output column represents one category, avoiding an artificial numeric ranking.

**When is ordinal encoding appropriate?**  
When categories have a meaningful order, such as Low, Medium and High, and the mapping is defined by the business meaning.

**Why do we scale numerical features?**  
Some algorithms depend on distances or optimization and can be influenced by features with larger numerical ranges. Scaling puts features on a more comparable representation.

**What is the difference between StandardScaler and MinMaxScaler?**  
StandardScaler centers features and scales their spread using training statistics. MinMaxScaler maps the training range to a chosen interval, often zero to one. Min-max scaling can be sensitive to extreme values.

**Why use ColumnTransformer and Pipeline?**  
They organize different transformations by column type and ensure preprocessing is fitted during training and reused consistently for validation, test and inference.

## 17. Scenario questions

1. A city column is encoded as Hyderabad=1, Chennai=2 and Vizag=3. What is the concern?  
   **Answer:** The codes may imply an order or distance that does not exist. Use a suitable nominal encoding.

2. A distance-based model is dominated by annual income rather than age. What should you inspect?  
   **Answer:** Compare feature ranges and consider suitable scaling fitted on training data.

3. A new city appears during production prediction. What should the pipeline do?  
   **Answer:** Apply the documented unknown-category strategy, such as an encoder configured to ignore unknown values, and monitor the new category.

4. A scaler was fitted before train-test splitting. What is the issue?  
   **Answer:** The held-out data influenced the learned preprocessing statistics. Rebuild the evaluation with split-first, train-only fitting.

5. A tree-based model performs similarly with and without scaling. Is that surprising?  
   **Answer:** Not necessarily. Tree-based models are generally less sensitive to monotonic feature scaling than distance-based or gradient-based methods.

## 18. Lesson completion checklist

- [ ] I can identify nominal and ordinal categories.
- [ ] I can use one-hot encoding.
- [ ] I can define an explicit ordinal mapping.
- [ ] I can explain why arbitrary category numbers are risky.
- [ ] I can explain feature scaling.
- [ ] I can use StandardScaler and MinMaxScaler.
- [ ] I know why training-only fitting matters.
- [ ] I can transform held-out data without refitting.
- [ ] I can use ColumnTransformer and Pipeline.
- [ ] I can inspect transformed feature names and shapes.

**Next lesson:** Day 13 — Train, Validation and Test Split.
