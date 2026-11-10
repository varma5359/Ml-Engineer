# Day 50: Feature Engineering and Transformation

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives
- Explain feature engineering and why it matters.
- Create useful features from dates, categories and numerical columns.
- Handle skewed numerical features with transformations.
- Create bins and interaction features when appropriate.
- Understand one-hot and ordinal encoding choices.
- Use ColumnTransformer and Pipeline.
- Avoid leakage when creating features.
- Validate whether engineered features improve model performance.

## 2. What is feature engineering?

Feature engineering is the process of creating, transforming or selecting input variables so that a machine learning model can learn useful patterns.

Examples:
- Convert date of birth into age at a defined reference date.
- Convert transaction timestamp into hour, day or month.
- Calculate customer purchase frequency from transaction history.
- Create total spend from item price and quantity.
- Group a numerical value into meaningful ranges.
- Transform a highly skewed numerical variable.

A model learns from the features supplied to it. Useful features can help a simple model perform better, while poor features can make a sophisticated model unreliable.

## 3. Feature engineering versus feature selection

- Feature engineering creates or transforms features.
- Feature selection chooses which features to retain.
- Feature engineering changes the representation of information.
- Feature selection reduces the input set.

Both should be validated using a leakage-safe workflow.

## 4. Start with a business scenario

A subscription company wants to predict customer churn. The raw dataset contains:
- Signup date.
- Last login date.
- Monthly bill.
- Number of support calls.
- Total purchases.
- Subscription type.

Potential engineered features:
- Customer tenure in months.
- Days since last login.
- Average purchase value.
- Support calls per month.
- Subscription category indicators.

A feature must be calculated using information available at the prediction time. Do not use future events to create historical prediction features.

## 5. Create a sample customer dataset

### 5.1 Import libraries

~~~python
import numpy as np
import pandas as pd

print("Libraries ready")
~~~

### 5.2 Create raw customer data

~~~python
customer_df = pd.DataFrame({
    "customer_id": ["C101", "C102", "C103", "C104", "C105"],
    "signup_date": [
        "2024-01-10", "2024-04-15", "2023-11-20",
        "2024-06-01", "2023-08-05"
    ],
    "last_login": [
        "2024-06-10", "2024-06-08", "2024-06-01",
        "2024-06-11", "2024-05-20"
    ],
    "monthly_bill": [499, 999, 699, 1499, 299],
    "total_purchases": [12, 4, 20, 2, 8],
    "support_calls": [1, 5, 2, 8, 0],
    "subscription_type": [
        "Basic", "Premium", "Standard", "Premium", "Basic"
    ]
})

print(customer_df)
~~~

**Expected output**
A five-row customer table with date, numerical and categorical columns.

### 5.3 Inspect column types

~~~python
print(customer_df.dtypes)
~~~

**Explanation**
Dates are currently strings. We must convert them before extracting date-related features.

## 6. Create date and time features

Date columns contain information that can be useful after extracting appropriate components.

Possible date features:
- Year.
- Month.
- Day of week.
- Days since last activity.
- Customer tenure at a defined observation date.
- Whether a date falls on a weekend.

### 6.1 Convert signup date to datetime

~~~python
customer_df["signup_date"] = pd.to_datetime(
    customer_df["signup_date"],
    errors="coerce"
)

print(customer_df["signup_date"])
~~~

**Expected output**
A datetime Series.

**Explanation**
errors="coerce" converts invalid date strings to missing values. Inspect the resulting missing values rather than silently ignoring invalid source data.

### 6.2 Convert last login to datetime

~~~python
customer_df["last_login"] = pd.to_datetime(
    customer_df["last_login"],
    errors="coerce"
)

print(customer_df["last_login"])
~~~

### 6.3 Define a fixed observation date

~~~python
observation_date = pd.Timestamp("2024-06-15")

print(observation_date)
~~~

**Explanation**
For reproducible training, define the date at which the prediction is assumed to be made. Do not calculate historical features using today's date when recreating an older prediction.

### 6.4 Create customer tenure in days

~~~python
customer_df["tenure_days"] = (
    observation_date - customer_df["signup_date"]
).dt.days

print(customer_df[["customer_id", "tenure_days"]])
~~~

**Explanation**
The feature measures how long each customer has been registered at the observation date.

### 6.5 Create days since last login

~~~python
customer_df["days_since_last_login"] = (
    observation_date - customer_df["last_login"]
).dt.days

print(customer_df[["customer_id", "days_since_last_login"]])
~~~

**Explanation**
This feature describes recent activity. The observation date must be earlier than the outcome being predicted.

### 6.6 Extract signup month

~~~python
customer_df["signup_month"] = customer_df["signup_date"].dt.month

print(customer_df[["signup_date", "signup_month"]])
~~~

### 6.7 Extract signup weekday

~~~python
customer_df["signup_weekday"] = customer_df["signup_date"].dt.day_name()

print(customer_df[["signup_date", "signup_weekday"]])
~~~

**Explanation**
Month and weekday can capture seasonal or operational patterns. They are not automatically useful; validate their contribution.

## 7. Create numerical derived features

### 7.1 Calculate average purchases per month

~~~python
customer_df["tenure_months"] = (
    customer_df["tenure_days"] / 30.44
).clip(lower=1)

customer_df["purchases_per_month"] = (
    customer_df["total_purchases"] /
    customer_df["tenure_months"]
)

print(
    customer_df[
        ["customer_id", "purchases_per_month"]
    ]
)
~~~

**Explanation**
This creates an approximate purchase frequency. The definition of the time window should match the business question.

### 7.2 Calculate support calls per month

~~~python
customer_df["support_calls_per_month"] = (
    customer_df["support_calls"] /
    customer_df["tenure_months"]
)

print(
    customer_df[
        ["customer_id", "support_calls_per_month"]
    ]
)
~~~

**Explanation**
A normalized rate can make customers with different tenure lengths more comparable.

### 7.3 Create a high-support indicator

~~~python
customer_df["frequent_support_user"] = (
    customer_df["support_calls"] >= 5
).astype(int)

print(
    customer_df[
        ["support_calls", "frequent_support_user"]
    ]
)
~~~

**Explanation**
This creates a binary indicator from a business-defined threshold. The threshold should be justified and validated, not chosen arbitrarily.

## 8. Handle categorical features

Categorical columns contain labels such as Basic, Standard and Premium.

Common approaches:
- One-hot encoding for categories without natural order.
- Ordinal encoding when a meaningful order exists and the model can use that ordering appropriately.
- Grouping rare categories when justified by domain context.

Do not assign arbitrary numeric codes to nominal categories and assume the numbers have meaningful distance.

### 8.1 Inspect subscription categories

~~~python
print(customer_df["subscription_type"].value_counts())
~~~

### 8.2 Apply one-hot encoding with Pandas

~~~python
subscription_encoded = pd.get_dummies(
    customer_df["subscription_type"],
    prefix="plan",
    dtype=int
)

print(subscription_encoded)
~~~

**Explanation**
Each category receives its own indicator column. A row has a value of 1 for its plan and 0 for other plans.

### 8.3 Join encoded columns

~~~python
customer_encoded = pd.concat(
    [customer_df, subscription_encoded],
    axis=1
)

print(customer_encoded.head())
~~~

**Explanation**
The encoded columns are added to the original table. In a production training pipeline, prefer a fitted encoder so future inputs use the same known category mapping.

## 9. Numerical transformations

Some numerical features are highly skewed. A transformation can make patterns easier for certain models to learn.

Common options:
- Log transformation for positive, right-skewed data.
- Power transformations for distribution shape.
- Robust scaling when outliers affect standard scaling.
- Binning when meaningful ranges are needed.

Transformations should be chosen after inspecting the feature distribution.

### 9.1 Create a skewed example

~~~python
spend = pd.Series([100, 120, 150, 180, 250, 400, 900, 5000])

print(spend.describe())
~~~

### 9.2 Inspect the distribution

~~~python
import matplotlib.pyplot as plt

plt.hist(spend, bins=8)
plt.xlabel("Spend")
plt.ylabel("Count")
plt.title("Original Spend Distribution")
plt.show()
~~~

### 9.3 Apply log1p transformation

~~~python
spend_log = np.log1p(spend)

print(spend_log)
~~~

**Explanation**
log1p applies a logarithmic transformation that can handle zero values. It cannot be used directly for values below -1. Check feature meaning and value range before applying it.

### 9.4 Compare original and transformed distributions

~~~python
plt.hist(spend_log, bins=8)
plt.xlabel("Log transformed spend")
plt.ylabel("Count")
plt.title("Transformed Spend Distribution")
plt.show()
~~~

**Explanation**
The transformation compresses large values. It does not remove outliers or guarantee a better model.

## 10. Binning numerical features

Binning converts a continuous numerical feature into ranges.

Example:
- Tenure below 6 months: New.
- Tenure from 6 to 24 months: Established.
- Tenure above 24 months: Long-term.

Use bins when:
- Business rules already define meaningful bands.
- Nonlinear ranges are easier to communicate.
- The model benefits from a grouped representation.

Limitations:
- Binning loses detail.
- Arbitrary boundaries can hide useful patterns.
- Thresholds should be defined using training data or business rules, not test outcomes.

### 10.1 Create tenure groups

~~~python
customer_df["tenure_group"] = pd.cut(
    customer_df["tenure_months"],
    bins=[0, 6, 24, np.inf],
    labels=["New", "Established", "Long-term"]
)

print(
    customer_df[
        ["tenure_months", "tenure_group"]
    ]
)
~~~

### 10.2 Inspect group counts

~~~python
print(customer_df["tenure_group"].value_counts())
~~~

## 11. Interaction features

An interaction feature combines two or more existing features to represent a relationship.

Example:
- Monthly bill multiplied by tenure.
- Price multiplied by quantity.
- Support calls combined with inactivity.
- Discount amount compared with original price.

Interactions can help models represent useful combinations, but too many interactions increase complexity and overfitting risk.

### 11.1 Create total billed value proxy

~~~python
customer_df["bill_tenure_proxy"] = (
    customer_df["monthly_bill"] *
    customer_df["tenure_months"]
)

print(
    customer_df[
        ["monthly_bill", "tenure_months", "bill_tenure_proxy"]
    ]
)
~~~

**Explanation**
This is a simple proxy for cumulative billed value, not an exact accounting measure. Validate the business definition before using it.

## 12. Build a reusable preprocessing workflow

For mixed numerical and categorical data, use ColumnTransformer and Pipeline.

Benefits:
- Apply transformations to selected columns.
- Keep preprocessing consistent between training and prediction.
- Fit transformations only on training folds during cross-validation.
- Reduce manual feature-order mistakes.

### 12.1 Define numerical and categorical columns

~~~python
numeric_features = [
    "monthly_bill",
    "total_purchases",
    "support_calls"
]

categorical_features = [
    "subscription_type"
]

print(numeric_features)
print(categorical_features)
~~~

### 12.2 Import preprocessing tools

~~~python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
~~~

### 12.3 Create numerical preprocessing

~~~python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

print(numeric_pipeline)
~~~

### 12.4 Create categorical preprocessing

~~~python
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

print(categorical_pipeline)
~~~

**Explanation**
handle_unknown="ignore" allows the encoder to process an unseen category without failing. The unseen category does not receive a learned dedicated indicator unless it was represented during fitting.

### 12.5 Combine transformations

~~~python
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print(preprocessor)
~~~

### 12.6 Fit and transform a training table

~~~python
X_raw = customer_df[
    numeric_features + categorical_features
]

X_transformed = preprocessor.fit_transform(X_raw)

print(X_transformed.shape)
~~~

**Explanation**
This demonstration fits preprocessing on the example table. In a real model evaluation, fit it only on the training fold or training split.

## 13. Avoid feature engineering leakage

Leakage occurs when a feature contains information that would not be available at the time the prediction is made, or when evaluation information influences training decisions.

Examples:
- Using cancellation date to predict churn before cancellation.
- Calculating customer lifetime spend using transactions after the prediction date.
- Creating target encoding using the full dataset before splitting.
- Computing a global average using test observations.
- Using a future event count in a historical feature.

Safe workflow:
1. Define prediction timestamp.
2. Define the feature observation window.
3. Use only data available at that timestamp.
4. Create train and test partitions appropriately.
5. Fit learned transformations on training data only.
6. Apply the fitted transformations to validation and test data.

## 14. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Extract dates without defining observation time | Define a historical prediction timestamp |
| Use future transactions in customer features | Restrict features to the observation window |
| Encode nominal categories as arbitrary ranks | Use one-hot encoding or a suitable encoder |
| Apply log to negative values without checking | Inspect range and choose a valid transformation |
| Create too many interaction features | Add only justified features and validate |
| Fit encoders on all data before splitting | Fit inside the training Pipeline |
| Assume binning always improves performance | Compare with continuous feature baseline |
| Treat transformed values as original units | Document and interpret transformations |

## 15. Interview-ready explanations

**What is feature engineering?**
- Creating or transforming inputs to represent useful patterns for a model.
- Examples include date parts, rates, ratios and interaction features.
- Features must be valid at prediction time.

**Why create date features?**
- Dates can contain seasonality, recency and lifecycle information.
- Extract meaningful components such as month, weekday or days since activity.
- Avoid using information from after the prediction timestamp.

**When use one-hot encoding?**
- For nominal categories with no natural order.
- It creates indicator columns so a model does not interpret arbitrary category numbers as meaningful distances.

**What is data leakage in feature engineering?**
- Using information unavailable at prediction time or allowing evaluation data to influence feature creation.
- It can make validation performance look unrealistically good.

## 16. Scenario-based interview practice

### Scenario 1: Customer tenure is a strong feature
**Expected response:**
- Define tenure relative to the prediction date.
- Check that signup date is valid.
- Calculate tenure consistently for training and future inputs.
- Review whether tenure adds value on held-out data.

### Scenario 2: A churn model uses cancellation date
**Expected response:**
- Cancellation date may reveal the outcome.
- It is unavailable before the customer churns.
- Remove it from pre-churn prediction features.
- Rebuild features using only information available at prediction time.

### Scenario 3: A new subscription category appears after deployment
**Expected response:**
- Check the encoder's unknown-category behaviour.
- Use a Pipeline with a documented category handling policy.
- Monitor unseen category frequency and decide whether retraining or schema updates are needed.

### Scenario 4: Log transformation improves one metric but reduces interpretability
**Expected response:**
- Explain why the transformation was applied.
- Compare performance and residuals.
- Decide whether the improvement justifies the interpretation cost.
- Keep clear documentation and transform predictions back when necessary for business reporting.

## 17. Lesson completion checklist

- [ ] I can explain feature engineering.
- [ ] I can create date-based features.
- [ ] I can create numerical rates and interaction features.
- [ ] I can encode categorical columns.
- [ ] I can apply log transformations and binning carefully.
- [ ] I can build a ColumnTransformer and Pipeline.
- [ ] I understand prediction-time leakage.
- [ ] I can justify engineered features in an interview.

**Next lesson:** Day 51 — Imbalanced Datasets and Resampling Concepts.
