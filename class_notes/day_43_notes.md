# Day 43: Unsupervised Learning Project Integration

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 3 hours

## 1. Learning objectives

- Connect data preparation, clustering, interpretation and evaluation into one workflow.
- Define a clustering use case and observation unit.
- Select meaningful features and exclude identifiers.
- Validate and prepare numerical inputs.
- Build an imputation, scaling and K-Means Pipeline.
- Fit the clustering model using historical observations.
- Assign clusters to incoming observations.
- Profile clusters using original feature units.
- Save and reload the complete pipeline.
- Validate input schema and monitor cluster assignments.
- Document limitations and responsible use.

## 2. What does project integration mean?

A clustering workflow is more than calling KMeans once.

A practical workflow includes:
1. Define the business question.
2. Identify what one row represents.
3. Select meaningful features.
4. Inspect and prepare data.
5. Fit preprocessing and clustering.
6. Evaluate candidate settings.
7. Interpret and profile clusters.
8. Assign new observations consistently.
9. Save the complete workflow.
10. Monitor changes and document limitations.

The goal of this lesson is to connect the techniques from Days 33–42 into one reproducible workflow.

## 3. Business scenario — Customer segmentation

A retail company wants to group customers using behaviour such as:
- Monthly spending.
- Purchase frequency.
- Customer tenure.
- Return frequency.

The company does not have predefined segment labels. It wants to explore groups and understand whether they can support different customer engagement strategies.

The clustering model does not decide which customer is valuable or which campaign should be sent. It discovers patterns that the business team must review.

## 4. Define the observation unit

Before clustering, define what one row represents.

In this lesson:
- One row represents one customer.
- Each customer appears once in the modelling table.
- Features describe a defined historical observation period.
- Customer ID is used for identification and reporting, not as a distance feature.

If one row represented an individual purchase instead, the discovered groups would describe transactions rather than customers. The observation unit changes the meaning of the model.

## 5. Create an example customer dataset

### 5.1 Import libraries

### 5.1. Import required libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries ready")
```

**Expected output**

```text
Libraries ready
```

**Explanation**

These libraries support data creation, preparation, analysis and visualization.



### 5.2 Generate customer behaviour

### 5.2. Create a simulated customer population

```python
rng = np.random.default_rng(42)

n_customers = 180

customer_data = pd.DataFrame({
    "customer_id": [f"C{i:04d}" for i in range(1, n_customers + 1)],
    "monthly_spend": rng.lognormal(mean=7.7, sigma=0.55, size=n_customers),
    "purchase_frequency": rng.poisson(lam=5, size=n_customers) + 1,
    "tenure_months": rng.integers(1, 61, size=n_customers),
    "return_count": rng.poisson(lam=1.2, size=n_customers)
})

print(customer_data.head())
```

**Expected output**

```text
A five-row customer table
```

**Explanation**

The dataset is simulated for workflow practice. Its values should not be interpreted as real retail behaviour.



### 5.3 Add a few missing feature values

### 5.3. Introduce missing values for preprocessing practice

```python
customer_data.loc[[4, 18, 37], "monthly_spend"] = np.nan
customer_data.loc[[8, 45], "purchase_frequency"] = np.nan
customer_data.loc[[20, 61], "return_count"] = np.nan

print(customer_data.isnull().sum())
```

**Expected output**

```text
Missing-value counts for selected columns
```

**Explanation**

The missing values allow us to demonstrate imputation inside a pipeline. In real work, investigate why values are missing before selecting an imputation rule.



### 5.4 Inspect shape and types

### 5.4. Check dataset structure

```python
print(customer_data.shape)
print(customer_data.dtypes)
```

**Expected output**

```text
The dataset shape and column data types
```

**Explanation**

Confirm the expected row count, identifier type and numerical feature types.



### 5.5 Inspect numerical summaries

### 5.5. Review descriptive statistics

```python
print(customer_data.describe().T)
```

**Expected output**

```text
Summary statistics for the numerical columns
```

**Explanation**

Inspect ranges and distributions before choosing features and preprocessing.



## 6. Define the feature schema

### 6.1 Create a list of modelling features

### 6.1. Select behaviour features

```python
feature_columns = [
    "monthly_spend",
    "purchase_frequency",
    "tenure_months",
    "return_count"
]

print(feature_columns)
```

**Expected output**

```text
A list containing four feature names
```

**Explanation**

These features describe customer behaviour. The identifier is intentionally excluded.



### 6.2 Keep the customer identifier separately

### 6.2. Create the identifier series

```python
customer_ids = customer_data["customer_id"]

print(customer_ids.head())
```

**Expected output**

```text
The first five customer identifiers
```

**Explanation**

The ID is retained for joining results back to customers, not used as a clustering feature.



### 6.3 Create the feature DataFrame

### 6.3. Build the modelling table

```python
X = customer_data[feature_columns].copy()

print(X.head())
```

**Expected output**

```text
A five-row table containing four numerical features
```

**Explanation**

The feature schema is explicit and can be reused when new customer records arrive.



### 6.4 Check missing values

### 6.4. Count missing values in model features

```python
print(X.isnull().sum())
```

**Expected output**

```text
Missing-value counts for the selected feature columns
```

**Explanation**

This check helps confirm which columns require imputation or further investigation.



### 6.5 Check for duplicate customer identifiers

### 6.5. Validate the customer key

```python
print(customer_data["customer_id"].duplicated().sum())
```

**Expected output**

```text
0
```

**Explanation**

The example expects one row per customer. A real dataset should validate the observation unit and business key before clustering.



## 7. Prepare historical and incoming records

For this workflow, we separate historical customers used to fit the segmentation from incoming customers that will receive a cluster assignment.

### 7.1 Split the customer table

### 7.1. Create historical and incoming subsets

```python
from sklearn.model_selection import train_test_split

historical_customers, incoming_customers = train_test_split(
    customer_data,
    test_size=0.2,
    random_state=42
)

print(historical_customers.shape)
print(incoming_customers.shape)
```

**Expected output**

```text
Historical and incoming customer table shapes
```

**Explanation**

This split demonstrates how a fitted segmentation can be applied to new records. It is not a supervised train-test evaluation.



### 7.2 Select historical features

### 7.2. Create historical modelling inputs

```python
X_historical = historical_customers[feature_columns].copy()

print(X_historical.head())
```

**Expected output**

```text
A preview of historical feature values
```

**Explanation**

Only the selected behavioural features are passed to the pipeline.



### 7.3 Select incoming features

### 7.3. Create incoming modelling inputs

```python
X_incoming = incoming_customers[feature_columns].copy()

print(X_incoming.head())
```

**Expected output**

```text
A preview of incoming customer features
```

**Explanation**

The incoming data must use the same feature names and meaning as the historical data.



## 8. Build preprocessing and clustering together

A Pipeline ensures that missing-value treatment, scaling and K-Means are stored as one workflow.

### 8.1 Import preprocessing tools

### 8.1. Import imputer and scaler

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Preprocessing tools ready")
```

**Expected output**

```text
Preprocessing tools ready
```

**Explanation**

SimpleImputer fills missing numerical values and StandardScaler transforms feature ranges.



### 8.2 Import KMeans

### 8.2. Import KMeans

```python
from sklearn.cluster import KMeans

print("KMeans ready")
```

**Expected output**

```text
KMeans ready
```

**Explanation**

The estimator discovers centroid-based customer groups.



### 8.3 Create the full pipeline

### 8.3. Connect imputation, scaling and clustering

```python
segmentation_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=3,
        n_init=20,
        random_state=42
    ))
])

print(segmentation_pipeline)
```

**Expected output**

```text
A Pipeline containing SimpleImputer, StandardScaler and KMeans
```

**Explanation**

The pipeline ensures that imputation and scaling are fitted on historical data and applied consistently to incoming customers.



### 8.4 Understand the selected settings

- SimpleImputer(strategy="median") fills missing numeric values using each feature's training median.
- StandardScaler standardizes the numerical features.
- KMeans(n_clusters=3) creates three customer groups.
- n_init=20 attempts multiple centroid initializations.
- random_state=42 supports repeatability.

The chosen cluster count is an example. In a real workflow, evaluate candidate settings using the methods from Day 42 before finalizing it.

## 9. Fit the clustering pipeline

### 9.1 Fit using historical customers

### 9.1. Train the complete segmentation pipeline

```python
segmentation_pipeline.fit(X_historical)

print("Customer segmentation fitted")
```

**Expected output**

```text
Customer segmentation fitted
```

**Explanation**

The imputer, scaler and K-Means model are all fitted using historical customer data.



### 9.2 Inspect learned cluster centres

### 9.2. Read centroids in scaled feature space

```python
scaled_centres = segmentation_pipeline.named_steps[
    "kmeans"
].cluster_centers_

print(scaled_centres)
```

**Expected output**

```text
Three centroid rows in scaled feature space
```

**Explanation**

These centres are expressed in the transformed feature space, not in original customer units.



### 9.3 Convert centroids to original units

### 9.3. Inverse-transform the centroids

```python
original_centres = segmentation_pipeline.named_steps[
    "scaler"
].inverse_transform(scaled_centres)

print(original_centres)
```

**Expected output**

```text
Three centroid rows in original numerical feature units
```

**Explanation**

The scaler is reversed so that centroids can be described using spend, purchase frequency, tenure and returns. Imputation has already been applied before scaling.



### 9.4 Create a readable centroid table

### 9.4. Attach feature names to centroids

```python
centroid_table = pd.DataFrame(
    original_centres,
    columns=feature_columns
)

centroid_table["cluster"] = range(3)

print(centroid_table)
```

**Expected output**

```text
A table with one centroid row for each cluster
```

**Explanation**

This provides a first summary of the customer groups. It should be interpreted with full cluster distributions, not by itself.



## 10. Assign clusters to historical customers

### 10.1 Predict historical cluster labels

### 10.1. Assign each historical customer

```python
historical_labels = segmentation_pipeline.predict(
    X_historical
)

print(pd.Series(historical_labels).value_counts().sort_index())
```

**Expected output**

```text
Counts for the three cluster IDs
```

**Explanation**

The fitted pipeline applies its imputer and scaler before assigning customers to the nearest learned centroid.



### 10.2 Add historical cluster labels

### 10.2. Create historical assignment results

```python
historical_results = historical_customers[
    ["customer_id"] + feature_columns
].copy()

historical_results["cluster"] = historical_labels

print(historical_results.head())
```

**Expected output**

```text
A preview containing customer IDs, original features and cluster assignments
```

**Explanation**

The identifiers are added back after clustering so that the business can review the assigned customers.



### 10.3 Check cluster sizes

### 10.3. Summarize historical cluster sizes

```python
historical_cluster_sizes = historical_results[
    "cluster"
].value_counts().sort_index()

print(historical_cluster_sizes)
```

**Expected output**

```text
A count for each cluster
```

**Explanation**

Cluster sizes help reveal dominant or very small groups.



## 11. Profile clusters using original data

### 11.1 Create a cluster profile

### 11.1. Calculate customer behaviour summaries

```python
cluster_profile = historical_results.groupby("cluster").agg(
    customer_count=("customer_id", "count"),
    average_monthly_spend=("monthly_spend", "mean"),
    median_monthly_spend=("monthly_spend", "median"),
    average_purchase_frequency=("purchase_frequency", "mean"),
    average_tenure=("tenure_months", "mean"),
    average_return_count=("return_count", "mean")
)

print(cluster_profile)
```

**Expected output**

```text
A profile table for the three customer groups
```

**Explanation**

The profile uses original customer units and gives business users a description of each group.



### 11.2 Calculate cluster percentages

### 11.2. Add cluster proportions

```python
cluster_profile["customer_percentage"] = (
    cluster_profile["customer_count"] /
    len(historical_results) * 100
)

print(cluster_profile)
```

**Expected output**

```text
A cluster profile with customer percentages
```

**Explanation**

Percentages help communicate how much of the historical population belongs to each group.



### 11.3 Sort by spending

### 11.3. Order profiles by average monthly spend

```python
print(
    cluster_profile.sort_values(
        "average_monthly_spend",
        ascending=False
    )
)
```

**Expected output**

```text
Cluster profiles ordered by average spending
```

**Explanation**

Sorting helps compare groups, but it does not change arbitrary cluster IDs into a permanent ranking.



### 11.4 Plot cluster sizes

### 11.4. Visualize group sizes

```python
cluster_profile["customer_count"].plot(kind="bar")
plt.xlabel("Cluster ID")
plt.ylabel("Customer count")
plt.title("Historical customer cluster sizes")
plt.show()
```

**Expected output**

```text
A bar chart showing the size of each cluster
```

**Explanation**

This makes very large or small groups easy to identify.



### 11.5 Compare spending distributions

### 11.5. Plot monthly spend by cluster

```python
sns.boxplot(
    data=historical_results,
    x="cluster",
    y="monthly_spend"
)
plt.xlabel("Cluster")
plt.ylabel("Monthly spend")
plt.title("Monthly spending across customer clusters")
plt.show()
```

**Expected output**

```text
A box plot comparing spending distributions
```

**Explanation**

The box plot reveals spread and potential outliers that a cluster mean may hide.



## 12. Assign clusters to incoming customers

### 12.1 Predict incoming cluster labels

### 12.1. Assign incoming customers to existing clusters

```python
incoming_labels = segmentation_pipeline.predict(
    X_incoming
)

print(incoming_labels[:15])
```

**Expected output**

```text
The first fifteen incoming cluster IDs
```

**Explanation**

The model does not create a new clustering solution. It assigns each incoming customer to one of the existing learned centroids.



### 12.2 Create incoming assignment results

### 12.2. Build the incoming customer result table

```python
incoming_results = incoming_customers[
    ["customer_id"] + feature_columns
].copy()

incoming_results["cluster"] = incoming_labels

print(incoming_results.head())
```

**Expected output**

```text
A preview of incoming customers with assigned cluster IDs
```

**Explanation**

This table can be used by downstream reporting or business workflows after validation.



### 12.3 Count incoming assignments

### 12.3. Summarize incoming cluster counts

```python
incoming_counts = incoming_results[
    "cluster"
].value_counts().sort_index()

print(incoming_counts)
```

**Expected output**

```text
A count for each existing cluster
```

**Explanation**

The incoming distribution can be compared with historical cluster distribution as one monitoring signal.



### 12.4 Compare historical and incoming distributions

### 12.4. Create a distribution comparison

```python
distribution_comparison = pd.DataFrame({
    "historical": historical_cluster_sizes,
    "incoming": incoming_counts
}).fillna(0)

print(distribution_comparison)
```

**Expected output**

```text
A table comparing historical and incoming cluster counts
```

**Explanation**

Differences may reflect sampling variation or a change in customer behaviour. Investigate before declaring drift.



## 13. Validate incoming feature schema

Production inputs may contain missing columns, extra columns or values in unexpected formats.

### 13.1 Define a reusable schema validator

### 13.1. Create a feature validation function

```python
def validate_feature_schema(data, required_columns):
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return data[required_columns].copy()

print("Schema validator defined")
```

**Expected output**

```text
Schema validator defined
```

**Explanation**

The function verifies that every required feature exists and returns only the expected feature columns.



### 13.2 Validate incoming data

### 13.2. Check the incoming feature table

```python
validated_incoming = validate_feature_schema(
    incoming_customers,
    feature_columns
)

print(validated_incoming.head())
```

**Expected output**

```text
A preview containing the four required features
```

**Explanation**

Extra columns such as customer_id are excluded from model input, while required feature columns are preserved.



### 13.3 Demonstrate a missing-column error

### 13.3. Test an invalid input schema

```python
invalid_input = incoming_customers.drop(
    columns=["monthly_spend"]
)

try:
    validate_feature_schema(invalid_input, feature_columns)
except ValueError as error:
    print(error)
```

**Expected output**

```text
A clear error message naming monthly_spend as missing
```

**Explanation**

A clear schema error is safer than silently passing incomplete inputs to the model.



## 14. Check numeric feature values

### 14.1 Create a numeric conversion function

### 14.1. Convert expected numeric columns safely

```python
def convert_numeric_features(data, columns):
    converted = data.copy()
    invalid_counts = {}

    for column in columns:
        original_not_missing = converted[column].notna()

        converted[column] = pd.to_numeric(
            converted[column],
            errors="coerce"
        )

        invalid_counts[column] = int(
            (
                original_not_missing &
                converted[column].isna()
            ).sum()
        )

    return converted, invalid_counts

print("Numeric validation function defined")
```

**Expected output**

```text
Numeric validation function defined
```

**Explanation**

The function converts numeric-looking values and counts non-missing values that could not be converted.



### 14.2 Validate incoming numerical values

### 14.2. Run numeric conversion

```python
numeric_incoming, invalid_counts = convert_numeric_features(
    validated_incoming,
    feature_columns
)

print(invalid_counts)
```

**Expected output**

```text
A dictionary showing invalid-value counts by feature
```

**Explanation**

Invalid text becomes missing in this demonstration. Investigate those records and confirm that the imputation policy is appropriate before using the result.



### 14.3 Inspect missing values after conversion

### 14.3. Check missing and invalid values

```python
print(numeric_incoming.isna().sum())
```

**Expected output**

```text
Missing-value counts after numeric conversion
```

**Explanation**

This includes original missing values and invalid numeric representations. The pipeline's imputer will apply the learned historical medians.



### 14.4 Predict after schema validation

### 14.4. Assign clusters to validated numeric data

```python
validated_labels = segmentation_pipeline.predict(
    numeric_incoming
)

print(validated_labels[:10])
```

**Expected output**

```text
The first ten cluster assignments
```

**Explanation**

The fitted pipeline applies the historical imputer and scaler consistently to the validated incoming values.



## 15. Inspect distance to the nearest centroid

K-Means can report distances from each observation to each centroid. A large nearest-centroid distance may indicate that an observation is unlike the historical groups.

This is a diagnostic signal, not a calibrated anomaly probability.

### 15.1 Transform incoming observations to centroid distances

### 15.1. Calculate distances to cluster centres

```python
incoming_distances = segmentation_pipeline.transform(
    numeric_incoming
)

print(incoming_distances[:5])
```

**Expected output**

```text
A distance matrix with one column per centroid
```

**Explanation**

Pipeline.transform applies imputation and scaling, then returns distances from each incoming observation to each learned centroid.



### 15.2 Find the nearest-centroid distance

### 15.2. Extract each customer's nearest-centroid distance

```python
nearest_centroid_distance = incoming_distances.min(axis=1)

print(nearest_centroid_distance[:10])
```

**Expected output**

```text
The first ten nearest-centroid distances
```

**Explanation**

A larger value means the observation is farther from every learned centroid in transformed feature space.



### 15.3 Add distances to incoming results

### 15.3. Create an assignment review table

```python
incoming_results["nearest_centroid_distance"] = nearest_centroid_distance

print(
    incoming_results.sort_values(
        "nearest_centroid_distance",
        ascending=False
    ).head(10)
)
```

**Expected output**

```text
The ten incoming customers with the largest nearest-centroid distance
```

**Explanation**

These records may deserve review because they are poorly represented by the historical centroids. Set any review threshold using historical validation evidence and domain knowledge.



## 16. Save the complete pipeline

Save preprocessing and clustering together. Saving only KMeans would lose the imputation and scaling behaviour required for consistent future predictions.

### 16.1 Create an artifacts folder

### 16.1. Prepare the artifact directory

```python
from pathlib import Path

artifact_dir = Path("artifacts")
artifact_dir.mkdir(exist_ok=True)

print(artifact_dir.exists())
```

**Expected output**

```text
True
```

**Explanation**

The directory stores the fitted pipeline and supporting metadata. In a production repository, artifact paths and retention rules should be configured explicitly.



### 16.2 Import joblib

### 16.2. Import joblib

```python
import joblib

print("joblib imported")
```

**Expected output**

```text
joblib imported
```

**Explanation**

joblib is commonly used to serialize scikit-learn estimators and pipelines.



### 16.3 Save the fitted pipeline

### 16.3. Write the complete pipeline

```python
model_path = artifact_dir / "customer_segmentation_pipeline.joblib"

joblib.dump(segmentation_pipeline, model_path)

print("Pipeline saved:", model_path)
```

**Expected output**

```text
A confirmation message with the artifact path
```

**Explanation**

The saved object contains the fitted imputer, scaler and K-Means model.



### 16.4 Verify the artifact exists

### 16.4. Check the saved artifact

```python
print(model_path.exists())
print(model_path.stat().st_size)
```

**Expected output**

```text
True and the artifact size in bytes
```

**Explanation**

This verifies that a file was written. It does not by itself prove that the artifact can be loaded correctly.



### 16.5 Load the pipeline

### 16.5. Load the saved pipeline

```python
loaded_pipeline = joblib.load(model_path)

print(type(loaded_pipeline))
```

**Expected output**

```text
A scikit-learn Pipeline type
```

**Explanation**

Load only artifacts from trusted sources. Serialized Python objects can execute code during loading.



### 16.6 Compare loaded and original predictions

### 16.6. Verify prediction consistency

```python
original_check = segmentation_pipeline.predict(
    numeric_incoming
)

loaded_check = loaded_pipeline.predict(
    numeric_incoming
)

print(np.array_equal(original_check, loaded_check))
```

**Expected output**

```text
True
```

**Explanation**

The check confirms that the loaded pipeline produces the same cluster labels for the same input in this environment.



## 17. Save model metadata

A production artifact should be accompanied by metadata describing the feature schema, algorithm settings and data context.

### 17.1 Create metadata

### 17.1. Prepare model metadata

```python
import json
from datetime import date

metadata = {
    "model_name": "customer_segmentation",
    "algorithm": "KMeans",
    "cluster_count": 3,
    "feature_columns": feature_columns,
    "observation_unit": "one row per customer",
    "preprocessing": [
        "median imputation",
        "standard scaling"
    ],
    "created_date": date.today().isoformat()
}

print(metadata)
```

**Expected output**

```text
A metadata dictionary
```

**Explanation**

Metadata records how the pipeline was built and how inputs should be supplied.



### 17.2 Save metadata as JSON

### 17.2. Write metadata to disk

```python
metadata_path = artifact_dir / "customer_segmentation_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("Metadata saved")
```

**Expected output**

```text
Metadata saved
```

**Explanation**

The JSON file can be inspected without loading the serialized estimator.



### 17.3 Read metadata

### 17.3. Load and inspect metadata

```python
with open(metadata_path, "r", encoding="utf-8") as file:
    loaded_metadata = json.load(file)

print(loaded_metadata)
```

**Expected output**

```text
The saved metadata dictionary
```

**Explanation**

The metadata provides the feature contract and high-level model settings.



## 18. Monitor cluster assignments

After deployment or operational use, monitor whether the incoming data continues to resemble the historical population.

Possible monitoring signals:
- Cluster assignment proportions.
- Feature distribution changes.
- Missing and invalid value counts.
- Nearest-centroid distances.
- New or unusual categories if categorical features are added.
- Business outcomes associated with each cluster.

A shift in cluster distribution is a signal to investigate, not automatic proof of model failure.

### 18.1 Calculate incoming cluster proportions

### 18.1. Calculate assignment proportions

```python
incoming_proportions = incoming_results["cluster"].value_counts(
    normalize=True
).sort_index()

print(incoming_proportions)
```

**Expected output**

```text
The proportion of incoming customers assigned to each cluster
```

**Explanation**

Proportions help compare new assignments with the historical population.



### 18.2 Calculate historical proportions

### 18.2. Calculate historical proportions

```python
historical_proportions = historical_results["cluster"].value_counts(
    normalize=True
).sort_index()

print(historical_proportions)
```

**Expected output**

```text
The historical proportion for each cluster
```

**Explanation**

This is the reference distribution from the historical customer sample.



### 18.3 Compare proportions

### 18.3. Create a distribution shift table

```python
proportion_comparison = pd.DataFrame({
    "historical": historical_proportions,
    "incoming": incoming_proportions
}).fillna(0)

proportion_comparison["difference"] = (
    proportion_comparison["incoming"] -
    proportion_comparison["historical"]
)

print(proportion_comparison)
```

**Expected output**

```text
A comparison of historical and incoming cluster proportions
```

**Explanation**

A large difference may indicate population change, sampling differences or changes in data collection. Investigate the reason before retraining.



### 18.4 Plot assignment proportions

### 18.4. Visualize historical and incoming proportions

```python
proportion_comparison[
    ["historical", "incoming"]
].plot(kind="bar")

plt.xlabel("Cluster ID")
plt.ylabel("Proportion")
plt.title("Historical and incoming cluster distributions")
plt.show()
```

**Expected output**

```text
A grouped bar chart comparing cluster proportions
```

**Explanation**

The chart helps identify large changes in the incoming assignment distribution.



## 19. Responsible interpretation and limitations

- Cluster IDs are arbitrary and can change after refitting.
- A customer segment is not a permanent customer identity.
- A cluster does not prove customer intent or future behaviour.
- Features must be available at the time the segmentation is used.
- Sensitive or proxy variables require careful governance review.
- Small clusters may be unstable.
- A model can become stale as customer behaviour changes.
- Business actions should be tested and monitored.
- Do not automatically deny services or benefits based only on an unsupervised cluster assignment.

## 20. Common integration mistakes

| Mistake | Better approach |
|---|---|
| Save only KMeans | Save the full preprocessing and clustering Pipeline |
| Include customer_id in distance features | Keep identifiers separate |
| Fit scaler separately for incoming data | Reuse the historical fitted scaler |
| Change feature order | Validate the feature schema |
| Ignore invalid numerical values | Count and review conversion failures |
| Interpret cluster IDs as ranked groups | Use profiles and business descriptions |
| Assign new customers by refitting every time | Use the saved model for consistent assignments |
| Ignore nearest-centroid distance | Monitor observations far from known groups |
| Treat cluster shift as automatic failure | Investigate data and population changes |
| Load an artifact from an unknown source | Load only trusted serialized artifacts |

## 21. Interview-ready explanations

**Why should preprocessing and K-Means be saved together?**  
The same imputation and scaling transformations must be applied to future records. Saving the complete pipeline keeps the training and assignment workflow consistent.

**How do you assign a new customer to an existing K-Means segmentation?**  
Use the fitted pipeline's predict method. It applies the learned preprocessing and assigns the observation to the nearest existing centroid.

**Why exclude customer ID from clustering features?**  
It is an identifier rather than a behavioural measurement. It should be retained separately for reporting and record lookup.

**What does a large nearest-centroid distance indicate?**  
The observation is far from all learned centroids in transformed feature space. It may represent unusual behaviour or a new population pattern, but the distance is not a probability.

**What should be monitored after deployment?**  
Input quality, feature distributions, cluster proportions, nearest-centroid distances, business outcomes and changes in the customer population.

## 22. Scenario questions

1. Incoming customers are assigned to one cluster much more often than historical customers. What do you investigate?  
   **Answer:** Feature distribution changes, missing values, source-system changes, population changes and whether the historical segmentation still represents current customers.

2. A new input is missing one required feature. What should happen?  
   **Answer:** Validate the schema and apply only a documented missing-value policy. Do not silently substitute an arbitrary value.

3. A customer has a very large nearest-centroid distance. What do you do?  
   **Answer:** Review the record, feature values and source quality. Determine whether it is a valid new behaviour or a data issue.

4. A developer refits StandardScaler on every incoming batch. What is wrong?  
   **Answer:** The feature representation changes from batch to batch. Use the scaler fitted with the historical clustering model.

5. The business wants to call cluster 2 "high-value customers". What evidence is needed?  
   **Answer:** Review original feature distributions, revenue or margin measures, retention and business definitions. Cluster ID alone does not establish value.

## 23. Lesson completion checklist

- [ ] I can define the observation unit and business question.
- [ ] I can select meaningful features and exclude identifiers.
- [ ] I can validate the feature schema.
- [ ] I can build an imputation, scaling and K-Means Pipeline.
- [ ] I can fit a segmentation using historical data.
- [ ] I can profile clusters using original feature units.
- [ ] I can assign incoming records using the fitted pipeline.
- [ ] I can inspect nearest-centroid distances.
- [ ] I can save and reload the full pipeline.
- [ ] I can document metadata and monitor assignment changes.
- [ ] I understand that clusters require interpretation and responsible business review.

**Next lesson:** Day 44 — Confusion Matrix and Classification Metrics.
