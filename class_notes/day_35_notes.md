# Day 35: Elbow Method and Silhouette Score

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain why K-Means needs a cluster count.
- Understand inertia and the elbow method.
- Use silhouette score to evaluate cluster separation.
- Compare several candidate K values.
- Visualize elbow and silhouette results.
- Understand why no single score determines the best segmentation.
- Profile clusters and validate their business meaning.
- Avoid selecting K using only one chart or metric.

## 2. Why do we need to choose K?

K-Means requires the number of clusters before fitting. The algorithm does not automatically know how many customer groups a business should have.

If K is too small:
- Different customer behaviours may be combined.
- Important segments may be hidden.

If K is too large:
- Similar customers may be split into unnecessary groups.
- The result may become difficult to explain and act on.

The elbow method and silhouette score are two common tools for comparing candidate cluster counts.

## 3. Understand inertia

**Inertia** summarizes how close observations are to their assigned centroids. It measures within-cluster compactness.

General behaviour:
- Increasing K usually reduces inertia.
- A very high K can make clusters small and tightly fitted.
- The goal is not simply to minimize inertia.
- Look for a point where adding more clusters provides diminishing improvement.

## 4. Understand the elbow method

The **elbow method** fits K-Means for several candidate cluster counts and plots inertia against K.

The curve often decreases quickly at first, then the improvement becomes smaller. The bend or "elbow" is a possible candidate for K.

The elbow may be unclear. It is a visual aid, not an exact mathematical answer.

## 5. Understand silhouette score

The **silhouette score** compares how close an observation is to its own cluster relative to other clusters.

Plain-language interpretation:
- A value closer to +1 generally indicates an observation fits its assigned cluster well and is separated from other clusters.
- A value around zero suggests the observation is near a cluster boundary.
- A negative value suggests it may be closer to another cluster.

The overall silhouette score summarizes observations. Inspect per-cluster behaviour too.

## 6. Create a clustering dataset

### 6.1 Import libraries

### 6.1. Import libraries

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

These libraries support data preparation, clustering and visualization.



### 6.2 Generate observations with several groups

### 6.2. Create a synthetic dataset

```python
from sklearn.datasets import make_blobs

X_values, _ = make_blobs(
    n_samples=500,
    centers=4,
    n_features=2,
    cluster_std=[0.7, 1.0, 1.2, 0.8],
    random_state=42
)

print(X_values.shape)
```

**Expected output**

```text
(500, 2)
```

**Explanation**

The sample is generated around four centres. The hidden generator labels are intentionally not used for model fitting.



### 6.3 Create a DataFrame

### 6.3. Assign feature names

```python
df = pd.DataFrame(
    X_values,
    columns=["feature_1", "feature_2"]
)

print(df.head())
```

**Expected output**

```text
A five-row feature preview
```

**Explanation**

The clustering algorithm will use these two features.



### 6.4 Visualize raw observations

### 6.4. Plot the raw data

```python
plt.scatter(df["feature_1"], df["feature_2"])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Observations before clustering")
plt.show()
```

**Expected output**

```text
A scatter plot of the observations
```

**Explanation**

The chart provides context, but cluster count should not be chosen by visual guess alone.



## 7. Scale the features

### 7.1 Import StandardScaler

### 7.1. Import the scaler

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

K-Means uses distances, so feature scale can affect cluster assignment.



### 7.2 Fit and transform features

### 7.2. Scale the dataset

```python
X_scaled = scaler.fit_transform(df)

print(X_scaled[:5])
```

**Expected output**

```text
The first five standardized observations
```

**Explanation**

This is an exploratory unsupervised dataset. In a held-out evaluation workflow, fit the scaler on development data and transform held-out data using that fitted scaler.



## 8. Calculate inertia for different K values

### 8.1 Import KMeans

### 8.1. Import KMeans

```python
from sklearn.cluster import KMeans

print("KMeans imported")
```

**Expected output**

```text
KMeans imported
```

**Explanation**

We will fit one model for each candidate cluster count.



### 8.2 Create a list of candidate K values

### 8.2. Define K values

```python
k_values = range(1, 11)

print(list(k_values))
```

**Expected output**

```text
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

**Explanation**

Testing a range helps compare how within-cluster compactness changes as more clusters are added.



### 8.3 Fit K-Means for each K

### 8.3. Collect inertia values

```python
inertia_values = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    model.fit(X_scaled)
    inertia_values.append(model.inertia_)

print(inertia_values)
```

**Expected output**

```text
A list of inertia values, generally decreasing as K increases
```

**Explanation**

Each candidate is fitted on the same scaled observations. n_init reduces sensitivity to one random initialization.



### 8.4 Create an elbow table

### 8.4. Store K and inertia

```python
elbow_table = pd.DataFrame({
    "k": list(k_values),
    "inertia": inertia_values
})

print(elbow_table)
```

**Expected output**

```text
A table containing K and inertia
```

**Explanation**

The table allows numerical comparison as well as plotting.



### 8.5 Plot the elbow curve

### 8.5. Visualize inertia by K

```python
plt.plot(
    elbow_table["k"],
    elbow_table["inertia"],
    marker="o"
)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.show()
```

**Expected output**

```text
A decreasing inertia curve
```

**Explanation**

Look for a bend where additional clusters provide less improvement. The elbow may not be sharp or unique.



## 9. Calculate silhouette scores

### 9.1 Import silhouette_score

### 9.1. Import the metric

```python
from sklearn.metrics import silhouette_score

print("Silhouette metric imported")
```

**Expected output**

```text
Silhouette metric imported
```

**Explanation**

Silhouette score requires at least two clusters and fewer clusters than observations.



### 9.2 Calculate scores for candidate K values

### 9.2. Evaluate silhouette for K from two to ten

```python
silhouette_values = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    silhouette_values.append({
        "k": k,
        "silhouette": score
    })

silhouette_table = pd.DataFrame(silhouette_values)

print(silhouette_table)
```

**Expected output**

```text
A table of silhouette scores for K values from 2 to 10
```

**Explanation**

K=1 is excluded because silhouette requires multiple clusters. A higher score generally indicates better separation and cohesion under the metric.



### 9.3 Plot silhouette scores

### 9.3. Visualize silhouette by K

```python
plt.plot(
    silhouette_table["k"],
    silhouette_table["silhouette"],
    marker="o"
)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score by cluster count")
plt.show()
```

**Expected output**

```text
A line chart of silhouette scores
```

**Explanation**

The score can help compare candidate K values. It does not measure business usefulness directly.



### 9.4 Find the highest silhouette score

### 9.4. Read the highest-scoring K

```python
best_silhouette_row = silhouette_table.loc[
    silhouette_table["silhouette"].idxmax()
]

print(best_silhouette_row)
```

**Expected output**

```text
The K value and score with the highest silhouette
```

**Explanation**

This is one candidate, not an automatic final decision. Compare the result with the elbow curve and cluster profiles.



## 10. Compare elbow and silhouette evidence

The two methods answer different questions:
- Inertia describes within-cluster compactness.
- Silhouette considers both within-cluster cohesion and separation from other clusters.

A useful selection process:
1. Inspect the elbow curve.
2. Inspect silhouette scores.
3. Fit candidate cluster counts.
4. Compare cluster sizes and profiles.
5. Check stability under different samples or settings.
6. Discuss whether the groups support meaningful business actions.

The elbow and silhouette methods may suggest different K values. That is a reason to investigate, not a reason to select one blindly.

## 11. Fit a selected candidate K

### 11.1 Select a candidate cluster count

### 11.1. Set a candidate K after reviewing evidence

```python
selected_k = 4

print(selected_k)
```

**Expected output**

```text
4
```

**Explanation**

This value is selected for the teaching dataset because it was generated with four groups. In a real project, the choice should be justified using several forms of evidence.



### 11.2 Fit the final K-Means model

### 11.2. Train the selected clustering model

```python
final_kmeans = KMeans(
    n_clusters=selected_k,
    n_init=20,
    random_state=42
)

final_labels = final_kmeans.fit_predict(X_scaled)

print(pd.Series(final_labels).value_counts().sort_index())
```

**Expected output**

```text
A count for each of the four clusters
```

**Explanation**

The model assigns every observation to one of the selected clusters.



### 11.3 Add cluster labels to the original table

### 11.3. Create a labelled DataFrame

```python
clustered_df = df.copy()
clustered_df["cluster"] = final_labels

print(clustered_df.head())
```

**Expected output**

```text
A preview with original features and cluster label
```

**Explanation**

Keep original feature values for cluster profiling and interpretation.



### 11.4 Plot selected clusters

### 11.4. Visualize selected clusters

```python
sns.scatterplot(
    data=clustered_df,
    x="feature_1",
    y="feature_2",
    hue="cluster",
    palette="tab10"
)
plt.title("Selected K-Means clusters")
plt.show()
```

**Expected output**

```text
A scatter plot with four cluster colors
```

**Explanation**

The plot provides a visual check of the selected grouping.



## 12. Inspect cluster sizes and profiles

### 12.1 Count observations per cluster

### 12.1. Calculate cluster sizes

```python
cluster_sizes = clustered_df["cluster"].value_counts().sort_index()

print(cluster_sizes)
```

**Expected output**

```text
A count for each cluster
```

**Explanation**

Very small clusters may need investigation. They could represent rare groups, noise or an unsuitable K.



### 12.2 Calculate average feature values

### 12.2. Create cluster profiles

```python
cluster_profile = clustered_df.groupby("cluster").agg(
    customer_count=("feature_1", "size"),
    feature_1_mean=("feature_1", "mean"),
    feature_2_mean=("feature_2", "mean")
)

print(cluster_profile)
```

**Expected output**

```text
A profile table containing cluster size and feature averages
```

**Explanation**

Cluster profiles explain what each group looks like in original feature units.



### 12.3 Compare cluster profiles

### 12.3. Sort profiles by feature one

```python
print(
    cluster_profile.sort_values(
        "feature_1_mean",
        ascending=False
    )
)
```

**Expected output**

```text
Profiles ordered by average feature one
```

**Explanation**

Sorting helps compare group characteristics. The cluster ID itself has no ranking meaning.



## 13. Understand silhouette limitations

- Silhouette is based on distances and may favour compact, well-separated clusters.
- It may not suit clusters with complex shapes or different densities.
- It can be expensive on very large datasets.
- It does not measure business usefulness.
- A high overall score can hide a weak individual cluster.

Always combine the score with cluster profiles, visual inspection and domain knowledge.

## 14. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Choose K only from the elbow chart | Combine elbow, silhouette and domain validation |
| Choose the highest silhouette without profiling | Inspect cluster sizes and business meaning |
| Include K=1 in silhouette calculation | Start at K=2 |
| Compare models using differently scaled data | Use the same preprocessing |
| Ignore tiny clusters | Investigate their size and profile |
| Assume inertia should reach zero | Look for diminishing returns |
| Treat cluster labels as ordered classes | Cluster IDs are arbitrary |
| Use the final test set to tune clustering settings | Use development data and reserve held-out evaluation where appropriate |

## 15. Interview-ready explanations

**What is the elbow method?**  
It compares K-Means inertia across different cluster counts and looks for a bend where additional clusters provide diminishing improvement.

**What is silhouette score?**  
It measures how well observations fit their own clusters compared with other clusters. Higher values generally indicate better separation and cohesion.

**Can elbow and silhouette recommend different K values?**  
Yes. They measure different aspects of clustering. Use cluster profiles, stability and business meaning to make the final decision.

**Does a high silhouette score prove that clusters are useful to the business?**  
No. It measures geometric structure, not commercial value or actionability.

## 16. Scenario questions

1. Inertia decreases continuously as K increases. What do you do?  
   **Answer:** Look for diminishing returns, silhouette scores, cluster profiles and business usefulness rather than minimizing inertia alone.

2. Silhouette is highest for K=2, but the business needs more detailed segments. What should you consider?  
   **Answer:** Compare the stability and usefulness of additional clusters and explain the trade-off. Do not ignore business needs, but do not invent segments unsupported by data.

3. One cluster contains only three records. What do you investigate?  
   **Answer:** Check whether they are valid rare observations, outliers or evidence that K is too high.

4. Two K values have similar silhouette scores. How do you choose?  
   **Answer:** Compare profile clarity, stability, cluster size and whether each segmentation supports a useful action.

5. A high silhouette score is achieved after including customer ID. What is the concern?  
   **Answer:** The score may reflect meaningless identifier structure. Rebuild using valid behavioural features.

## 17. Lesson completion checklist

- [ ] I can explain inertia.
- [ ] I can create an elbow curve.
- [ ] I can calculate silhouette scores.
- [ ] I can compare candidate K values.
- [ ] I can inspect cluster sizes and profiles.
- [ ] I understand the limitations of both metrics.
- [ ] I can explain why business interpretation is necessary.

**Next lesson:** Day 36 — Hierarchical Clustering and Dendrograms.
