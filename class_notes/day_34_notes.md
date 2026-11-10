# Day 34: K-Means Clustering and Centroid-Based Grouping

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain K-Means clustering in plain language.
- Understand a centroid and cluster assignment.
- Describe the iterative K-Means workflow.
- Prepare numerical features for clustering.
- Scale features before distance-based clustering.
- Train KMeans using scikit-learn.
- Inspect cluster labels and centres.
- Profile clusters using original feature values.
- Understand initialization and random_state.
- Recognize K-Means assumptions and limitations.

## 2. What is K-Means?

**K-Means** is an unsupervised clustering algorithm that groups observations around a specified number of cluster centres, called centroids.

Beginner explanation:
1. Decide how many clusters to create.
2. The algorithm starts with initial cluster centres.
3. Each observation is assigned to its nearest centre.
4. The centre of each cluster is recalculated from the observations assigned to it.
5. The assignment and centre update steps repeat.
6. The process stops when the clusters stop changing significantly or a stopping limit is reached.

K-Means does not know the business names of the groups. It assigns cluster IDs such as 0, 1 and 2. The analyst profiles each group and gives it a meaningful description.

## 3. What is a centroid?

A **centroid** is the centre of a cluster, calculated from the observations assigned to that cluster.

In K-Means:
- Every observation is assigned to a cluster.
- The centroid represents the central location of that cluster in feature space.
- The algorithm updates centroids during fitting.
- New observations can be assigned to the nearest learned centroid.

A centroid is not necessarily an actual customer or record in the dataset.

## 4. Create a synthetic clustering dataset

### 4.1 Import libraries

### 4.1. Import NumPy, Pandas and plotting tools

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

These libraries prepare, inspect and visualize the clustering data.



### 4.2 Generate three groups

### 4.2. Create synthetic customer-like clusters

```python
from sklearn.datasets import make_blobs

X_values, true_groups = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.0,
    random_state=42
)

print(X_values.shape)
```

**Expected output**

```text
(300, 2)
```

**Explanation**

The synthetic dataset contains 300 observations and two numerical features. true_groups is included only to help visualize the generated sample; real unsupervised tasks usually do not have these labels.



### 4.3 Create a DataFrame

### 4.3. Assign feature names

```python
df = pd.DataFrame(
    X_values,
    columns=["spend_score", "purchase_frequency"]
)

print(df.head())
```

**Expected output**

```text
A five-row DataFrame preview
```

**Explanation**

The two columns represent simplified customer behaviour features.



### 4.4 Visualize observations without labels

### 4.4. Plot the raw observations

```python
plt.scatter(
    df["spend_score"],
    df["purchase_frequency"]
)
plt.xlabel("Spend score")
plt.ylabel("Purchase frequency")
plt.title("Customer observations before clustering")
plt.show()
```

**Expected output**

```text
A scatter plot of the observations
```

**Explanation**

The plot may show visible groups. K-Means will discover clusters using the feature values, not the hidden teaching labels.



## 5. Understand why scaling is important

K-Means uses distances between observations and centroids. A feature with a much larger numerical range can dominate those distances.

Before clustering:
- Inspect feature ranges.
- Understand feature units.
- Scale numeric features where appropriate.
- Avoid including identifiers or unrelated numeric fields.

### 5.1 Inspect feature statistics

### 5.1. Use describe()

```python
print(df.describe().T)
```

**Expected output**

```text
Summary statistics for spend_score and purchase_frequency
```

**Explanation**

Inspect ranges and distributions before fitting K-Means.



### 5.2 Create a scaler

### 5.2. Initialize StandardScaler

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

The scaler learns the training feature statistics during fit.



### 5.3 Scale the features

### 5.3. Fit and transform the clustering features

```python
X_scaled = scaler.fit_transform(df)

print(X_scaled[:5])
```

**Expected output**

```text
The first five standardized rows
```

**Explanation**

This is an unsupervised exploration example using the complete dataset. If evaluating a clustering pipeline on held-out observations, fit the scaler on the development/training partition and transform held-out data using the fitted scaler.



### 5.4 Inspect scaled feature means

### 5.4. Check transformed means

```python
print(X_scaled.mean(axis=0).round(6))
```

**Expected output**

```text
Values close to zero for each feature
```

**Explanation**

StandardScaler centers the transformed data around zero using the fitted statistics.



### 5.5 Inspect scaled feature spread

### 5.5. Check transformed standard deviations

```python
print(X_scaled.std(axis=0).round(6))
```

**Expected output**

```text
Values close to one for each feature
```

**Explanation**

The standard deviations are close to one for non-constant features under the population convention used by StandardScaler.



## 6. Train K-Means

### 6.1 Import KMeans

### 6.1. Import KMeans

```python
from sklearn.cluster import KMeans

print("KMeans imported")
```

**Expected output**

```text
KMeans imported
```

**Explanation**

KMeans is the scikit-learn estimator for centroid-based clustering.



### 6.2 Create a three-cluster model

### 6.2. Initialize KMeans

```python
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42
)

print(kmeans)
```

**Expected output**

```text
A KMeans configuration with three clusters
```

**Explanation**

n_clusters is the requested number of clusters. k-means++ selects initial centres using a strategy designed to spread them out. n_init controls how many initializations are attempted.



### 6.3 Fit K-Means

### 6.3. Train the clustering model

```python
kmeans.fit(X_scaled)

print("K-Means fitted")
```

**Expected output**

```text
K-Means fitted
```

**Explanation**

The algorithm assigns observations, updates centroids and repeats until its stopping condition is met.



### 6.4 Inspect cluster labels

### 6.4. Read assigned labels

```python
cluster_labels = kmeans.labels_

print(cluster_labels[:20])
```

**Expected output**

```text
The first twenty cluster IDs, typically 0, 1 or 2
```

**Explanation**

Each row receives a cluster identifier. The numbers are labels, not rankings.



### 6.5 Count observations per cluster

### 6.5. Count cluster sizes

```python
print(pd.Series(cluster_labels).value_counts().sort_index())
```

**Expected output**

```text
A count for each cluster ID
```

**Explanation**

Cluster sizes help identify whether groups are balanced or whether one cluster contains most observations.



### 6.6 Add cluster labels to the original DataFrame

### 6.6. Create a labelled cluster table

```python
clustered_df = df.copy()
clustered_df["cluster"] = cluster_labels

print(clustered_df.head())
```

**Expected output**

```text
A DataFrame containing original features and cluster labels
```

**Explanation**

The original feature values are retained so that clusters can be profiled in business units.



## 7. Inspect cluster centres

### 7.1 Read centroids in scaled feature space

### 7.1. Inspect cluster centres

```python
print(kmeans.cluster_centers_)
```

**Expected output**

```text
A three-row centroid array
```

**Explanation**

Each row is a cluster centre in the scaled feature space used by K-Means.



### 7.2 Convert centres back to original feature units

### 7.2. Inverse-transform centroids

```python
centres_original = scaler.inverse_transform(
    kmeans.cluster_centers_
)

print(centres_original)
```

**Expected output**

```text
Three centroid rows in original feature units
```

**Explanation**

Inverse transformation makes centroid values easier to interpret in the original spend-score and purchase-frequency units.



### 7.3 Create a centroid table

### 7.3. Attach feature names to centres

```python
centroid_df = pd.DataFrame(
    centres_original,
    columns=df.columns
)

centroid_df["cluster"] = range(kmeans.n_clusters)

print(centroid_df)
```

**Expected output**

```text
A table containing one centroid row per cluster
```

**Explanation**

This table helps explain the typical feature values represented by each cluster.



## 8. Visualize the K-Means clusters

### 8.1 Plot clustered observations

### 8.1. Create a cluster scatter plot

```python
sns.scatterplot(
    data=clustered_df,
    x="spend_score",
    y="purchase_frequency",
    hue="cluster",
    palette="tab10"
)
plt.title("K-Means customer clusters")
plt.show()
```

**Expected output**

```text
A scatter plot with observations colored by cluster ID
```

**Explanation**

The chart displays the groups found by K-Means. Cluster numbers are arbitrary and may be assigned differently in another run.



### 8.2 Plot observations and centroids

### 8.2. Show centroid positions

```python
plt.scatter(
    clustered_df["spend_score"],
    clustered_df["purchase_frequency"],
    c=clustered_df["cluster"],
    alpha=0.6
)

plt.scatter(
    centroid_df["spend_score"],
    centroid_df["purchase_frequency"],
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("Spend score")
plt.ylabel("Purchase frequency")
plt.title("K-Means observations and centroids")
plt.legend()
plt.show()
```

**Expected output**

```text
A cluster scatter plot with large X markers for centroids
```

**Explanation**

The centroids represent the centres learned by the algorithm. They are not necessarily real observations.



## 9. Profile clusters in business language

Do not label a cluster only by its number. Summarize the original features for each group.

### 9.1 Calculate cluster means

### 9.1. Create a cluster profile

```python
cluster_profile = clustered_df.groupby("cluster").agg(
    customer_count=("spend_score", "size"),
    average_spend_score=("spend_score", "mean"),
    average_purchase_frequency=("purchase_frequency", "mean")
)

print(cluster_profile)
```

**Expected output**

```text
A summary table with customer count and average feature values
```

**Explanation**

The profile describes each cluster using original units. These are descriptive summaries, not predictions.



### 9.2 Sort clusters by spending

### 9.2. Rank cluster profiles by average spend

```python
print(
    cluster_profile.sort_values(
        "average_spend_score",
        ascending=False
    )
)
```

**Expected output**

```text
Cluster profiles ordered by average spend score
```

**Explanation**

Sorting helps compare groups. It does not change the original cluster IDs or make the IDs ordinal.



### 9.3 Assign human-readable names after inspection

After reviewing the profiles, a business analyst might describe clusters as:
- Higher-spend, frequent-purchase customers.
- Moderate-spend, occasional-purchase customers.
- Lower-spend, less-frequent customers.

These names are examples only. Use actual cluster profiles and domain review before assigning labels.

## 10. Understand K-Means parameters

| Parameter | Meaning | Why it matters |
|---|---|---|
| n_clusters | Number of clusters requested | Determines how many groups are formed |
| init | Initial centroid selection method | Affects starting positions |
| n_init | Number of initializations | Reduces dependence on one start |
| max_iter | Maximum iterations for one run | Limits training duration |
| tol | Convergence tolerance | Helps determine when to stop |
| random_state | Reproducibility setting | Helps repeat initialization |

### 10.1 Compare initialization counts

### 10.1. Train with more initializations

```python
kmeans_more_init = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=20,
    random_state=42
)

kmeans_more_init.fit(X_scaled)

print(kmeans_more_init.inertia_)
```

**Expected output**

```text
An inertia value
```

**Explanation**

More initializations can reduce the chance of ending at a poor local solution, at the cost of additional computation.



### 10.2 Inspect iteration count

### 10.2. Read iteration count

```python
print(kmeans.n_iter_)
```

**Expected output**

```text
An integer iteration count
```

**Explanation**

This reports how many iterations were used before the stopping condition was reached.



### 10.3 Inspect inertia

### 10.3. Read model inertia

```python
print(kmeans.inertia_)
```

**Expected output**

```text
A non-negative inertia value
```

**Explanation**

Inertia summarizes within-cluster squared distances to centroids. It generally decreases as more clusters are allowed, so it should not be used alone to choose K.



## 11. Understand the limitations of K-Means

K-Means works best when clusters are reasonably compact and separated under the selected distance representation.

Potential difficulties:
- Clusters with very different sizes.
- Clusters with very different densities.
- Non-spherical or curved groups.
- Many outliers.
- Poor feature scaling.
- Irrelevant features.
- High-dimensional feature spaces.
- Choosing the wrong number of clusters.

K-Means always assigns observations to one of the requested clusters. It does not naturally label an observation as noise.

## 12. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Choose n_clusters without investigation | Compare candidate cluster counts |
| Include customer IDs in features | Use meaningful behavioural attributes |
| Skip scaling without checking ranges | Inspect feature units and scale when suitable |
| Interpret cluster 2 as better than cluster 1 | Cluster IDs are arbitrary labels |
| Use centroids as real customers | Centroids are representative feature locations |
| Assume a cluster label is a ground-truth class | Validate clusters using domain context |
| Ignore outliers | Investigate unusual observations and algorithm sensitivity |
| Use one initialization | Set n_init to reduce initialization sensitivity |
| Use inertia alone to choose K | Combine elbow, silhouette and business interpretation |

## 13. Interview-ready explanations

**What is K-Means?**  
It is a centroid-based clustering algorithm that assigns observations to the nearest centre and repeatedly updates cluster centres.

**What is a centroid?**  
It is the centre of a cluster, calculated from the observations assigned to that cluster.

**Why scale before K-Means?**  
K-Means uses distances, so a feature with a larger numerical range can dominate cluster assignment.

**What is inertia?**  
It summarizes how close observations are to their assigned centroids. Lower inertia means observations are closer to their centres, but inertia usually decreases as more clusters are added.

**Does K-Means identify noise points?**  
Not naturally. It assigns each observation to one of the requested clusters, so unusual observations can influence the centroids.

## 14. Scenario questions

1. K-Means forms clusters mainly based on annual spending, while purchase frequency has little effect. What do you inspect?  
   **Answer:** Feature ranges, scaling and business relevance of the selected variables.

2. The same K-Means code produces different cluster IDs in a new run. Is that necessarily a problem?  
   **Answer:** No. Cluster IDs are arbitrary, and initialization can change assignments. Compare profiles and stability rather than relying on ID numbers.

3. One cluster contains almost all customers. What do you investigate?  
   **Answer:** Feature scaling, selected K, outliers, cluster shape and whether the data naturally supports distinct groups.

4. A centroid represents a spend score of 7.2 and purchase frequency of 10. Is that a real customer?  
   **Answer:** Not necessarily. It is an average location in feature space.

5. The elbow curve continues decreasing as K increases. Should you keep increasing K?  
   **Answer:** No. Consider diminishing returns, silhouette, stability and business usefulness.

## 15. Lesson completion checklist

- [ ] I can explain K-Means and centroids.
- [ ] I understand cluster assignment and centroid updates.
- [ ] I can scale features.
- [ ] I can fit KMeans and inspect labels.
- [ ] I can inspect cluster centres and inverse-transform them.
- [ ] I can profile clusters in original feature units.
- [ ] I understand n_clusters, n_init and inertia.
- [ ] I can explain K-Means limitations.
- [ ] I can communicate clusters without treating IDs as ranked classes.

**Next lesson:** Day 35 — Elbow Method and Silhouette Score.
