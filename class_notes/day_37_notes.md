# Day 37: DBSCAN and Density-Based Clustering

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain density-based clustering in simple language.
- Understand DBSCAN's eps and min_samples parameters.
- Distinguish core, border and noise observations.
- Train DBSCAN using scikit-learn.
- Scale features before distance-based clustering.
- Interpret the noise label.
- Compare DBSCAN with K-Means.
- Visualize non-spherical clusters.
- Recognize how eps and min_samples affect results.
- Understand DBSCAN limitations.

## 2. What is DBSCAN?

**DBSCAN** stands for Density-Based Spatial Clustering of Applications with Noise.

It groups observations that are located in dense regions and can mark observations in sparse regions as noise.

Unlike K-Means:
- DBSCAN does not require the number of clusters in advance.
- It can identify noise observations.
- It can find clusters with non-spherical shapes.
- It depends on a distance radius and a minimum number of nearby observations.

DBSCAN is useful when clusters are separated by lower-density regions and when noise detection is important.

## 3. Understand DBSCAN vocabulary

| Term | Meaning |
|---|---|
| eps | Neighbourhood radius around an observation |
| min_samples | Minimum number of observations needed for a dense neighbourhood |
| Core point | Observation with enough neighbours within eps |
| Border point | Observation close to a core point but without enough neighbours of its own |
| Noise point | Observation not assigned to a dense cluster |
| Cluster | Connected group of density-reachable observations |

## 4. Create a dataset with curved clusters

### 4.1 Import libraries

### 4.1. Import Pandas, NumPy and plotting tools

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

These libraries prepare and visualize the clustering observations.



### 4.2 Generate non-spherical groups

### 4.2. Create a two-moons dataset

```python
from sklearn.datasets import make_moons

X_values, hidden_labels = make_moons(
    n_samples=400,
    noise=0.06,
    random_state=42
)

print(X_values.shape)
```

**Expected output**

```text
(400, 2)
```

**Explanation**

The two-moons dataset contains curved groups. hidden_labels are used only for teaching visualization, not for DBSCAN fitting.



### 4.3 Create a DataFrame

### 4.3. Name the features

```python
df = pd.DataFrame(
    X_values,
    columns=["feature_1", "feature_2"]
)

print(df.head())
```

**Expected output**

```text
A five-row preview
```

**Explanation**

The two features describe each observation in two-dimensional space.



### 4.4 Plot raw observations

### 4.4. Visualize the data

```python
plt.scatter(df["feature_1"], df["feature_2"])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Two-moons observations")
plt.show()
```

**Expected output**

```text
A scatter plot showing two curved shapes
```

**Explanation**

This is a useful example because centroid-based methods may struggle with non-spherical groups.



## 5. Scale the data

DBSCAN relies on neighbourhood distances. Feature scale can strongly affect which points fall within eps.

### 5.1 Inspect feature ranges

### 5.1. Review feature summary

```python
print(df.describe().T)
```

**Expected output**

```text
Summary statistics for both features
```

**Explanation**

Check units and ranges before choosing a distance radius.



### 5.2 Apply StandardScaler

### 5.2. Scale the observations

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print(X_scaled[:5])
```

**Expected output**

```text
The first five standardized observations
```

**Explanation**

This example scales the complete unsupervised dataset for exploration. For held-out evaluation, fit scaling on development data and transform held-out data with the fitted scaler.



## 6. Train DBSCAN

### 6.1 Import DBSCAN

### 6.1. Import the estimator

```python
from sklearn.cluster import DBSCAN

print("DBSCAN imported")
```

**Expected output**

```text
DBSCAN imported
```

**Explanation**

DBSCAN discovers dense groups using the eps and min_samples settings.



### 6.2 Initialize DBSCAN

### 6.2. Create a DBSCAN model

```python
dbscan = DBSCAN(
    eps=0.3,
    min_samples=5
)

print(dbscan)
```

**Expected output**

```text
DBSCAN(eps=0.3, min_samples=5)
```

**Explanation**

eps sets the neighbourhood radius and min_samples controls how many observations are required to form a dense region. Values depend on the scaled data.



### 6.3 Fit and assign cluster labels

### 6.3. Run DBSCAN

```python
dbscan_labels = dbscan.fit_predict(X_scaled)

print(dbscan_labels[:20])
```

**Expected output**

```text
Cluster IDs and possibly -1 noise labels
```

**Explanation**

DBSCAN returns an integer label for each observation. The label -1 is reserved for noise points.



### 6.4 Count cluster labels

### 6.4. Count clusters and noise

```python
label_counts = pd.Series(dbscan_labels).value_counts().sort_index()

print(label_counts)
```

**Expected output**

```text
A count for each cluster ID and -1 if noise exists
```

**Explanation**

The number of distinct non-negative labels indicates how many clusters were found. The -1 count indicates observations classified as noise.



### 6.5 Count clusters excluding noise

### 6.5. Count discovered clusters

```python
cluster_ids = set(dbscan_labels) - {-1}

print(len(cluster_ids))
```

**Expected output**

```text
The number of non-noise clusters
```

**Explanation**

Subtracting -1 excludes the noise label from the cluster count.



## 7. Visualize DBSCAN results

### 7.1 Add labels to the original features

### 7.1. Create a labelled DataFrame

```python
clustered_df = df.copy()
clustered_df["cluster"] = dbscan_labels

print(clustered_df.head())
```

**Expected output**

```text
A preview containing the original features and DBSCAN cluster label
```

**Explanation**

The original units are preserved for visualization and interpretation.



### 7.2 Plot cluster assignments

### 7.2. Visualize clusters and noise

```python
sns.scatterplot(
    data=clustered_df,
    x="feature_1",
    y="feature_2",
    hue="cluster",
    palette="tab10"
)
plt.title("DBSCAN clusters and noise")
plt.show()
```

**Expected output**

```text
A scatter plot showing the discovered groups and any noise points
```

**Explanation**

The plot helps determine whether the selected eps and min_samples values recover the expected density structure.



### 7.3 Highlight noise observations

### 7.3. Filter noise points

```python
noise_points = clustered_df[
    clustered_df["cluster"] == -1
]

print(noise_points.head())
print("Noise count:", len(noise_points))
```

**Expected output**

```text
A preview of noise observations and their count
```

**Explanation**

Noise points are not automatically errors. They may be rare valid observations or unusual data requiring investigation.



## 8. Understand eps

eps defines the radius around a point used to identify its neighbourhood.

- Too small: many observations may be labelled noise, and clusters may fragment.
- Too large: separate groups may merge.
- Suitable: dense regions are connected while sparse gaps remain.

### 8.1 Try a smaller eps

### 8.1. Fit DBSCAN with smaller eps

```python
small_eps_model = DBSCAN(
    eps=0.15,
    min_samples=5
)

small_eps_labels = small_eps_model.fit_predict(X_scaled)

print(pd.Series(small_eps_labels).value_counts().sort_index())
```

**Expected output**

```text
Counts for clusters and noise under the smaller radius
```

**Explanation**

A smaller neighbourhood can make it harder for observations to qualify as part of a dense group.



### 8.2 Try a larger eps

### 8.2. Fit DBSCAN with larger eps

```python
large_eps_model = DBSCAN(
    eps=0.6,
    min_samples=5
)

large_eps_labels = large_eps_model.fit_predict(X_scaled)

print(pd.Series(large_eps_labels).value_counts().sort_index())
```

**Expected output**

```text
Counts for clusters and noise under the larger radius
```

**Explanation**

A larger neighbourhood can connect groups that were previously separate.



### 8.3 Compare eps values

### 8.3. Summarize cluster counts and noise

```python
eps_results = []

for eps in [0.15, 0.2, 0.3, 0.4, 0.6]:
    candidate = DBSCAN(
        eps=eps,
        min_samples=5
    )

    labels = candidate.fit_predict(X_scaled)

    eps_results.append({
        "eps": eps,
        "clusters": len(set(labels) - {-1}),
        "noise_points": int(np.sum(labels == -1))
    })

print(pd.DataFrame(eps_results))
```

**Expected output**

```text
A table comparing eps, number of clusters and noise count
```

**Explanation**

This makes the effect of neighbourhood radius easier to inspect. Do not select eps only to obtain a preferred number of clusters.



## 9. Understand min_samples

min_samples determines how many observations must be in an eps neighbourhood for a point to qualify as a core point.

- Smaller min_samples can create more clusters and accept less dense regions.
- Larger min_samples requires stronger density.
- It can increase the number of noise observations.
- It should be chosen with data size, dimensionality and noise level in mind.

### 9.1 Try smaller min_samples

### 9.1. Fit DBSCAN with min_samples 3

```python
few_neighbors_model = DBSCAN(
    eps=0.3,
    min_samples=3
)

few_neighbor_labels = few_neighbors_model.fit_predict(X_scaled)

print(pd.Series(few_neighbor_labels).value_counts().sort_index())
```

**Expected output**

```text
Counts for clusters and noise with min_samples 3
```

**Explanation**

A lower minimum density requirement can allow more observations to become part of clusters.



### 9.2 Try larger min_samples

### 9.2. Fit DBSCAN with min_samples 10

```python
many_neighbors_model = DBSCAN(
    eps=0.3,
    min_samples=10
)

many_neighbor_labels = many_neighbors_model.fit_predict(X_scaled)

print(pd.Series(many_neighbor_labels).value_counts().sort_index())
```

**Expected output**

```text
Counts for clusters and noise with min_samples 10
```

**Explanation**

A higher density requirement may mark more observations as noise.



### 9.3 Compare min_samples choices

### 9.3. Create a min_samples comparison

```python
min_samples_results = []

for min_samples in [3, 5, 8, 10, 15]:
    candidate = DBSCAN(
        eps=0.3,
        min_samples=min_samples
    )

    labels = candidate.fit_predict(X_scaled)

    min_samples_results.append({
        "min_samples": min_samples,
        "clusters": len(set(labels) - {-1}),
        "noise": int(np.sum(labels == -1))
    })

print(pd.DataFrame(min_samples_results))
```

**Expected output**

```text
A table showing how cluster count and noise change
```

**Explanation**

This helps understand sensitivity to the density requirement.



## 10. Evaluate DBSCAN clusters

Silhouette score can be used when at least two non-noise clusters exist. A simple approach excludes noise points before calculating the score, but this means the score does not describe the noise group.

### 10.1 Identify non-noise observations

### 10.1. Create a mask for clustered observations

```python
non_noise_mask = dbscan_labels != -1

print(non_noise_mask.sum())
```

**Expected output**

```text
The number of observations assigned to a cluster
```

**Explanation**

Noise points are excluded for this particular silhouette calculation.



### 10.2 Check the number of clusters

### 10.2. Count non-noise clusters

```python
non_noise_clusters = set(dbscan_labels[non_noise_mask])

print(non_noise_clusters)
```

**Expected output**

```text
A set of non-negative cluster IDs
```

**Explanation**

Silhouette requires at least two clusters and more than one observation overall.



### 10.3 Calculate silhouette only when valid

### 10.3. Calculate silhouette safely

```python
from sklearn.metrics import silhouette_score

if len(non_noise_clusters) >= 2:
    score = silhouette_score(
        X_scaled[non_noise_mask],
        dbscan_labels[non_noise_mask]
    )
    print(score)
else:
    print("Silhouette is not defined for this result")
```

**Expected output**

```text
A silhouette score or an explanatory message
```

**Explanation**

The metric is calculated only when at least two non-noise clusters exist. Always report that noise points were excluded.



## 11. Compare DBSCAN with K-Means

K-Means:
- Requires K in advance.
- Assigns every observation to a cluster.
- Uses centroids and tends to favour compact groups.

DBSCAN:
- Does not require K in advance.
- Can label observations as noise.
- Can find non-spherical dense groups.
- Depends on eps and min_samples.
- Can struggle when clusters have very different densities.

### 11.1 Fit K-Means on the same scaled data

### 11.1. Create a two-cluster K-Means model

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=2,
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(X_scaled)

print(pd.Series(kmeans_labels).value_counts())
```

**Expected output**

```text
Counts for two K-Means clusters
```

**Explanation**

K-Means assigns every observation to one of the requested groups.



### 11.2 Visualize K-Means assignments

### 11.2. Plot K-Means clusters

```python
kmeans_view = df.copy()
kmeans_view["cluster"] = kmeans_labels

sns.scatterplot(
    data=kmeans_view,
    x="feature_1",
    y="feature_2",
    hue="cluster"
)
plt.title("K-Means result on two-moons data")
plt.show()
```

**Expected output**

```text
A scatter plot of K-Means assignments
```

**Explanation**

K-Means may struggle with curved non-spherical groups because it uses centroid-based partitions.



### 11.3 Compare DBSCAN assignments

### 11.3. Plot DBSCAN result again

```python
dbscan_view = df.copy()
dbscan_view["cluster"] = dbscan_labels

sns.scatterplot(
    data=dbscan_view,
    x="feature_1",
    y="feature_2",
    hue="cluster"
)
plt.title("DBSCAN result on two-moons data")
plt.show()
```

**Expected output**

```text
A scatter plot of DBSCAN assignments
```

**Explanation**

DBSCAN can identify curved dense groups when eps and min_samples are appropriate.



## 12. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Choose eps without considering scale | Scale features and inspect neighbourhood distances |
| Set min_samples too low | Compare cluster stability and noise |
| Treat -1 as a normal cluster | Interpret -1 as noise |
| Assume every noise point is an error | Investigate source and business meaning |
| Use DBSCAN when densities vary greatly | Consider methods designed for variable-density clusters |
| Compare silhouette including noise without explanation | State how noise was handled |
| Assume DBSCAN always finds the true number of groups | Validate with domain knowledge and stability |

## 13. Interview-ready explanations

**What is DBSCAN?**  
It is a density-based clustering algorithm that groups dense regions and can mark sparse observations as noise.

**What is eps?**  
It is the radius used to define the neighbourhood around an observation.

**What is min_samples?**  
It is the minimum number of observations required in a neighbourhood for a point to qualify as a core point.

**How is DBSCAN different from K-Means?**  
DBSCAN does not require the number of clusters in advance, can find non-spherical dense groups and can label noise. K-Means uses a specified number of centroids and assigns every observation to a cluster.

**What does cluster label -1 mean?**  
It identifies observations that DBSCAN did not assign to a dense cluster. They may be noise or unusual observations requiring investigation.

## 14. Scenario questions

1. DBSCAN labels most observations as noise. What do you inspect?  
   **Answer:** Feature scaling, eps, min_samples, data density and whether the features are appropriate.

2. Two distinct groups merge when eps increases. What does that indicate?  
   **Answer:** The neighbourhood radius may be too large for the gap between those groups.

3. K-Means struggles with curved clusters. What could you try?  
   **Answer:** DBSCAN or another method that can represent non-spherical cluster structures.

4. A noise point is a high-value customer. Should it be removed?  
   **Answer:** No. It may be a valid rare customer segment. Investigate and decide how the business should handle it.

5. One cluster is very dense and another is sparse. What is a DBSCAN challenge?  
   **Answer:** A single eps and min_samples setting may not represent both densities well.

## 15. Lesson completion checklist

- [ ] I can explain density-based clustering.
- [ ] I understand core, border and noise points.
- [ ] I can use eps and min_samples.
- [ ] I can fit DBSCAN and inspect labels.
- [ ] I can compare parameter settings.
- [ ] I can calculate silhouette carefully.
- [ ] I can compare DBSCAN with K-Means.
- [ ] I understand DBSCAN limitations.

**Next lesson:** Day 38 — Principal Component Analysis and Dimensionality Reduction.
