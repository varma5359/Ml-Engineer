# Day 36: Hierarchical Clustering and Dendrograms

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain hierarchical clustering in simple language.
- Understand agglomerative clustering.
- Explain linkage and distance concepts.
- Read a dendrogram at a beginner level.
- Use AgglomerativeClustering in scikit-learn.
- Compare linkage strategies.
- Select a cluster count by cutting a hierarchy.
- Visualize cluster assignments.
- Understand computational and interpretation limitations.

## 2. What is hierarchical clustering?

**Hierarchical clustering** creates a hierarchy of groups by repeatedly merging or splitting clusters.

This lesson focuses on **agglomerative clustering**, a bottom-up approach:
1. Start with every observation as its own cluster.
2. Find the two closest clusters.
3. Merge them.
4. Recalculate distances between the new cluster and the remaining clusters.
5. Repeat until the requested hierarchy is built.

The process creates a tree-like structure that shows how observations or groups are related at different levels.

## 3. What is a dendrogram?

A **dendrogram** is a tree diagram that shows how clusters are merged.

How to read it:
- Each leaf at the bottom represents an observation.
- Branches show which observations or groups are merged.
- The vertical merge height represents the distance or dissimilarity at which groups are joined.
- A horizontal cut across the dendrogram can produce a selected number of clusters.

A large vertical gap between merges can suggest a useful cut, but it is not an automatic business decision.

## 4. Understand linkage

**Linkage** defines how the distance between two clusters is calculated.

| Linkage | Beginner description | Common behaviour |
|---|---|---|
| Single | Uses the closest pair of observations | Can create long chain-like clusters |
| Complete | Uses the farthest pair | Can favour compact groups |
| Average | Uses average pairwise distance | A middle-ground approach |
| Ward | Merges groups to limit increase in within-cluster variation | Often produces compact clusters; uses Euclidean distance |

The best linkage depends on data geometry and the clustering objective.

## 5. Create a practice dataset

### 5.1 Import libraries

### 5.1. Import Pandas, NumPy and plotting tools

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

These libraries support data preparation and visualization.



### 5.2 Generate observations

### 5.2. Create three synthetic groups

```python
from sklearn.datasets import make_blobs

X_values, _ = make_blobs(
    n_samples=90,
    centers=3,
    n_features=2,
    cluster_std=1.0,
    random_state=42
)

print(X_values.shape)
```

**Expected output**

```text
(90, 2)
```

**Explanation**

The data contains 90 observations and two features. The generator labels are not used by the clustering algorithm.



### 5.3 Create a DataFrame

### 5.3. Assign feature names

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

The two features are used to measure similarity between observations.



### 5.4 Plot the raw data

### 5.4. Visualize observations

```python
plt.scatter(df["feature_1"], df["feature_2"])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Observations before hierarchical clustering")
plt.show()
```

**Expected output**

```text
A scatter plot of the observations
```

**Explanation**

The chart gives an initial view of possible groups.



## 6. Scale the features

### 6.1 Inspect feature ranges

### 6.1. Review summary statistics

```python
print(df.describe().T)
```

**Expected output**

```text
Summary statistics for both features
```

**Explanation**

Distance-based clustering is sensitive to feature ranges.



### 6.2 Scale the observations

### 6.2. Apply StandardScaler

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

The example scales the full unsupervised dataset for exploration. In a held-out evaluation, fit scaling on development data and transform held-out data using the fitted scaler.



## 7. Build a dendrogram

SciPy provides functions for hierarchical linkage and dendrogram visualization.

### 7.1 Import SciPy tools

### 7.1. Import linkage and dendrogram

```python
from scipy.cluster.hierarchy import linkage, dendrogram

print("Hierarchical tools ready")
```

**Expected output**

```text
Hierarchical tools ready
```

**Explanation**

linkage builds the merge hierarchy and dendrogram visualizes it.



### 7.2 Create a Ward linkage matrix

### 7.2. Calculate Ward linkage

```python
linkage_matrix = linkage(
    X_scaled,
    method="ward"
)

print(linkage_matrix.shape)
```

**Expected output**

```text
A shape of (89, 4)
```

**Explanation**

With 90 observations, agglomerative merging creates 89 merge steps. Ward linkage uses Euclidean distance and favours compact clusters.



### 7.3 Inspect the first merge records

### 7.3. Display the linkage matrix

```python
print(linkage_matrix[:5])
```

**Expected output**

```text
The first five rows of the linkage matrix
```

**Explanation**

Each row records the two items or clusters merged, the merge distance and the number of original observations in the new cluster.



### 7.4 Plot a dendrogram

### 7.4. Create a dendrogram

```python
plt.figure(figsize=(14, 6))
dendrogram(linkage_matrix)
plt.xlabel("Observation or merged cluster")
plt.ylabel("Merge distance")
plt.title("Ward hierarchical clustering dendrogram")
plt.show()
```

**Expected output**

```text
A dendrogram showing the hierarchy of merges
```

**Explanation**

The vertical height indicates the distance at which clusters are merged. Look for large gaps as possible cut locations.



### 7.5 Plot a truncated dendrogram

### 7.5. Show only the upper levels

```python
plt.figure(figsize=(12, 6))
dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=20
)
plt.xlabel("Merged groups")
plt.ylabel("Merge distance")
plt.title("Truncated dendrogram")
plt.show()
```

**Expected output**

```text
A shorter dendrogram displaying the last major merge levels
```

**Explanation**

Truncation improves readability for larger datasets by hiding many lower-level merges.



## 8. Fit AgglomerativeClustering

### 8.1 Import the estimator

### 8.1. Import AgglomerativeClustering

```python
from sklearn.cluster import AgglomerativeClustering

print("AgglomerativeClustering imported")
```

**Expected output**

```text
AgglomerativeClustering imported
```

**Explanation**

This estimator creates hierarchical clusters and returns cluster labels for observations.



### 8.2 Create a three-cluster Ward model

### 8.2. Initialize the model

```python
hierarchical_model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

print(hierarchical_model)
```

**Expected output**

```text
AgglomerativeClustering(n_clusters=3, linkage='ward')
```

**Explanation**

n_clusters selects how many groups to return. Ward linkage merges groups based on within-cluster variation.



### 8.3 Fit and predict cluster labels

### 8.3. Fit hierarchical clustering

```python
hierarchical_labels = hierarchical_model.fit_predict(X_scaled)

print(hierarchical_labels[:20])
```

**Expected output**

```text
The first twenty cluster IDs
```

**Explanation**

The model builds the hierarchy and returns a cluster assignment for every observation.



### 8.4 Count observations per cluster

### 8.4. Inspect cluster sizes

```python
print(pd.Series(hierarchical_labels).value_counts().sort_index())
```

**Expected output**

```text
A count for each cluster ID
```

**Explanation**

Cluster sizes help identify very small or dominant groups.



### 8.5 Add labels to the original data

### 8.5. Create a labelled DataFrame

```python
clustered_df = df.copy()
clustered_df["cluster"] = hierarchical_labels

print(clustered_df.head())
```

**Expected output**

```text
A preview containing original features and cluster label
```

**Explanation**

Keep original units so that the groups can be interpreted in business terms.



### 8.6 Visualize hierarchical clusters

### 8.6. Plot cluster assignments

```python
sns.scatterplot(
    data=clustered_df,
    x="feature_1",
    y="feature_2",
    hue="cluster",
    palette="tab10"
)
plt.title("Agglomerative clustering results")
plt.show()
```

**Expected output**

```text
A scatter plot with three cluster colors
```

**Explanation**

The visualization helps inspect how the selected cut separates the observations.



## 9. Compare linkage strategies

### 9.1 Fit single linkage

### 9.1. Create a single-linkage model

```python
single_model = AgglomerativeClustering(
    n_clusters=3,
    linkage="single"
)

single_labels = single_model.fit_predict(X_scaled)

print(pd.Series(single_labels).value_counts())
```

**Expected output**

```text
A count of observations in each single-linkage cluster
```

**Explanation**

Single linkage uses the closest pair of observations to determine cluster distance and can create chain-like groups.



### 9.2 Fit complete linkage

### 9.2. Create a complete-linkage model

```python
complete_model = AgglomerativeClustering(
    n_clusters=3,
    linkage="complete"
)

complete_labels = complete_model.fit_predict(X_scaled)

print(pd.Series(complete_labels).value_counts())
```

**Expected output**

```text
A count of observations in each complete-linkage cluster
```

**Explanation**

Complete linkage uses the farthest pair of observations between clusters and often favours compact groups.



### 9.3 Fit average linkage

### 9.3. Create an average-linkage model

```python
average_model = AgglomerativeClustering(
    n_clusters=3,
    linkage="average"
)

average_labels = average_model.fit_predict(X_scaled)

print(pd.Series(average_labels).value_counts())
```

**Expected output**

```text
A count of observations in each average-linkage cluster
```

**Explanation**

Average linkage uses average pairwise distances between clusters.



### 9.4 Compare silhouette scores

### 9.4. Evaluate linkage choices

```python
from sklearn.metrics import silhouette_score

for name, labels in [
    ("Ward", hierarchical_labels),
    ("Single", single_labels),
    ("Complete", complete_labels),
    ("Average", average_labels)
]:
    print(name, silhouette_score(X_scaled, labels))
```

**Expected output**

```text
A silhouette score for each linkage method
```

**Explanation**

This compares geometric separation for the same observations and requested cluster count. Also inspect cluster profiles and stability.



## 10. Understand the number of clusters

### 10.1 Fit two clusters

### 10.1. Create a two-cluster model

```python
two_cluster_model = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

labels_2 = two_cluster_model.fit_predict(X_scaled)

print(pd.Series(labels_2).value_counts())
```

**Expected output**

```text
A count for each of two clusters
```

**Explanation**

The hierarchy can be cut at different levels to create different numbers of groups.



### 10.2 Fit four clusters

### 10.2. Create a four-cluster model

```python
four_cluster_model = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

labels_4 = four_cluster_model.fit_predict(X_scaled)

print(pd.Series(labels_4).value_counts())
```

**Expected output**

```text
A count for each of four clusters
```

**Explanation**

More clusters create a more detailed segmentation but may also split natural groups unnecessarily.



### 10.3 Compare silhouette for several cluster counts

### 10.3. Evaluate cluster counts

```python
cluster_count_results = []

for k in [2, 3, 4, 5, 6]:
    candidate = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = candidate.fit_predict(X_scaled)

    cluster_count_results.append({
        "clusters": k,
        "silhouette": silhouette_score(X_scaled, labels)
    })

print(pd.DataFrame(cluster_count_results))
```

**Expected output**

```text
A table comparing silhouette across cluster counts
```

**Explanation**

Use this as one diagnostic. Cluster count should also reflect interpretability, stability and business needs.



## 11. Profile hierarchical clusters

### 11.1 Add the chosen labels

### 11.1. Use three-cluster labels

```python
profile_df = df.copy()
profile_df["cluster"] = hierarchical_labels

print(profile_df.head())
```

**Expected output**

```text
A preview with original features and cluster labels
```

**Explanation**

The cluster ID is added to the original feature values for interpretation.



### 11.2 Calculate cluster means

### 11.2. Create cluster profiles

```python
profiles = profile_df.groupby("cluster").agg(
    observation_count=("feature_1", "size"),
    feature_1_mean=("feature_1", "mean"),
    feature_2_mean=("feature_2", "mean")
)

print(profiles)
```

**Expected output**

```text
A cluster profile table
```

**Explanation**

The means describe the typical location of each group in the original feature units.



### 11.3 Compare profiles

### 11.3. Sort by feature one

```python
print(profiles.sort_values("feature_1_mean"))
```

**Expected output**

```text
Cluster profiles ordered by feature one
```

**Explanation**

This supports interpretation of the groups. The IDs remain arbitrary labels.



## 12. Hierarchical clustering versus K-Means

| Topic | K-Means | Agglomerative hierarchical |
|---|---|---|
| Cluster count | Must be specified before fitting | Can inspect hierarchy and select a cut |
| Cluster representation | Centroids | Nested merge structure |
| Assignment | Nearest centroid | Built from pairwise cluster distances |
| Scaling | Important for distance-based use | Important for distance-based use |
| Computation | Often efficient on larger tabular data | Can be expensive as dataset grows |
| New observations | Predict available in KMeans | Standard agglomerative estimator does not provide the same simple predict workflow |
| Interpretation | Centroid profiles | Dendrogram and cluster profiles |

## 13. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Read dendrogram height as a probability | It represents a merge distance |
| Assume a dendrogram gives one obvious K | Compare cuts and validate clusters |
| Ignore scaling | Inspect feature units before distance calculations |
| Use Ward with an incompatible distance | Ward requires Euclidean distance |
| Assume labels have an order | Cluster IDs are arbitrary |
| Ignore computational cost | Hierarchical methods can be expensive for large datasets |
| Select K from silhouette alone | Include cluster profiles and domain validation |

## 14. Interview-ready explanations

**What is hierarchical clustering?**  
It builds a hierarchy of groups by repeatedly merging or splitting clusters. Agglomerative clustering starts with individual observations and merges them.

**What is a dendrogram?**  
A tree diagram showing the order and distance at which observations or groups are merged.

**What is linkage?**  
It defines how the distance between clusters is calculated, such as single, complete, average or Ward linkage.

**What is the difference between K-Means and hierarchical clustering?**  
K-Means uses centroids and requires K in advance. Hierarchical clustering builds a merge hierarchy that can be inspected at different levels.

## 15. Scenario questions

1. The dendrogram has a large vertical gap before a final merge. What might you investigate?  
   **Answer:** A cut below that merge may produce well-separated groups, but confirm with cluster metrics and domain interpretation.

2. Single linkage creates one long chain-like cluster. Why might this happen?  
   **Answer:** It uses the closest pairwise points and can connect groups through chains of nearby observations.

3. Hierarchical clustering is too slow on millions of records. What do you consider?  
   **Answer:** Sample-based exploration, more scalable clustering methods or a different algorithm suited to dataset size.

4. Ward linkage is requested with a non-Euclidean metric. What is the issue?  
   **Answer:** Ward's merge criterion is based on Euclidean geometry; choose a compatible method or distance strategy.

5. Two cluster counts have similar silhouette scores. How do you choose?  
   **Answer:** Compare profile clarity, stability, group size and business actionability.

## 16. Lesson completion checklist

- [ ] I can explain agglomerative clustering.
- [ ] I can read a dendrogram at a basic level.
- [ ] I understand single, complete, average and Ward linkage.
- [ ] I can train AgglomerativeClustering.
- [ ] I can compare cluster counts and linkage strategies.
- [ ] I can profile hierarchical clusters.
- [ ] I understand computational and interpretation limitations.

**Next lesson:** Day 37 — DBSCAN and Density-Based Clustering.
