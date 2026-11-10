# Day 42: Clustering Evaluation and Limitations

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain why clustering evaluation differs from supervised evaluation.
- Understand internal and external clustering metrics.
- Calculate silhouette score, Davies-Bouldin and Calinski-Harabasz scores.
- Evaluate clusters while handling DBSCAN noise carefully.
- Compare candidate cluster counts.
- Assess clustering stability across runs.
- Use known labels for external evaluation when available.
- Identify metric limitations and business interpretation risks.
- Write a concise clustering quality report.

## 2. Why is clustering evaluation difficult?

In supervised learning, the dataset often contains known target labels. We can compare predictions with the correct answers.

In unsupervised clustering, the algorithm discovers groups without being given the correct group for each observation.

This creates a challenge:
- There may be no ground-truth cluster labels.
- Different algorithms can discover different but plausible structures.
- A metric can favour compact geometric clusters that are not useful to the business.
- Cluster IDs are arbitrary.
- The best cluster count may depend on the intended decision.

A good clustering evaluation combines numerical metrics, stability checks, cluster profiles and domain review.

## 3. Types of clustering evaluation

### Internal evaluation
Uses input features and discovered cluster labels. Examples include silhouette score, Davies-Bouldin index, Calinski-Harabasz score and within-cluster compactness.

### External evaluation
Compares discovered clusters with known reference labels, if reliable labels exist. Examples include Adjusted Rand Index and Normalized Mutual Information.

External labels must be kept out of the clustering fit. They are used only to evaluate how discovered groups compare with reference categories.

### Stability evaluation
Checks whether similar groups are found when random initialization, samples, time periods or reasonable preprocessing settings change.

### Business evaluation
Checks whether clusters have understandable profiles, are stable enough to use and support a useful business action.

## 4. Understand the main internal metrics

| Metric | What it measures | Preferred direction |
|---|---|---|
| Silhouette | Cohesion within clusters and separation between clusters | Higher is generally better |
| Davies-Bouldin | Similarity between each cluster and its most similar other cluster | Lower is generally better |
| Calinski-Harabasz | Between-cluster separation compared with within-cluster spread | Higher is generally better |
| Inertia | Within-cluster distance to K-Means centroids | Lower, but it usually falls as K increases |

No metric should be used alone.

## 5. Create a sample dataset

### 5.1 Import libraries

### 5.1. Import Pandas, NumPy and visualization libraries

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

These libraries prepare the dataset and display evaluation results.



### 5.2 Generate observations and reference labels

### 5.2. Create a synthetic clustering dataset

```python
from sklearn.datasets import make_blobs

X_values, reference_labels = make_blobs(
    n_samples=450,
    centers=4,
    n_features=2,
    cluster_std=1.0,
    random_state=42
)

print(X_values.shape)
print(pd.Series(reference_labels).value_counts())
```

**Expected output**

```text
(450, 2) and counts for four generated groups
```

**Explanation**

The reference labels are available only because the dataset is simulated. They will not be passed into the clustering algorithm.



### 5.3 Create a DataFrame

### 5.3. Name the feature columns

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

The clustering algorithm will use the two feature columns.



### 5.4 Visualize the observations

### 5.4. Plot the raw data

```python
plt.scatter(df["feature_1"], df["feature_2"])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Observations before clustering")
plt.show()
```

**Expected output**

```text
A scatter plot of the generated observations
```

**Explanation**

This gives a visual reference. Real clustering tasks usually do not have known reference labels.



## 6. Scale the features

### 6.1 Apply StandardScaler

### 6.1. Scale the clustering features

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

K-Means and distance-based metrics depend on feature geometry. Scaling is appropriate when feature units differ.



### 6.2 Inspect scaled shape

### 6.2. Check transformed dimensions

```python
print(X_scaled.shape)
```

**Expected output**

```text
(450, 2)
```

**Explanation**

Scaling changes values but not the number of observations or features.



## 7. Fit a K-Means candidate

### 7.1 Import KMeans

### 7.1. Import KMeans

```python
from sklearn.cluster import KMeans

print("KMeans imported")
```

**Expected output**

```text
KMeans imported
```

**Explanation**

K-Means creates candidate cluster labels for evaluation.



### 7.2 Fit a four-cluster model

### 7.2. Train K-Means

```python
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)

print(pd.Series(cluster_labels).value_counts().sort_index())
```

**Expected output**

```text
Counts for four discovered clusters
```

**Explanation**

The model is fitted without reference_labels. Cluster labels are assigned using feature values only.



### 7.3 Add labels to the original table

### 7.3. Create a clustering result table

```python
clustered_df = df.copy()
clustered_df["cluster"] = cluster_labels

print(clustered_df.head())
```

**Expected output**

```text
A preview containing features and cluster labels
```

**Explanation**

This table supports cluster profiling and visualization.



## 8. Calculate silhouette score

### 8.1 Import silhouette_score

### 8.1. Import the silhouette metric

```python
from sklearn.metrics import silhouette_score

print("Silhouette score imported")
```

**Expected output**

```text
Silhouette score imported
```

**Explanation**

The metric requires at least two clusters and fewer clusters than observations.



### 8.2 Calculate overall silhouette

### 8.2. Calculate the score

```python
silhouette = silhouette_score(
    X_scaled,
    cluster_labels
)

print(silhouette)
```

**Expected output**

```text
A score between -1 and 1
```

**Explanation**

Higher values generally indicate observations are closer to their own clusters and separated from other clusters.



### 8.3 Understand silhouette interpretation

- Values closer to +1 generally indicate a good assignment.
- Values near zero suggest observations may lie near cluster boundaries.
- Negative values suggest some observations may be closer to another cluster.

A high overall score does not prove that groups are commercially meaningful.

### 8.4 Calculate per-observation silhouette values

### 8.4. Inspect individual silhouette values

```python
from sklearn.metrics import silhouette_samples

individual_silhouette = silhouette_samples(
    X_scaled,
    cluster_labels
)

print(individual_silhouette[:10])
```

**Expected output**

```text
The first ten observation-level silhouette values
```

**Explanation**

Individual values help identify observations that fit poorly within their assigned clusters.



### 8.5 Attach silhouette values to the dataset

### 8.5. Add individual scores

```python
clustered_df["silhouette"] = individual_silhouette

print(clustered_df.head())
```

**Expected output**

```text
A preview with cluster label and silhouette value
```

**Explanation**

This supports cluster-level inspection rather than relying only on one overall score.



### 8.6 Summarize silhouette by cluster

### 8.6. Calculate average silhouette per cluster

```python
cluster_silhouette = clustered_df.groupby("cluster")["silhouette"].agg(
    ["count", "mean", "min", "max"]
)

print(cluster_silhouette)
```

**Expected output**

```text
A summary of silhouette values for each cluster
```

**Explanation**

A low average or many negative values in one cluster may indicate overlap or poor assignment for that group.



## 9. Calculate Davies-Bouldin index

**Davies-Bouldin index** compares the similarity of each cluster with its most similar other cluster.

- Lower values generally indicate more compact and separated clusters.
- It is not directly comparable across unrelated datasets with different feature representations.
- A low value does not prove business usefulness.

### 9.1 Import Davies-Bouldin metric

### 9.1. Import the metric

```python
from sklearn.metrics import davies_bouldin_score

print("Davies-Bouldin metric ready")
```

**Expected output**

```text
Davies-Bouldin metric ready
```

**Explanation**

The function uses the feature matrix and cluster labels.



### 9.2 Calculate Davies-Bouldin index

### 9.2. Calculate the index

```python
db_index = davies_bouldin_score(
    X_scaled,
    cluster_labels
)

print(db_index)
```

**Expected output**

```text
A non-negative Davies-Bouldin value
```

**Explanation**

Lower values generally indicate better separation and compactness under this metric.



## 10. Calculate Calinski-Harabasz score

**Calinski-Harabasz score** compares how separated clusters are from one another with how spread out observations are within clusters.

- Higher values generally indicate more distinct clusters.
- The score can increase with cluster count and dataset structure.
- Compare candidate settings on the same data representation.

### 10.1 Import the metric

### 10.1. Import Calinski-Harabasz score

```python
from sklearn.metrics import calinski_harabasz_score

print("Calinski-Harabasz metric ready")
```

**Expected output**

```text
Calinski-Harabasz metric ready
```

**Explanation**

The metric evaluates separation of the discovered groups.



### 10.2 Calculate the score

### 10.2. Calculate Calinski-Harabasz

```python
ch_score = calinski_harabasz_score(
    X_scaled,
    cluster_labels
)

print(ch_score)
```

**Expected output**

```text
A positive Calinski-Harabasz score
```

**Explanation**

Higher values generally indicate stronger separation relative to within-cluster spread.



## 11. Create a combined internal evaluation report

### 11.1 Collect metrics

### 11.1. Create the evaluation dictionary

```python
evaluation = {
    "silhouette": silhouette,
    "davies_bouldin": db_index,
    "calinski_harabasz": ch_score,
    "inertia": kmeans.inertia_,
    "cluster_count": len(set(cluster_labels))
}

print(evaluation)
```

**Expected output**

```text
A dictionary containing four internal metrics and cluster count
```

**Explanation**

Each metric has a different interpretation and preferred direction.



### 11.2 Convert metrics into a table

### 11.2. Create a readable metric report

```python
evaluation_table = pd.DataFrame({
    "metric": list(evaluation.keys()),
    "value": list(evaluation.values())
})

print(evaluation_table)
```

**Expected output**

```text
A metric summary table
```

**Explanation**

A report makes the values easier to compare and document.



## 12. Compare different K values

### 12.1 Define candidate cluster counts

### 12.1. Create a candidate K range

```python
candidate_ks = range(2, 9)

print(list(candidate_ks))
```

**Expected output**

```text
[2, 3, 4, 5, 6, 7, 8]
```

**Explanation**

Silhouette, Davies-Bouldin and Calinski-Harabasz require at least two clusters.



### 12.2 Calculate multiple metrics for each K

### 12.2. Evaluate candidate K values

```python
candidate_results = []

for k in candidate_ks:
    candidate = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = candidate.fit_predict(X_scaled)

    candidate_results.append({
        "k": k,
        "inertia": candidate.inertia_,
        "silhouette": silhouette_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels)
    })

candidate_table = pd.DataFrame(candidate_results)

print(candidate_table)
```

**Expected output**

```text
A table comparing four metrics across candidate cluster counts
```

**Explanation**

All candidate models use the same scaled observations and random seed. No single metric should determine K by itself.



### 12.3 Sort by silhouette

### 12.3. Review highest silhouette candidates

```python
print(
    candidate_table.sort_values(
        "silhouette",
        ascending=False
    )
)
```

**Expected output**

```text
Candidate K values ordered by silhouette score
```

**Explanation**

This is one ranking only. Compare it with other metrics, profiles and business interpretation.



### 12.4 Sort by Davies-Bouldin

### 12.4. Review lower Davies-Bouldin candidates

```python
print(
    candidate_table.sort_values(
        "davies_bouldin",
        ascending=True
    )
)
```

**Expected output**

```text
Candidate K values ordered by Davies-Bouldin index
```

**Explanation**

Lower is generally better for this metric. Check whether the suggested K produces useful group sizes.



### 12.5 Sort by Calinski-Harabasz

### 12.5. Review higher Calinski-Harabasz candidates

```python
print(
    candidate_table.sort_values(
        "calinski_harabasz",
        ascending=False
    )
)
```

**Expected output**

```text
Candidate K values ordered by Calinski-Harabasz score
```

**Explanation**

Higher is generally better for this metric, but the score should be interpreted with the other evidence.



## 13. Evaluate DBSCAN carefully

DBSCAN can label observations as noise using -1. Internal metrics need careful handling because noise is not an ordinary cluster.

### 13.1 Fit DBSCAN

### 13.1. Train DBSCAN

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.25,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(X_scaled)

print(pd.Series(dbscan_labels).value_counts().sort_index())
```

**Expected output**

```text
Counts for DBSCAN cluster labels and possible noise
```

**Explanation**

The algorithm identifies dense groups and may label some observations as -1.



### 13.2 Create a mask for non-noise observations

### 13.2. Exclude noise for this silhouette calculation

```python
non_noise = dbscan_labels != -1

print(non_noise.sum())
```

**Expected output**

```text
The number of observations assigned to non-noise clusters
```

**Explanation**

The following silhouette calculation describes only assigned clusters, not noise observations.



### 13.3 Count non-noise clusters

### 13.3. Check the number of clusters

```python
non_noise_labels = dbscan_labels[non_noise]
cluster_ids = set(non_noise_labels)

print(len(cluster_ids))
```

**Expected output**

```text
The number of non-noise clusters
```

**Explanation**

Silhouette is only valid when at least two clusters exist and there are fewer clusters than observations.



### 13.4 Calculate silhouette safely

### 13.4. Calculate DBSCAN silhouette when valid

```python
if len(cluster_ids) >= 2 and len(cluster_ids) < len(non_noise_labels):
    dbscan_silhouette = silhouette_score(
        X_scaled[non_noise],
        non_noise_labels
    )
    print(dbscan_silhouette)
else:
    print("Silhouette is not defined for this clustering result")
```

**Expected output**

```text
A silhouette score or a message explaining why it is unavailable
```

**Explanation**

Always report that noise points were excluded. A score calculated after removing noise does not evaluate how the detector handled unusual observations.



### 13.5 Report noise proportion

### 13.5. Calculate DBSCAN noise percentage

```python
noise_count = int(np.sum(dbscan_labels == -1))
noise_percent = noise_count / len(dbscan_labels) * 100

print("Noise count:", noise_count)
print("Noise percentage:", noise_percent)
```

**Expected output**

```text
The count and percentage of noise observations
```

**Explanation**

Noise proportion is important for understanding DBSCAN behaviour and operational impact.



## 14. External evaluation when reference labels exist

In some applications, a reference grouping may exist, such as known product categories or manually reviewed segments.

Use those labels only for evaluation, not as input to an unsupervised model when the goal is unsupervised discovery.

### 14.1 Import external metrics

### 14.1. Import ARI and NMI

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

print("External metrics ready")
```

**Expected output**

```text
External clustering metrics ready
```

**Explanation**

These metrics compare discovered clusters with reference labels while accounting for label permutation.



### 14.2 Calculate Adjusted Rand Index

### 14.2. Compare clusters with reference labels

```python
ari = adjusted_rand_score(
    reference_labels,
    cluster_labels
)

print(ari)
```

**Expected output**

```text
An Adjusted Rand Index value
```

**Explanation**

ARI compares pairwise grouping agreement and adjusts for chance. Higher values indicate stronger agreement with reference labels.



### 14.3 Calculate Normalized Mutual Information

### 14.3. Calculate NMI

```python
nmi = normalized_mutual_info_score(
    reference_labels,
    cluster_labels
)

print(nmi)
```

**Expected output**

```text
A Normalized Mutual Information value
```

**Explanation**

NMI measures shared information between reference groups and discovered clusters. It is useful when labels are available, but it is not a business-value metric.



## 15. Check clustering stability

A cluster result should not change completely when random initialization changes slightly.

Adjusted Rand Index can compare cluster assignments across runs because it does not depend on the numeric order of cluster IDs.

### 15.1 Fit a second K-Means run

### 15.1. Train with another random seed

```python
kmeans_second = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=7
)

labels_second = kmeans_second.fit_predict(X_scaled)

print(pd.Series(labels_second).value_counts().sort_index())
```

**Expected output**

```text
A count for each cluster from the second run
```

**Explanation**

The second model uses the same data and cluster count but a different initialization seed.



### 15.2 Compare assignments using ARI

### 15.2. Calculate stability across runs

```python
stability_ari = adjusted_rand_score(
    cluster_labels,
    labels_second
)

print(stability_ari)
```

**Expected output**

```text
An ARI value comparing the two cluster assignments
```

**Explanation**

A high score indicates similar group membership across the two runs, even if cluster IDs are permuted.



### 15.3 Compare several seeds

### 15.3. Repeat clustering with multiple seeds

```python
stability_rows = []

for seed in [1, 7, 21, 42, 99]:
    candidate = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=seed
    )

    labels = candidate.fit_predict(X_scaled)

    stability_rows.append({
        "seed": seed,
        "inertia": candidate.inertia_,
        "ari_vs_reference_run": adjusted_rand_score(
            cluster_labels,
            labels
        )
    })

print(pd.DataFrame(stability_rows))
```

**Expected output**

```text
A table of inertia and ARI stability values
```

**Explanation**

This helps determine whether cluster membership is sensitive to initialization.



## 16. Cluster profile and business validation

Numerical metrics do not replace interpretation.

Review:
- Cluster sizes and percentages.
- Original feature distributions.
- Missing-value rates.
- Geographic and time coverage.
- Whether groups are stable.
- Whether the groups support a business action.
- Whether any group represents a sensitive or protected population.
- Whether the clustering process could reinforce bias.

### 16.1 Create cluster profiles

### 16.1. Profile the K-Means result

```python
profile_df = df.copy()
profile_df["cluster"] = cluster_labels

profile = profile_df.groupby("cluster").agg(
    count=("feature_1", "size"),
    feature_1_mean=("feature_1", "mean"),
    feature_2_mean=("feature_2", "mean"),
    feature_1_std=("feature_1", "std"),
    feature_2_std=("feature_2", "std")
)

print(profile)
```

**Expected output**

```text
A profile table with cluster sizes, means and standard deviations
```

**Explanation**

This summarizes each group's location and spread in original units.



### 16.2 Calculate cluster percentages

### 16.2. Report cluster proportions

```python
profile["percentage"] = (
    profile["count"] / len(profile_df) * 100
)

print(profile[["count", "percentage"]])
```

**Expected output**

```text
Cluster counts and percentages
```

**Explanation**

Percentages make group sizes easier to communicate to stakeholders.



## 17. Write a clustering evaluation report

A concise report should contain:
1. Business question and observation unit.
2. Dataset version and selected features.
3. Preprocessing and scaling choices.
4. Clustering algorithm and settings.
5. Internal evaluation metrics.
6. Stability evidence.
7. Cluster sizes and profiles.
8. Noise count where relevant.
9. Business interpretation and limitations.
10. Recommended next action or additional evidence required.

Do not write "the model discovered the correct segments" unless there is a defensible reference definition and evidence.

## 18. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Use silhouette as the only metric | Combine several metrics and domain review |
| Compare scores from different feature scales | Use consistent preprocessing |
| Ignore noise points in DBSCAN reports | Report noise proportion and state how metrics were calculated |
| Assume high internal score means business value | Profile clusters and validate usefulness |
| Ignore stability | Compare runs and samples |
| Use external labels during unsupervised fitting | Keep them for evaluation only |
| Compare cluster IDs directly | Use permutation-invariant comparisons such as ARI |
| Ignore small clusters | Inspect size, validity and operational relevance |

## 19. Interview-ready explanations

**Why is clustering evaluation difficult?**  
There may be no correct labels. Different algorithms can find different structures, so evaluation combines internal metrics, external metrics when labels exist, stability and domain validation.

**What is silhouette score?**  
It compares how well an observation fits its assigned cluster with how separated it is from other clusters. Higher values generally indicate better separation.

**What is Davies-Bouldin index?**  
It summarizes similarity between clusters based on within-cluster spread and separation. Lower values generally indicate better clustering under this measure.

**What is Calinski-Harabasz score?**  
It compares between-cluster separation with within-cluster spread. Higher values generally indicate more distinct clusters.

**What is ARI used for?**  
It compares two cluster assignments or a discovered clustering with reference labels while accounting for arbitrary cluster numbering.

## 20. Scenario questions

1. Silhouette is high but one cluster contains only two observations. What do you do?  
   **Answer:** Investigate the small cluster, profile its records and assess whether it is a rare segment or over-segmentation.

2. Davies-Bouldin and silhouette prefer different K values. What is your response?  
   **Answer:** Explain that the metrics measure different aspects. Compare profiles, stability and business usefulness before selecting K.

3. Two runs use different cluster IDs but have similar memberships. Is that unstable?  
   **Answer:** Not necessarily. Compare with ARI or another label-permutation-invariant measure.

4. DBSCAN silhouette is high after excluding noise. What must be reported?  
   **Answer:** Report the noise count and percentage, and state that silhouette was calculated only on non-noise observations.

5. Internal metrics are good but stakeholders cannot act on the clusters. Is the model successful?  
   **Answer:** Not yet. The groups need to support a useful and validated decision.

## 21. Lesson completion checklist

- [ ] I can explain internal, external and stability evaluation.
- [ ] I can calculate silhouette score.
- [ ] I can calculate Davies-Bouldin and Calinski-Harabasz scores.
- [ ] I can compare candidate cluster counts.
- [ ] I can evaluate DBSCAN while reporting noise.
- [ ] I can use ARI and NMI when reference labels exist.
- [ ] I can compare clustering stability across runs.
- [ ] I can profile clusters and write a quality report.
- [ ] I understand why no single metric proves business usefulness.

**Next lesson:** Day 43 — Unsupervised Learning Project Integration.
