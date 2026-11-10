# Day 34: K-Means and Centroid-Based Clustering

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain centroid and K-Means.
- Describe assignment and centroid-update steps.
- Use KMeans in scikit-learn.
- Understand K, initialization, scaling and outlier sensitivity.

## 1. What Is K-Means?
K-Means partitions observations into K clusters. It assigns each observation to the nearest centroid, recalculates centroids and repeats until assignments/centres stabilize or a stopping condition is reached.

A **centroid** is the mean position of the observations assigned to a cluster.

## 2. Syntax
```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels = model.fit_predict(X_scaled)
centres = model.cluster_centers_
```
- n_clusters: requested number of clusters.
- random_state: helps make initialization repeatable.
- n_init: number of initializations/restarts.
- labels: cluster assignment for each row.
- cluster_centers_: learned centroid coordinates.

## 3. Three Examples
1. Customer segments using frequency and spend.
2. Group stores by sales and footfall.
3. Group products by numerical attributes.

## 4. Limitations
K must be chosen; scaling often matters; outliers can pull centroids; K-Means works best for roughly compact/spherical clusters under the selected distance.

## 5. Scenario Questions
1. One feature is measured in thousands and another in single digits. What should you consider? Scaling.
2. One extreme customer spend value pulls a centroid. What is the issue? K-Means sensitivity to outliers.
3. K=3 gives three groups but the business needs five. What next? Compare candidate K values and interpret usefulness; do not set K only to match a desired story.
