# Day 33: Clustering and Unsupervised Learning

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define unsupervised learning and clustering.
- Explain similarity and distance.
- Understand why scaling and feature selection matter.
- Interpret clusters without assuming they are true labels.
- Identify clustering use cases.

## 1. What Is Clustering?
Clustering is an unsupervised learning task that groups observations based on similarity in selected features. The dataset does not provide a correct group label for each row.

## 2. Three Examples
1. Group customers by order frequency and spending.
2. Group products by attributes and purchase patterns.
3. Group locations by similar service demand.

## 3. How to Think About a Cluster
A cluster label such as 0, 1 or 2 is an algorithm identifier, not a business meaning. After fitting, inspect group sizes, feature averages and representative records. Then assign a meaningful description only if evidence supports it.

## 4. Syntax: K-Means as a first example
```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels = model.fit_predict(X_scaled)
print(labels)
```

## 5. Scenario Questions
1. A business asks for customer groups but has no existing segment labels. Which learning category? Unsupervised learning.
2. The algorithm returns cluster 0. Does that mean “premium customer”? No; inspect the cluster profile.
3. Clusters change greatly when one feature is removed. What does this suggest? The grouping is sensitive to feature selection and needs validation.
