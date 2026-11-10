# Day 36: Hierarchical Clustering and Dendrograms

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define hierarchical clustering.
- Distinguish agglomerative and divisive approaches.
- Understand linkage and distance.
- Read a dendrogram.
- Recognize scale and computational limitations.

## 1. What Is Hierarchical Clustering?
Hierarchical clustering creates nested groups. Agglomerative clustering starts with individual observations and repeatedly merges groups. Divisive clustering starts with one group and repeatedly splits it.

## 2. What Is a Dendrogram?
A dendrogram is a tree diagram showing which observations or groups are merged and at what distance. Cutting the tree at a chosen height produces a set of clusters.

## 3. Syntax
```python
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

Z = linkage(X_scaled, method="ward")
dendrogram(Z)
plt.show()
```
Ward linkage merges groups to minimize the increase in within-cluster variance and is associated with Euclidean distance.

## 4. Three Applications
1. Group products by similarity.
2. Explore customer segments at different levels of granularity.
3. Group documents or biological samples based on a defined distance representation.

## 5. Scenario Questions
1. A team wants to inspect group structure before choosing K. What can help? Dendrogram.
2. Why can feature scaling matter? Distance calculations can be dominated by large-unit features.
3. Dataset has millions of records. Is hierarchical clustering always practical? No; computational and memory costs can be high.
