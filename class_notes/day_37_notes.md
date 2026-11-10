# Day 37: DBSCAN and Density-Based Clustering

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define density-based clustering.
- Explain eps and min_samples.
- Understand core, border and noise points.
- Compare DBSCAN with K-Means.
- Recognize density and scaling limitations.

## 1. What Is DBSCAN?
DBSCAN groups observations in dense regions and labels isolated observations as noise. It does not require the number of clusters to be specified in advance.

Key terms:
- eps: neighbourhood radius.
- min_samples: minimum number of nearby points needed for a core point.
- Core point: has enough neighbours in its eps neighbourhood.
- Noise point: does not belong to a dense cluster under the chosen settings.

## 2. Syntax
```python
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X_scaled)
print(labels)
```
In scikit-learn, -1 indicates noise.

## 3. Three Applications
1. Discover dense GPS stop locations.
2. Identify groups of similar spatial events.
3. Flag isolated observations for investigation.

## 4. Scenario Questions
1. DBSCAN labels many records -1. What does that mean? They were treated as noise under current parameters.
2. Clusters disappear when eps changes slightly. What should you investigate? Parameter sensitivity, scale and data density.
3. Data has clusters with very different densities. What limitation may arise? One eps value may not suit all clusters.
