# Day 35: Elbow Method and Silhouette Score

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain why selecting K matters.
- Define inertia and the elbow method.
- Explain silhouette score.
- Compare cluster quality and business usefulness.

## 1. What Is Inertia?
In K-Means, inertia is the sum of squared distances from each observation to its assigned cluster centre. It generally decreases as K increases, so a lower value alone does not prove the best K.

## 2. Elbow Method
Fit K-Means for several K values and plot inertia. Look for a point where the reduction begins to slow—the “elbow”. Sometimes no clear elbow exists.

## 3. Silhouette Score
Silhouette compares how close an observation is to its own cluster versus other clusters. Scores range from -1 to +1; higher values generally indicate better separation under the chosen distance.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(X_scaled)
    print(k, model.inertia_, silhouette_score(X_scaled, labels))
```

## 4. Three Selection Considerations
1. Elbow shape and inertia trend.
2. Silhouette score and cluster separation.
3. Cluster size, stability and whether business teams can act on the groups.

## 5. Scenario Questions
1. Inertia keeps decreasing as K grows. Is that surprising? No; more clusters usually reduce within-cluster distance.
2. Highest silhouette produces one tiny cluster. Is that automatically best? No; inspect sizes and usefulness.
3. No clear elbow exists. What should you do? Compare other evidence and business interpretability; explain uncertainty.
