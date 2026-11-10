# Day 38: Principal Component Analysis and Dimensionality Reduction

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define dimensionality reduction.
- Explain PCA and principal components.
- Understand explained variance.
- Apply scaling and fit PCA safely.
- Recognize interpretability trade-offs.

## 1. What Is Dimensionality Reduction?
Dimensionality reduction reduces the number of input dimensions while attempting to preserve useful information or structure.

## 2. What Is PCA?
Principal Component Analysis (PCA) transforms correlated numeric features into new orthogonal variables called principal components. The first component captures the greatest variance; later components capture as much remaining variance as possible under the PCA procedure.

## 3. Syntax
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train_scaled)
print(pca.n_components_)
print(pca.explained_variance_ratio_)
```
n_components=0.95 asks PCA to retain enough components to explain at least 95% of training variance.

## 4. Three Applications
1. Compress many correlated sensor features.
2. Project high-dimensional data into two components for visualization.
3. Reduce redundancy before a downstream model, after validating performance.

## 5. Scenario Questions
1. PCA retains 95% variance. Does that guarantee 95% predictive information? No; variance explained is not the same as target relevance.
2. Components are difficult to explain to a manager. Why? Each component combines original features.
3. Should PCA be fitted on the full dataset before splitting? No; fit on training data only and transform held-out data.
