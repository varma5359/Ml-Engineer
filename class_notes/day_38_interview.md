# Day 38: Interview Questions — Principal Component Analysis and Dimensionality Reduction

**Pathway:** Machine Learning Engineer

### Q1. What is dimensionality reduction?
**Answer:** It reduces the number of input dimensions while attempting to preserve useful information.
**Follow-up:** Why might it help?  
**Answer:** It can reduce redundancy, computation and difficulty visualizing high-dimensional data.

### Q2. What is PCA?
**Answer:** It is an unsupervised technique that creates new combined features called principal components, ordered by how much variation they represent.
**Follow-up:** Does PCA select original columns?  
**Answer:** No. It creates combinations of original features.

### Q3. What is explained variance ratio?
**Answer:** It describes the share of original feature variation represented by a principal component.
**Follow-up:** Does high explained variance guarantee high classification accuracy?  
**Answer:** No. PCA does not use the target and may preserve variation that is not useful for predicting it.

### Q4. Why scale features before PCA?
**Answer:** PCA is influenced by feature variance. If units differ greatly, large-scale features can dominate the components.
**Follow-up:** Is scaling always identical for every dataset?  
**Answer:** No. Choose preprocessing based on units, distributions and the task.

### Q5. What are component loadings?
**Answer:** They describe how original features contribute to a principal component.
**Follow-up:** Is PC1 automatically a business concept?  
**Answer:** No. Interpret it by examining loadings and domain context.

## Scenario questions

### Q6. A dataset has 500 numerical features. What might PCA help with?
**Answer:** It can create a smaller representation that retains a chosen share of feature variation, helping exploration or modelling.

### Q7. PCA retains 95% of variation but model accuracy decreases. Is that possible?
**Answer:** Yes. PCA preserves overall feature variation, not necessarily the information most useful for predicting the target.

### Q8. Should PCA be fitted before train-test split?
**Answer:** No. Fit PCA on training data only and transform held-out data using the fitted components.

### Q9. A business team cannot interpret principal components. What can you do?
**Answer:** Show original-feature loadings, compare with original features and assess whether interpretability requirements justify PCA.

### Q10. Does PCA guarantee better model performance?
**Answer:** No. Compare a model using original features with one using PCA under the same validation strategy.

**Senior developer principle:** PCA compresses feature variation; it does not guarantee that the compressed representation preserves the target information.
