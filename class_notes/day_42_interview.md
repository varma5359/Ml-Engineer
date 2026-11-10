# Day 42: Interview Questions — Clustering Evaluation and Limitations

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. Why is clustering evaluation difficult?
**Answer:** There may be no correct labels. Evaluation combines internal metrics, external metrics when reference labels exist, stability checks and domain validation.

### Q2. What is silhouette score?
**Answer:** It compares how well an observation fits its own cluster with how separated it is from other clusters. Higher values generally indicate better cohesion and separation.
**Follow-up:** What does a negative value suggest?  
**Answer:** The observation may be closer to another cluster than its assigned cluster.

### Q3. What is Davies-Bouldin index?
**Answer:** It summarizes similarity between clusters using within-cluster spread and separation. Lower values generally indicate better clustering under this measure.

### Q4. What is Calinski-Harabasz score?
**Answer:** It compares between-cluster separation with within-cluster spread. Higher values generally indicate more distinct clusters.

### Q5. What is Adjusted Rand Index?
**Answer:** ARI compares two cluster assignments or discovered clusters with reference labels while accounting for arbitrary cluster numbering.
**Follow-up:** Why is it useful across different random seeds?  
**Answer:** It compares membership agreement without requiring the numeric cluster IDs to match.

## Scenario questions

### Q6. Silhouette is high but one cluster contains only two observations. What do you do?
**Answer:** Investigate the small cluster, profile its records and assess whether it is a rare segment or over-segmentation.

### Q7. Davies-Bouldin and silhouette prefer different K values. What is your response?
**Answer:** Explain that the metrics measure different aspects. Compare profiles, stability and business usefulness before selecting K.

### Q8. Two runs use different cluster IDs but have similar memberships. Is that unstable?
**Answer:** Not necessarily. Compare using ARI or another label-permutation-invariant measure.

### Q9. DBSCAN silhouette is high after excluding noise. What must be reported?
**Answer:** Report the noise count and percentage and state that silhouette was calculated only on non-noise observations.

### Q10. Internal metrics are good but stakeholders cannot act on the clusters. Is the model successful?
**Answer:** Not yet. The groups need to support a useful and validated decision.

### Q11. Can external labels be used to fit an unsupervised clustering model?
**Answer:** Not if the goal is unsupervised discovery. Keep reference labels out of fitting and use them only for external evaluation.

**Senior developer principle:** No single clustering metric proves that a segmentation is stable, meaningful or useful.
