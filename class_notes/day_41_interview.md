# Day 41: Interview Questions — Unsupervised Model Interpretation

**Pathway:** Machine Learning Engineer

### Q1. Why do unsupervised model outputs need interpretation?
**Answer:** The algorithm discovers patterns but does not know their business meaning. We must inspect feature values, cluster sizes, distributions and stability before describing groups.

### Q2. How do you interpret a cluster?
**Answer:** Profile it using original feature values, compare its size and distributions with other clusters, and review the pattern with domain experts.

### Q3. Are cluster IDs rankings?
**Answer:** No. Cluster IDs are arbitrary identifiers assigned by the algorithm. Cluster 2 is not automatically better or more important than cluster 1.

### Q4. What is a centroid?
**Answer:** It is a representative centre calculated from observations assigned to a cluster. It may not correspond to a real customer or record.

### Q5. Why are cluster sizes important?
**Answer:** They show how many observations support each profile. Very small groups may be rare valid segments, noise or over-segmentation.

### Q6. Why inspect distributions instead of only averages?
**Answer:** Similar averages can hide very different spreads, ranges or subgroups. Medians, quartiles and plots provide additional context.

## Scenario questions

### Q7. Cluster 1 has the highest average spend. Can the company call it the most valuable segment?
**Answer:** Not automatically. Check purchase frequency, margin, retention, group size and the business definition of value.

### Q8. Two clusters have similar average spend but different purchase frequency. How do you describe them?
**Answer:** Use both features and inspect distributions to explain the behavioural difference.

### Q9. A cluster contains only a handful of customers. What do you do?
**Answer:** Review those records, assess whether they are rare valid cases or noise, and avoid broad conclusions from a tiny group.

### Q10. Cluster labels change between runs. What do you compare?
**Answer:** Compare profiles and membership overlap rather than relying on numeric label values.

### Q11. A stakeholder says clustering proves that a campaign caused higher spending. How do you respond?
**Answer:** Clustering describes patterns. It does not establish causation. Use an appropriate experiment to measure campaign impact.

**Senior developer principle:** Explain what the data shows, what it does not show, and what evidence is needed before taking action.
