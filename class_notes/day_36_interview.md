# Day 36: Interview Questions — Hierarchical Clustering and Dendrograms

**Pathway:** Machine Learning Engineer

### Q1. What is hierarchical clustering?
**Answer:** It builds a hierarchy of groups by repeatedly merging or splitting clusters. Agglomerative clustering starts with individual observations and merges them.
**Follow-up:** What is the opposite direction?  
**Answer:** Divisive clustering starts with one group and repeatedly splits it.

### Q2. What is a dendrogram?
**Answer:** A tree diagram showing the order and distance at which observations or groups are merged.
**Follow-up:** What does merge height represent?  
**Answer:** The distance or dissimilarity at which the merge occurs.

### Q3. What is linkage?
**Answer:** It defines how the distance between clusters is calculated.
**Follow-up:** Name common linkage methods.  
**Answer:** Single, complete, average and Ward.

### Q4. Compare single and complete linkage.
**Answer:** Single uses the closest pair between clusters and can create chain-like groups. Complete uses the farthest pair and often favours compact groups.

### Q5. What is Ward linkage?
**Answer:** It merges clusters in a way that limits the increase in within-cluster variation. It uses Euclidean distance.
**Follow-up:** Can it be paired with any arbitrary distance?  
**Answer:** No. Ward's criterion is based on Euclidean geometry.

## Scenario questions

### Q6. The dendrogram has a large vertical gap before a final merge. What might you investigate?
**Answer:** A cut below that merge may create separated groups, but confirm using cluster metrics and domain interpretation.

### Q7. Single linkage creates one long chain-like cluster. Why?
**Answer:** It merges based on the closest pair of observations, so nearby points can connect otherwise distinct groups.

### Q8. Hierarchical clustering is too slow on millions of records. What do you consider?
**Answer:** Use a representative sample for exploration, a more scalable clustering method or an algorithm better suited to the data size.

### Q9. Two cluster counts have similar silhouette scores. How do you choose?
**Answer:** Compare profile clarity, stability, cluster size and business actionability.

### Q10. Does AgglomerativeClustering provide the same simple prediction workflow as KMeans for new records?
**Answer:** Standard scikit-learn agglomerative clustering does not provide the same simple predict method. A new-observation assignment strategy must be designed if required.

**Senior developer principle:** A dendrogram helps explore grouping structure, but the selected clusters still need validation and interpretation.
