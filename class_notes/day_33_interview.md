# Day 33: Interview Questions — Clustering and Unsupervised Learning

**Pathway:** Machine Learning Engineer

### Q1. What is unsupervised learning?
**Answer:** It learns patterns or structure from input data without supplied target labels.
**Follow-up:** Give an example.  
**Answer:** Grouping customers based on purchasing behaviour when no customer segment labels exist.

### Q2. What is clustering?
**Answer:** It groups observations based on similarity in selected features.
**Follow-up:** Does the algorithm know what a cluster means to the business?  
**Answer:** No. Analysts profile the groups and interpret them.

### Q3. How is clustering different from classification?
**Answer:** Classification predicts predefined labels using labelled examples. Clustering discovers groups without predefined labels.
**Follow-up:** Are cluster IDs like 0 and 1 known business classes?  
**Answer:** No. They are identifiers assigned to discovered groups.

### Q4. Why is feature selection important for clustering?
**Answer:** Clustering depends on the selected feature representation. Irrelevant features or IDs can distort similarity and produce meaningless groups.

### Q5. Why might scaling be important?
**Answer:** Distance-based clustering can be dominated by features with larger numerical ranges. Scaling can make feature magnitudes more comparable.

### Q6. How do you evaluate clusters without target labels?
**Answer:** Use internal metrics, cluster profiles, stability and domain validation. A numerical score alone does not prove business usefulness.

### Q7. One cluster contains very few customers. Is it automatically wrong?
**Answer:** No. It may be a rare segment or noise. Inspect its profile, source records and stability.

### Q8. A clustering metric improves but business teams cannot explain the groups. Is the model ready?
**Answer:** Not necessarily. The groups need meaningful interpretation and a useful business action.

**Senior developer principle:** Clustering discovers structure; it does not automatically discover meaningful business segments.
