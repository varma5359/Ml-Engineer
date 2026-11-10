# Day 35: Interview Questions — Elbow Method and Silhouette Score

**Pathway:** Machine Learning Engineer

### Q1. Why do we need to select K in K-Means?
**Answer:** K-Means needs the number of clusters before fitting. The selected K controls how many groups the algorithm creates.
**Follow-up:** Is the largest K always best?  
**Answer:** No. More clusters can create unnecessary complexity.

### Q2. What is inertia?
**Answer:** It summarizes how close observations are to their assigned centroids. Lower inertia means observations are closer to their cluster centres.
**Follow-up:** Why does inertia usually decrease when K increases?  
**Answer:** More centres can represent the data more closely.

### Q3. What is the elbow method?
**Answer:** It plots inertia across candidate K values and looks for a bend where adding clusters gives diminishing improvement.
**Follow-up:** Is the elbow always clear?  
**Answer:** No. Some datasets do not have a sharp elbow.

### Q4. What is silhouette score?
**Answer:** It compares how well an observation fits its assigned cluster with how separated it is from other clusters.
**Follow-up:** What does a negative silhouette suggest?  
**Answer:** The observation may be closer to another cluster than the one assigned.

### Q5. Does a high silhouette score prove business value?
**Answer:** No. It describes geometric separation, not commercial usefulness. Cluster profiles and domain review are still needed.

## Scenario questions

### Q6. Inertia keeps decreasing as K increases. What do you do?
**Answer:** Look for diminishing returns, silhouette scores, cluster sizes, profile clarity and business usefulness instead of minimizing inertia alone.

### Q7. Silhouette is highest for K=2, but the business asks for detailed segments. What do you consider?
**Answer:** Compare the stability and usefulness of additional clusters. Do not create segments that the data cannot support.

### Q8. One cluster contains only three records. What do you investigate?
**Answer:** Check whether the observations are valid rare cases, outliers or evidence that K is too high.

### Q9. Two K values have similar silhouette scores. How do you choose?
**Answer:** Compare cluster profiles, stability, size and whether each grouping supports a useful action.

### Q10. A high silhouette score appears after including customer ID. What is the concern?
**Answer:** The score may reflect meaningless identifier structure. Rebuild using meaningful behavioural features.

**Senior developer principle:** Use elbow and silhouette as evidence, then validate the clusters using their profiles and intended business use.
