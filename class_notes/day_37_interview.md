# Day 37: Interview Questions — DBSCAN and Density-Based Clustering

**Pathway:** Machine Learning Engineer

### Q1. What is DBSCAN?
**Answer:** It is a density-based clustering algorithm that groups dense regions and can label sparse observations as noise.
**Follow-up:** Does it require the number of clusters in advance?  
**Answer:** No. The number of clusters is discovered from density connectivity.

### Q2. What does eps control?
**Answer:** It defines the neighbourhood radius around an observation.
**Follow-up:** What if eps is too small?  
**Answer:** Many points may be labelled noise and clusters may fragment.

### Q3. What does min_samples control?
**Answer:** It is the minimum number of observations needed in a neighbourhood for a point to qualify as a core point.
**Follow-up:** What if min_samples is increased?  
**Answer:** The density requirement becomes stronger and more points may become noise.

### Q4. What is a core point?
**Answer:** A point with enough observations in its eps neighbourhood.
**Follow-up:** What is a border point?  
**Answer:** A point near a core point but without enough neighbours of its own to qualify as core.

### Q5. What does label -1 mean?
**Answer:** It indicates a noise observation not assigned to a dense cluster.
**Follow-up:** Is it necessarily an error?  
**Answer:** No. It may be a rare but valid observation.

## Scenario questions

### Q6. DBSCAN labels most observations as noise. What do you inspect?
**Answer:** Feature scaling, eps, min_samples, data density and whether selected features represent meaningful similarity.

### Q7. Two groups merge when eps increases. What does that indicate?
**Answer:** The neighbourhood radius may be too large relative to the gap between groups.

### Q8. K-Means struggles with curved clusters. What could you try?
**Answer:** DBSCAN or another method that can identify non-spherical dense structures.

### Q9. One cluster is dense and another is sparse. What is the challenge?
**Answer:** A single eps and min_samples setting may not represent both densities well.

### Q10. How do you evaluate DBSCAN?
**Answer:** Inspect cluster count, noise proportion, silhouette where appropriate, cluster profiles, stability and domain usefulness. State how noise points are handled in any metric.

**Senior developer principle:** DBSCAN identifies dense regions under chosen distance settings; noise is a model output that needs context.
