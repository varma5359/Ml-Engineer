# Day 34: Interview Questions — K-Means Clustering and Centroid-Based Grouping

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is K-Means?
**Answer:** It is a centroid-based clustering algorithm that assigns observations to the nearest centre and repeatedly updates the cluster centres.
**Follow-up:** Is K-Means supervised?  
**Answer:** No. It is an unsupervised learning method.

### Q2. What is a centroid?
**Answer:** It is the centre of a cluster, calculated from the observations assigned to that cluster.
**Follow-up:** Is a centroid necessarily a real customer?  
**Answer:** No. It is a representative feature location.

### Q3. What does n_clusters control?
**Answer:** It specifies how many clusters K-Means should create.
**Follow-up:** How do you select it?  
**Answer:** Use methods such as elbow and silhouette analysis, cluster profiles, stability and business usefulness.

### Q4. What is inertia?
**Answer:** It summarizes within-cluster distances to centroids. Lower inertia means observations are closer to their assigned centres.
**Follow-up:** Why not choose the largest K because it lowers inertia?  
**Answer:** Inertia generally decreases as K increases. More clusters can create unnecessary complexity.

### Q5. Why use k-means++ initialization?
**Answer:** It selects initial centres in a way that tends to spread them out, often providing better starting points than arbitrary initialization.

## Scenario questions

### Q6. K-Means groups customers mainly by annual spending. What do you inspect?
**Answer:** Feature ranges, scaling, units and whether the selected features reflect the intended customer behaviour.
**Follow-up:** What if spending should genuinely dominate?  
**Answer:** Keep the feature influence if it matches the business objective, but document and validate the choice.

### Q7. Cluster IDs change between runs. Is that necessarily a problem?
**Answer:** No. IDs are arbitrary and initialization can change the order or assignment. Compare cluster profiles and stability.
**Follow-up:** How can you improve repeatability?  
**Answer:** Set random_state and use suitable n_init settings.

### Q8. One cluster contains almost all customers. What do you investigate?
**Answer:** Selected K, feature scaling, outliers, feature relevance and whether the data naturally supports distinct groups.

### Q9. A centroid represents a spend score and purchase frequency that no actual customer has. Is that valid?
**Answer:** Yes. A centroid is an average feature location and does not need to correspond to an actual observation.

### Q10. Why should customer IDs be excluded from K-Means features?
**Answer:** IDs are labels, not behavioural measurements. Their numeric or string representation does not indicate customer similarity.

**Senior developer principle:** Interpret K-Means clusters using original business features, not the cluster number alone.
