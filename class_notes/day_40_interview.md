# Day 40: Interview Questions — Anomaly Detection Concepts

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is anomaly detection?
**Answer:** It identifies observations that differ from expected or common patterns and may require investigation.
**Follow-up:** Does anomaly mean fraud?  
**Answer:** No. An anomaly can be valid, erroneous or suspicious.

### Q2. What is the difference between an anomaly and a data error?
**Answer:** An anomaly is unusual relative to a pattern. A data error is incorrect information caused by collection, entry, parsing or system problems. An anomaly may be valid.

### Q3. What is Isolation Forest?
**Answer:** It isolates observations through randomized feature splits. Unusual observations are often isolated more easily than common ones.
**Follow-up:** What does contamination mean?  
**Answer:** It is an estimate of the expected outlier proportion used to set a threshold, not a guarantee of the true anomaly rate.

### Q4. What is Local Outlier Factor?
**Answer:** It compares the local density around an observation with the density around its neighbours to identify unusual local patterns.
**Follow-up:** What does n_neighbors control?  
**Answer:** The size of the local neighbourhood used for comparison.

### Q5. What is the difference between point, contextual and collective anomalies?
**Answer:** A point anomaly is one unusual observation; a contextual anomaly is unusual in a specific context; a collective anomaly is an unusual group or sequence.

## Scenario questions

### Q6. A detector flags a large transaction from a premium customer. What do you do?
**Answer:** Review the customer's history, transaction context, source data and business rules before deciding whether it is suspicious.

### Q7. Isolation Forest flags 20% of transactions but the review team can investigate only 2%. What do you consider?
**Answer:** Revisit feature quality, threshold settings, ranking strategy and review capacity. Prioritize by risk and business impact.

### Q8. A new fraud pattern is not detected because it looks similar to normal transactions. What is the limitation?
**Answer:** Unsupervised anomaly detection cannot identify every harmful event. Labels, domain rules and better features may be required.

### Q9. The alert count doubles after a system update. What do you investigate?
**Answer:** Input schema, units, missing values, feature distributions, source-system changes and threshold configuration before assuming customer behaviour changed.

### Q10. Why should flagged records not be deleted automatically?
**Answer:** They may be valid rare events. Review them and apply a documented treatment policy.

**Senior developer principle:** An anomaly detector produces investigation candidates, not final fraud or error decisions.
