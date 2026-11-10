# Day 11: Interview Questions — Outlier Detection and Treatment

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is an outlier?
**Answer:** An observation unusually far from the general pattern. It may be a data error, a valid rare event or a different population segment.
**Follow-up:** Is every outlier an error?  
**Answer:** No. Statistical rules flag values for investigation; they do not know business validity.

### Q2. Explain the IQR method.
**Answer:** IQR is the spread between the first and third quartiles. A common fence-based rule flags unusually low or high values for review.
**Follow-up:** Is the IQR fence a business limit?  
**Answer:** No. It is a statistical screening rule.

### Q3. What is a z-score?
**Answer:** It describes how far a value is from the mean in standard-deviation units. It can be useful when the distribution is reasonably compatible with the method's assumptions.
**Follow-up:** Why can it be sensitive to outliers?  
**Answer:** The mean and standard deviation used to calculate it can themselves be affected by extreme observations.

### Q4. What is the difference between retaining, capping and removing an outlier?
**Answer:** Retaining keeps a value, capping limits its magnitude and removing excludes a record. Each action changes the information available to the model.
**Follow-up:** What should you preserve before capping?  
**Answer:** Original values, learned boundaries and an audit of affected records.

## Scenario questions

### Q5. A transaction is 100 times larger than usual. Do you delete it?
**Answer:** No. Verify whether it is a bulk order, premium transaction, currency issue, aggregation error or data-entry problem. Decide only after checking the source and target population.
**Follow-up:** What if it is a valid rare event?  
**Answer:** Retain it if it belongs to the task's population, and evaluate whether a robust method or transformation is suitable.

### Q6. The IQR method flags 8% of transactions. Is that too many?
**Answer:** Not necessarily. Inspect the flagged records, segments, source validity and business process. Report the count and percentage with the treatment decision.
**Follow-up:** Why include segment analysis?  
**Answer:** A group of legitimate high-value customers may be incorrectly treated as anomalous when analysed together with all customers.

### Q7. A box plot marks several points beyond its whiskers. Should they be removed?
**Answer:** Not automatically. Whiskers are statistical conventions. Inspect original records and domain rules.
**Follow-up:** What else can help?  
**Answer:** Histograms, sorted values and group-level summaries.

### Q8. Why must learned outlier thresholds use training data only?
**Answer:** If validation or test observations contribute to the threshold, held-out data influences preprocessing. Learn thresholds on training data and apply them unchanged to held-out data.
**Follow-up:** How can this be implemented consistently?  
**Answer:** Include the transformation in a fitted preprocessing workflow or pipeline.

### Q9. Your model performs poorly for high-value customers. What do you investigate?
**Answer:** Compare sample sizes, feature distributions, data quality and prediction errors for that segment. Check whether the segment is underrepresented or behaves differently.
**Follow-up:** Why isn't the overall metric enough?  
**Answer:** Aggregate performance can hide a concentrated failure in an important segment.

## Code interpretation

### Q10. What does this code calculate?
```python
q1 = series.quantile(0.25)
q3 = series.quantile(0.75)
iqr = q3 - q1
```
**Answer:** The first quartile, third quartile and interquartile range.
**Follow-up:** Does it explain why a value is unusual?  
**Answer:** No. The source and business context must be investigated.

### Q11. What does clip(upper=100) do?
**Answer:** It limits values above 100 to 100 and leaves smaller values unchanged.
**Follow-up:** Why keep the original value?  
**Answer:** To audit the transformation and understand how much information was changed.

### Q12. What should you check before applying log()?
**Answer:** Check for zero and negative values, the feature's meaning and the transformation's suitability.
**Follow-up:** Why?  
**Answer:** The ordinary real-valued logarithm is not defined for zero or negative values.

## Interview approach
1. Identify unusual observations.
2. Check source and business meaning.
3. Compare affected segments.
4. Choose a justified treatment.
5. Fit learned thresholds on training data only.
6. Preserve originals and record changes.
7. Evaluate impact on overall and segment-level performance.

**Senior developer principle:** Outlier detection is an investigation step, not a delete command.
