# Day 08: Interview Questions — Statistics, Probability and Distributions

**Pathway:** Machine Learning Engineer  
**Focus:** Statistical interpretation, uncertainty and business data scenarios

## 1. Core concept questions

### Q1. What is the difference between mean and median?

**Answer:** Mean is the arithmetic average. Median is the middle value after sorting. Median is less affected by extreme values and can better describe a typical observation in a highly skewed dataset.

**Scenario:** Salaries are 20,000, 22,000, 24,000, 25,000 and 200,000. Which centre may better represent a typical employee?

**Answer:** The median, because the unusually high salary pulls the mean upward.

**Follow-up:** Is the mean always wrong when data is skewed?

**Answer:** No. It remains a valid arithmetic average. It may simply be less representative of a typical observation.

### Q2. What are variance and standard deviation?

**Answer:** They describe how spread out observations are around the mean. Standard deviation is expressed in the original units, which often makes it easier to explain to business stakeholders.

**Follow-up:** If standard deviation is high, what does it tell us?

**Answer:** The observations are more spread out around the mean. Whether this is a problem depends on the business context and expected variation.

### Q3. Why can sample and population standard deviation differ?

**Answer:** They use different statistical conventions because a sample is being used to estimate variation in a larger population, while a population calculation describes the complete group being studied. Python libraries may expose separate functions or configuration options for these conventions.

**Follow-up:** Why should you check a library's default?

**Answer:** Different defaults can produce different reported values. The engineer should use the convention that matches the analysis and document it.

### Q4. What is probability?

**Answer:** Probability is a numerical description of how likely an event is under defined conditions. It can be based on a mathematical model or estimated from observed data.

**Follow-up:** Does an observed 20% cancellation rate guarantee 20% of next month's orders will cancel?

**Answer:** No. It is a historical estimate. Future outcomes can differ because customer behaviour and operating conditions may change.

### Q5. What is conditional probability?

**Answer:** It describes the likelihood of an event when another condition is already known. For example, the repeat-purchase rate among premium customers is conditional on the customer being in the premium group.

**Follow-up:** Is that the same as the overall repeat-purchase rate?

**Answer:** No. The overall rate uses all customers, while the conditional rate uses only the specified group.

### Q6. What is a probability distribution?

**Answer:** A distribution describes how values or outcomes occur across observations. It helps us understand common values, spread, skewness and the frequency of unusual outcomes.

**Follow-up:** Should an engineer assume every numerical feature follows a normal distribution?

**Answer:** No. Inspect the data and assess whether a normal distribution is a reasonable description.

## 2. ML and business scenarios

### Q7. A dataset contains a few very high transaction amounts while most values are small. Should you remove the high values?

**Answer:** Not automatically. Check whether they are data-entry errors, valid large transactions, fraud events or a different customer segment. Document the decision and consider the effect on model performance.

**Follow-up:** Which measure of centre may be less affected by extreme values?

**Answer:** Median.

### Q8. Two delivery centres have the same average delivery time, but one has much less consistent performance. How would you explain this?

**Answer:** Compare a measure of spread, such as standard deviation, alongside the average. The centre with greater variation has less predictable delivery times even though the averages are equal.

**Follow-up:** Is lower variation always better?

**Answer:** Not necessarily. It depends on the desired operating range, customer needs and whether the values are accurate.

### Q9. A churn model returns a probability score of 0.90. Does this guarantee the customer will leave?

**Answer:** No. It is a model-estimated probability or score. Its reliability depends on model quality, calibration and whether the current population resembles the data used for evaluation.

**Follow-up:** How can you assess calibration?

**Answer:** Compare predicted probabilities with observed outcome frequencies on held-out data, for example using calibration curves or suitable calibration metrics.

### Q10. Premium customers have a higher repeat-purchase rate than standard customers. Can you conclude that premium membership causes repeat purchases?

**Answer:** No. The difference is an association. Premium customers may differ in income, purchase history, engagement or other factors. A suitable study design is needed to make causal claims.

**Follow-up:** Can the pattern still be useful for prediction?

**Answer:** It may be useful if the feature is available at prediction time, is not leaking the target and remains reliable on new data.

### Q11. A historical fraud rate is very low. Why might accuracy be misleading for a fraud model?

**Answer:** A model that predicts every transaction as non-fraudulent may achieve high accuracy when fraud is rare, while detecting no fraud at all. The team should inspect metrics that reflect the rare class and the cost of missed fraud and false alerts.

**Follow-up:** Which metrics might you examine?

**Answer:** Precision, recall, F1-score, precision-recall curves and business costs. Detailed metric selection is covered later in the pathway.

### Q12. A business report compares repeat-purchase percentages across two customer groups with very different sizes. What should you check?

**Answer:** Check the numerator and denominator for each group, the sample sizes, the time window and the customer definition. Percentages should be calculated consistently and interpreted with their sample sizes.

**Follow-up:** Why are counts useful alongside rates?

**Answer:** A rate based on a small group may be unstable. Counts provide context for how much data supports the comparison.

## 3. Code interpretation questions

### Q13. What is the difference between these functions?

```python
import statistics
statistics.pvariance(values)
statistics.variance(values)
```

**Answer:** The first calculates population variance; the second calculates sample variance.

**Follow-up:** Why should you not compare them without checking the convention?

**Answer:** They answer related but different statistical questions and can return different values.

### Q14. What does this code calculate?

```python
orders = 100
cancelled = 12
rate = cancelled / orders
print(rate)
```

**Answer:** It calculates the observed cancellation proportion, 0.12, or 12%.

**Follow-up:** Is that a prediction for the next 100 orders?

**Answer:** Not by itself. It is a historical rate from the given observations.

### Q15. What is the meaning of this code?

```python
premium_rate = 30 / 100
standard_rate = 40 / 200
```

**Answer:** It calculates repeat-purchase proportions separately for premium and standard groups.

**Follow-up:** What should you verify before comparing the rates?

**Answer:** The group definitions, time window, event definition and sample sizes must be consistent.

## 4. Interview answer framework

When explaining a statistical result:

1. State what observations and group the result describes.
2. Explain the metric in plain language.
3. Mention the denominator and sample size where relevant.
4. Identify extreme values or distribution patterns.
5. Distinguish association from causation.
6. Explain uncertainty and limitations.
7. Connect the finding to the actual business decision.

**Senior developer principle:** A statistical value is not a business conclusion by itself. Explain what data it summarizes, what assumptions it depends on and what it does not prove.
