# Day 08: Interview Questions — Mean, Median, Variance, Standard Deviation, Probability and Distributions

**Pathway:** Machine Learning Engineer

---

## Question 1: What is the difference between mean and median?

**Answer:** Mean is the sum divided by the number of values. Median is the middle value after sorting. Median is less sensitive to extreme values.

**Scenario:** Salaries are 20k, 22k, 24k, 25k and 200k. Which better represents a typical employee salary?

**Answer:** Median (24k) is more representative of the middle employee because the 200k value pulls the mean upward.

**Follow-up:** Is the mean always wrong when data is skewed?

**Answer:** No. It remains a mathematically valid arithmetic average. It may simply be less representative of a typical observation.

## Question 2: What is variance and standard deviation?

**Answer:** Variance measures average squared deviation from the mean. Standard deviation is the square root of variance and expresses spread in the original units.

**Follow-up:** If standard deviation is high, what does it tell us?

**Answer:** Values are more spread out around the mean, but we need domain context to determine whether this is problematic.

## Question 3: Why do sample variance and population variance use different denominators?

**Answer:** Population variance divides by `n`. The common unbiased sample-variance estimator divides by `n-1` to account for estimating the mean from the sample.

**Follow-up:** How do you calculate sample variance in NumPy?

**Answer:** `np.var(values, ddof=1)`.

## Question 4: What is probability?

**Answer:** Probability is a numerical measure of how likely an event is, from 0 to 1. It can be based on a mathematical model or estimated from observed data.

**Follow-up:** Does an observed 20% rate mean exactly 20% will happen next month?

**Answer:** No. It is a historical estimate. Future outcomes can differ, and the estimate has uncertainty.

## Question 5: What is conditional probability?

**Answer:** It is the probability of an event given that another event has occurred.

**Example:** If 12 of 40 discount users return a product, the return rate among discount users is 30%.

**Follow-up:** Is that the same as the overall return rate?

**Answer:** No. The overall rate uses all customers, including those who did not use a discount.

## Question 6: Explain normal distribution and the 68–95–99.7 rule.

**Answer:** A normal distribution is symmetric and bell-shaped. Approximately 68%, 95% and 99.7% of observations lie within one, two and three standard deviations from the mean, respectively, when the distribution is normal.

**Follow-up:** Can we apply this rule to any dataset?

**Answer:** No. We should first assess whether a normal model is a reasonable fit.

## Question 7: A dataset has one delivery time of 300 minutes while most are around 30 minutes. Should you delete it?

**Answer:** Not automatically. Verify whether it is an entry error, a valid extreme event or a special operating condition. Investigate the record and document the treatment.

**Follow-up:** Which summary statistic is more robust here?

**Answer:** Median is more robust than mean to an extreme observation.

## Question 8: A churn model predicts 0.90 probability. Does this mean the customer has a 90% guaranteed chance of leaving?

**Answer:** No. It is a model-estimated score. Its probability interpretation depends on calibration and whether the model generalizes to the current customer population.

**Follow-up:** How do you check calibration?

**Answer:** Compare predicted probabilities with observed outcome frequencies across probability bins or use calibration curves and appropriate calibration metrics on held-out data.

## Question 9: What is the difference between correlation and causation?

**Answer:** Correlation describes association or co-movement. Causation means a change in one factor produces a change in another under a defined causal interpretation. Correlation alone does not prove causation.

**Follow-up:** Can a highly correlated feature be useful for prediction?

**Answer:** Yes, it may be predictive, provided it is available at prediction time, is not leaking the target and remains reliable. Predictive usefulness does not prove causal influence.