# Day 08: Mean, Median, Variance, Standard Deviation, Probability and Distributions

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

- Calculate and interpret mean and median.
- Understand variance and standard deviation as measures of spread.
- Recognize how outliers affect summary statistics.
- Explain basic probability and conditional probability.
- Understand common distributions used in ML.
- Connect statistical thinking to model evaluation and business decisions.

## 2. Why Statistics Matters in ML

Statistics helps engineers understand data before training, summarize patterns, reason about uncertainty and evaluate whether model results are meaningful.

Examples:

- Mean and median summarize typical customer spending.
- Standard deviation describes how much values vary.
- Probability expresses uncertainty about an event.
- Distributions describe how values are spread across possible outcomes.

## 3. Mean (Average)

The mean is the sum of all values divided by the number of values.

Example monthly spends: `[100, 200, 300, 400]`

- Sum = 1,000
- Number of values = 4
- Mean = 1,000 / 4 = 250

```python
import numpy as np

spend = np.array([100, 200, 300, 400])
print(np.mean(spend))
```

Output: `250.0`

**When useful:** The mean is useful when values are reasonably balanced and extreme values do not dominate the summary.

## 4. Median

The median is the middle value after sorting the data. For an even number of values, it is the average of the two middle values.

Example:

- Values: `[100, 200, 300, 400, 500]`
- Median = 300

Even example:

- Values: `[100, 200, 300, 400]`
- Median = (200 + 300) / 2 = 250

```python
values = np.array([100, 200, 300, 400, 500])
print(np.median(values))
```

Output: `300.0`

### Mean vs median when an outlier exists

Monthly customer spend:

`[100, 120, 130, 150, 1500]`

- Mean = 400
- Median = 130

The one very large value pulls the mean upward. The median better represents the middle customer in this example.

| Mean | Median |
|---|---|
| Uses every value in the sum | Depends on the middle position(s) |
| Sensitive to extreme values | More resistant to extreme values |
| Useful for arithmetic averages | Useful for skewed distributions |

## 5. Variance

Variance measures the average squared distance of values from the mean. It describes spread.

For a population, variance is the mean of squared deviations from the population mean. For a sample, the usual sample variance divides by `n - 1` rather than `n`.

Example values: `[2, 4, 6]`

- Mean = 4
- Deviations = -2, 0, 2
- Squared deviations = 4, 0, 4
- Population variance = (4 + 0 + 4) / 3 = 2.667

```python
values = np.array([2, 4, 6])
print(np.var(values))          # Population variance, ddof=0
print(np.var(values, ddof=1)) # Sample variance
```

Output approximately:

```text
2.6667
4.0
```

The sample calculation divides by 2, giving 4. Use the version that matches whether your data is treated as a complete population or a sample used to estimate a larger population.

## 6. Standard Deviation

Standard deviation is the square root of variance. It expresses spread in the same units as the original data.

For `[2, 4, 6]`, population standard deviation is approximately 1.633.

```python
print(np.std(np.array([2, 4, 6])))
```

**Output:** Approximately `1.63299`.

Interpretation: a larger standard deviation indicates greater spread around the mean. It does not by itself tell us whether values are good or bad.

## 7. Percentiles and Quartiles

A percentile tells us the value below which a specified percentage of observations falls.

- 25th percentile (Q1): approximately the lower quartile.
- 50th percentile: median.
- 75th percentile (Q3): approximately the upper quartile.

```python
values = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print(np.percentile(values, [25, 50, 75]))
```

NumPy uses a specified percentile calculation method (default interpolation method); different tools or conventions may produce slightly different quartile values on small datasets.

## 8. Probability Basics

Probability measures how likely an event is, from 0 to 1 (or 0% to 100%).

- 0: impossible under the defined model.
- 1: certain under the defined model.
- 0.5: 50% likelihood.

For equally likely outcomes, probability of an event is:

```text
Number of favourable outcomes / Total number of possible outcomes
```

Example: A fair six-sided die has 6 equally likely outcomes. Probability of rolling a 4 is 1/6.

### Empirical probability

If 20 out of 100 historical deliveries were late, the observed late-delivery rate is 20/100 = 0.20 or 20%. This historical rate is an estimate and may not equal the future rate.

## 9. Conditional Probability

Conditional probability is the probability of an event given that another event has occurred.

Example:

Among 100 customers, 40 used a discount code. Of those 40, 12 returned the product.

Probability of return given discount use = 12/40 = 0.30 or 30%.

This is not the same as the overall return rate, which needs the total number of returns across all customers.

## 10. Probability in Machine Learning

A binary classifier may output an estimated probability for the positive class.

Example:

```text
Customer A: estimated churn probability = 0.82
Customer B: estimated churn probability = 0.18
```

These values represent model estimates, not guarantees. Whether they can be interpreted as well-calibrated probabilities should be tested with calibration evaluation.

A decision threshold converts scores into labels. For example, with a threshold of 0.50, score 0.82 is classified positive and 0.18 negative. A different threshold may be appropriate when missing a churner is expensive or outreach capacity is limited.

## 11. Common Distributions

A probability distribution describes possible values and how probability is assigned to them.

| Distribution | Main idea | Example |
|---|---|---|
| Bernoulli | One trial with two outcomes | Customer churns or does not churn |
| Binomial | Number of successes across fixed independent trials with the same success probability | Number of successful deliveries in 20 trials under simplified assumptions |
| Normal | Symmetric, bell-shaped continuous distribution | Measurement errors may be approximately normal in some settings |
| Uniform | Values in a defined interval have equal density/probability | Random number generated uniformly between 0 and 1 |
| Poisson | Count of events in a fixed interval under assumptions | Calls arriving per minute in a simplified model |

### Normal distribution intuition

A normal distribution is symmetric around its mean. Values near the mean are more common than values far away, when the data genuinely follows this distribution.

<Note: The 68–95–99.7 rule applies to a normal distribution: approximately 68% of values lie within 1 standard deviation of the mean, 95% within 2, and 99.7% within 3. This rule should not be assumed for every dataset.>

## 12. Correlation Is Not Causation

Two variables may move together without one causing the other.

Example: Ice-cream sales and swimming incidents may both increase during hot weather. Temperature may influence both. It would be incorrect to conclude that ice-cream sales cause swimming incidents.

In ML, a feature can help predict a target without causing it. Prediction and causal explanation are different goals.

## 13. Industry Scenario — Delivery Delay Analysis

A logistics company records delivery times (minutes): `[28, 30, 31, 32, 35, 90]`.

- The mean is 41 minutes.
- The median is 31.5 minutes.
- The value 90 is unusually high relative to the others and pulls the mean upward.

An analyst should investigate the 90-minute delivery rather than remove it automatically. It may represent a data error, a special route or a genuine operational delay.

If the company predicts late delivery, it should define a late-delivery threshold, create labels consistently and evaluate false alerts versus missed delays.

## 14. Common Mistakes

- Using mean alone for heavily skewed data.
- Confusing population and sample variance.
- Saying standard deviation measures accuracy; it measures spread.
- Treating an observed historical rate as a guaranteed future probability.
- Assuming all data follows a normal distribution.
- Treating correlation as proof of cause.
- Treating a model score as a certainty without calibration and validation.

## 15. Quick Revision

- Mean = arithmetic average; median = middle value.
- Variance measures average squared spread; standard deviation is its square root.
- Probability quantifies uncertainty; conditional probability adds a condition.
- Distributions describe how values or events are spread.
- Statistical summaries must be interpreted in the context of the data and business question.

---

## Beginner-to-Developer Expansion: More Examples and Real-World Reasoning

### A. Start with the simplest meaning
Before memorizing a function, explain the idea in your own words. Identify the input, output and the reason the operation exists. A correct syntax call on the wrong data is still a wrong solution.

### B. Additional practical scenario
Monthly salaries may be 25k, 28k, 30k, 32k and 500k. The mean is pulled upward by the extreme salary; the median better represents the middle employee in this skewed example.

### C. Syntax and use: how to choose
Mean is sum/count; median is the middle after sorting; variance measures squared spread; standard deviation is the square root of variance. Probability describes uncertainty, not certainty.

### D. What happens if this is skipped?
If a team reports only the mean, it can misrepresent a skewed population. Always inspect distribution and sample size.

### E. Senior developer review checklist
- Is the business definition clear?
- Are input types, units, shape and missing values checked?
- Can the result be verified on a small hand-checkable example?
- Have normal, boundary and invalid cases been considered?
- Is the approach simple enough to explain and maintain?
- Are limitations and possible failure costs documented?

### F. Explain it in an interview
Use this order: definition → why it is needed → syntax or method → small example → real business scenario → what can go wrong → how you validate the result. Avoid answering with only a library function name.
