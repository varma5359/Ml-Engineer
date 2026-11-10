# Day 08: Mean, Median, Variance, Standard Deviation, Probability and Distributions

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, Mathematics and Statistics Foundations  
**Level:** Beginner  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

By the end of this lesson, you should be able to:

- Explain why statistics is important in machine learning.
- Interpret mean and median in business datasets.
- Explain variance and standard deviation as measures of spread.
- Recognize the effect of extreme values on summary statistics.
- Explain probability and conditional probability in plain language.
- Recognize common distributions and their practical examples.
- Distinguish historical rates from guaranteed future outcomes.
- Use simple Python tools to inspect statistical summaries.

## 2. Why does an ML Engineer need statistics?

Machine learning learns patterns from data. Before trusting those patterns, an engineer needs to understand what the data looks like.

Statistics helps answer practical questions:

- What is a typical customer spend?
- Are delivery times consistent or highly variable?
- Are a few extreme values affecting the average?
- How frequently does a particular outcome occur?
- Does a predicted probability represent a meaningful likelihood?
- Does the current dataset look different from previous data?

Statistics is not only about calculating numbers. It is about interpreting what those numbers mean and identifying when they may be misleading.

## 3. Mean — the arithmetic average

The **mean** is the total of all observed values divided by the number of values.

### Business example

A delivery company records delivery times of 20, 25, 30, 35 and 40 minutes. The mean provides one summary of the average delivery time in this sample.

### 3.1 Calculate mean using Python

```python
delivery_times = [20, 25, 30, 35, 40]

average_time = sum(delivery_times) / len(delivery_times)

print(average_time)
```

**Expected output**

```text
30.0
```

**Explanation**

- `sum()` adds the values.
- `len()` counts the observations.
- Dividing the total by the count produces the mean.

### 3.2 Calculate mean using statistics

```python
import statistics

delivery_times = [20, 25, 30, 35, 40]

print(statistics.mean(delivery_times))
```

**Expected output**

```text
30
```

**Explanation**

The standard-library `statistics.mean()` function calculates the arithmetic mean.

### 3.3 Calculate mean using NumPy

```python
import numpy as np

delivery_times = np.array([20, 25, 30, 35, 40])

print(np.mean(delivery_times))
```

**Expected output**

```text
30.0
```

**Explanation**

NumPy is convenient when working with numerical arrays and larger datasets.

## 4. Median — the middle observation

The **median** is the middle value after observations are arranged in order.

If there is an even number of observations, the median is the midpoint between the two central values.

The median is useful when a dataset contains extreme values that pull the mean away from what is typical for most observations.

### 4.1 Calculate median with an odd number of observations

```python
import statistics

delivery_times = [20, 25, 30, 35, 40]

print(statistics.median(delivery_times))
```

**Expected output**

```text
30
```

**Explanation**

There are five ordered values. The middle observation is 30.

### 4.2 Calculate median with an even number of observations

```python
delivery_times = [20, 25, 30, 35]

print(statistics.median(delivery_times))
```

**Expected output**

```text
27.5
```

**Explanation**

There are four observations. The median lies halfway between the two middle values, 25 and 30.

### 4.3 Compare mean and median when there is an extreme value

```python
delivery_times = [20, 25, 30, 35, 180]

print("Mean:", statistics.mean(delivery_times))
print("Median:", statistics.median(delivery_times))
```

**Expected output**

```text
Mean: 58
Median: 30
```

**Explanation**

The unusually long delivery time pulls the mean upward. The median remains at the middle observation. Neither value is automatically the correct business answer; inspect why the 180-minute delivery occurred.

### 4.4 Compare salaries

```python
salaries = [25000, 28000, 30000, 32000, 250000]

print("Mean salary:", statistics.mean(salaries))
print("Median salary:", statistics.median(salaries))
```

**Expected output**

```text
Mean salary: 73000
Median salary: 30000
```

**Explanation**

A small number of high salaries can make the mean much higher than the salary received by the middle employee. When describing a typical employee salary, median may be more informative.

## 5. Mean versus median — when should you use each?

| Situation | Useful summary | Reason |
|---|---|---|
| Values are fairly balanced with no extreme observations | Mean and median | Both can describe the centre |
| Strongly skewed income or transaction values | Median | Less affected by extreme values |
| Average total workload or cost is required | Mean may be useful | It reflects the total divided across observations |
| Typical customer waiting time | Median may be useful | It describes the middle observation |
| A dataset contains data-entry errors | Neither alone is enough | Investigate and validate the errors |

**Important:** Do not delete an extreme value just because it changes the mean. Determine whether it is invalid, rare but genuine or evidence of a different group.

## 6. Variance — how spread out are values?

**Variance** describes how widely values are spread around their mean.

- Low variance means observations are relatively close to the mean.
- High variance means observations are more spread out.
- Two groups can have the same mean but very different levels of consistency.

Variance uses squared deviations, so its units are squared compared with the original measurement. This is one reason standard deviation is often easier to explain.

### 6.1 Calculate population variance

```python
import statistics

values = [2, 4, 6]

print(statistics.pvariance(values))
```

**Expected output**

```text
2.6666666666666665
```

**Explanation**

`pvariance()` calculates population variance. Use it when the provided values represent the entire population being described.

### 6.2 Calculate sample variance

```python
values = [2, 4, 6]

print(statistics.variance(values))
```

**Expected output**

```text
4
```

**Explanation**

`variance()` calculates sample variance. It uses the sample convention rather than the population convention. Choose the function that matches the statistical question.

### 6.3 Calculate variance with NumPy

```python
import numpy as np

values = np.array([2, 4, 6])

print(np.var(values))
print(np.var(values, ddof=1))
```

**Expected output**

```text
2.6666666666666665
4.0
```

**Explanation**

- NumPy's default `np.var()` uses the population convention.
- Setting `ddof=1` requests the common sample-variance convention.
- Do not compare results from different tools without checking their defaults.

## 7. Standard deviation — spread in the original units

**Standard deviation** describes how spread out observations are around the mean. It is expressed in the same units as the original values.

For delivery times, standard deviation is measured in minutes. For salary, it is measured in the same currency unit as salary.

### 7.1 Calculate population standard deviation

```python
import statistics

delivery_times = [20, 25, 30, 35, 40]

print(statistics.pstdev(delivery_times))
```

**Expected output**

```text
7.0710678118654755
```

**Explanation**

`pstdev()` calculates population standard deviation. The result describes the spread of these five values around their mean.

### 7.2 Calculate sample standard deviation

```python
print(statistics.stdev(delivery_times))
```

**Expected output**

```text
7.905694150420948
```

**Explanation**

`stdev()` calculates sample standard deviation. It uses the sample convention and therefore differs from the population result.

### 7.3 Compare two groups with the same mean

```python
consistent_scores = [48, 49, 50, 51, 52]
variable_scores = [20, 35, 50, 65, 80]

print("Group A mean:", statistics.mean(consistent_scores))
print("Group A spread:", statistics.pstdev(consistent_scores))

print("Group B mean:", statistics.mean(variable_scores))
print("Group B spread:", statistics.pstdev(variable_scores))
```

**Expected output**

```text
Group A mean: 50
Group A spread: 1.4142135623730951
Group B mean: 50
Group B spread: 21.213203435596427
```

**Explanation**

Both groups have the same mean, but their values behave very differently. Group A is consistent; Group B has much greater variation.

### 7.4 Interpret standard deviation in a business context

Suppose two delivery centres both report an average delivery time of 30 minutes.

- Centre A has a low standard deviation: delivery times are generally close to 30 minutes.
- Centre B has a high standard deviation: delivery times vary much more.

A manager may care about both average speed and reliability. The average alone does not describe the full customer experience.

## 8. Probability — understanding uncertainty

**Probability** is a numerical way to describe how likely an event is. It ranges from zero, meaning impossible under the defined conditions, to one, meaning certain under those conditions.

Business examples:
- The chance that an order is cancelled.
- The likelihood that a payment fails.
- The proportion of customers who renew a subscription.
- The estimated likelihood that a transaction is fraudulent.

Probability does not guarantee the outcome for one particular customer or transaction.

### 8.1 Estimate an observed event rate

```python
total_orders = 100
cancelled_orders = 12

cancellation_rate = cancelled_orders / total_orders

print(cancellation_rate)
print(cancellation_rate * 100)
```

**Expected output**

```text
0.12
12.0
```

**Explanation**

In this historical sample, 12 out of 100 orders were cancelled. The observed cancellation rate is 12%. This is a historical proportion, not a guarantee that exactly 12% of future orders will be cancelled.

### 8.2 Count outcomes from a small dataset

```python
outcomes = ["success", "success", "failed", "success", "failed"]

success_count = outcomes.count("success")
total_count = len(outcomes)

print(success_count)
print(success_count / total_count)
```

**Expected output**

```text
3
0.6
```

**Explanation**

Three of the five payment attempts succeeded. The observed success proportion is 0.6 for this small sample. A larger and representative dataset is usually needed for a stable estimate.

### 8.3 Understand the probability scale

| Value | Plain-language meaning |
|---:|---|
| 0.00 | Impossible under the defined event |
| 0.25 | Low likelihood |
| 0.50 | Even likelihood |
| 0.75 | High likelihood |
| 1.00 | Certain under the defined conditions |

## 9. Events and sample space

An **outcome** is one possible result. A **sample space** is the set of outcomes being considered. An **event** is the outcome or group of outcomes relevant to a question.

Example: one online payment attempt may be recorded as:
- Successful.
- Failed.

For this simplified classification, the sample space contains these two possible final outcomes.

The business question might be: "What proportion of payment attempts fail?"

Before calculating a probability, define the event clearly. Ambiguous event definitions lead to misleading rates.

## 10. Complementary and mutually exclusive events

### Complementary events

A complementary event is the opposite of a defined event. If the categories cover all possible outcomes, the event and its complement together account for all cases.

Example:
- Delivery on time.
- Delivery not on time.

### Mutually exclusive events

Two events are mutually exclusive when they cannot happen at the same time for the same observation under the same definitions.

Example: a single payment attempt cannot have both a final status of successful and a final status of failed if the system's status definitions are exclusive.

### Do not confuse mutually exclusive and independent

| Mutually exclusive | Independent |
|---|---|
| Two events cannot occur together | One event does not change the likelihood of the other |
| Example: one payment attempt is either successful or failed | Example: two unrelated customers' payment outcomes may be independent |
| Occurrence of one rules out the other | Both events can occur together |

Independence is an assumption that needs context. Two events may share a common cause, such as a payment gateway outage, making their outcomes related.

## 11. Conditional probability — probability under a known condition

**Conditional probability** asks how likely an event is when another condition is already known.

Business example:

A retailer groups customers by membership type and checks repeat purchases.

| Membership | Customers | Repeat purchasers |
|---|---:|---:|
| Premium | 100 | 30 |
| Standard | 200 | 40 |

Among premium customers, 30 of 100 made a repeat purchase. The observed repeat-purchase rate within that group is 30%.

This is not the same as the overall repeat-purchase rate across all customers.

### 11.1 Calculate a conditional rate in Python

```python
premium_customers = 100
premium_repeat_purchases = 30

premium_repeat_rate = premium_repeat_purchases / premium_customers

print(premium_repeat_rate)
```

**Expected output**

```text
0.3
```

**Explanation**

The calculation uses only the premium customer group. Always state the group or condition when reporting a rate.

### 11.2 Compare conditional rates across groups

```python
premium_rate = 30 / 100
standard_rate = 40 / 200

print("Premium:", premium_rate)
print("Standard:", standard_rate)
```

**Expected output**

```text
Premium: 0.3
Standard: 0.2
```

**Explanation**

The observed repeat-purchase rate is higher in the premium group in this example. This does not prove membership caused the difference; other customer characteristics may explain it.

## 12. Common probability distributions

A **distribution** describes how values or outcomes occur across a dataset or process.

| Distribution | Plain-language idea | Example |
|---|---|---|
| Normal | Symmetric, bell-shaped pattern around a centre | Some measurement errors or natural measurements |
| Binomial | Count of successes across a fixed number of suitable trials | Number of successful deliveries in a set of attempts |
| Poisson | Count of events over a fixed interval under suitable assumptions | Support calls arriving per minute |
| Uniform | Values have equal likelihood or density across a defined range | An idealized random selection over a fixed interval |

Do not assume every business dataset follows a normal distribution. Inspect the actual values and context.

### 12.1 Create sample values from a normal distribution

```python
import numpy as np

rng = np.random.default_rng(42)
delivery_variation = rng.normal(loc=30, scale=5, size=10)

print(np.round(delivery_variation, 2))
```

**Expected output**

A repeatable list of ten decimal values, generated around the configured centre with the configured spread.

**Explanation**

- `default_rng(42)` creates a random number generator with a fixed seed for repeatability.
- `normal()` generates values from a normal distribution.
- `loc` is the distribution centre.
- `scale` is its standard deviation.
- `size` controls how many values are generated.

This is simulated data for learning, not a claim that real delivery times are normally distributed.

### 12.2 Inspect generated values

```python
print("Mean:", round(delivery_variation.mean(), 2))
print("Median:", round(np.median(delivery_variation), 2))
print("Standard deviation:", round(delivery_variation.std(), 2))
```

**Expected output**

Three summary values for the generated sample.

**Explanation**

A random sample's summaries will not necessarily equal the distribution settings exactly. Small samples can vary noticeably.

## 13. Why distributions matter in ML

Understanding distributions helps an engineer:

- Detect skewed features.
- Identify unusual values for investigation.
- Understand whether a feature is dominated by a small number of extreme observations.
- Compare training and production data.
- Select suitable summaries and visualizations.
- Interpret uncertainty and probability outputs more carefully.

A distribution alone does not tell us whether a feature is useful for prediction. It must be considered with the target, prediction time and business context.

## 14. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Reporting only the mean | Also inspect median and spread |
| Removing every extreme value | Investigate whether it is valid or an error |
| Mixing population and sample standard deviation | Check the function and statistical question |
| Treating historical rate as a guarantee | Explain that it is an estimate from observed data |
| Treating model probability as certainty | Validate calibration and model performance |
| Assuming all features are normally distributed | Inspect the real distribution |
| Comparing rates with different denominators | State the group and denominator |
| Claiming correlation or group differences prove causation | Consider other explanations and study design |

## 15. Interview-ready explanations

### What is the difference between mean and median?

Mean is the arithmetic average of the observations. Median is the middle value after sorting. Median is less affected by extreme values, so it can better describe a typical observation in a highly skewed dataset.

### What is standard deviation?

Standard deviation describes how spread out values are around the mean and uses the same units as the original data. A larger value indicates more variation, but whether that variation is acceptable depends on the business context.

### What is probability?

Probability describes the likelihood of an event under defined conditions. It may be calculated from a mathematical model or estimated from observed data. It does not guarantee an individual future outcome.

### What is conditional probability?

It is the likelihood of an event when another condition is already known. For example, a repeat-purchase rate among premium customers is conditional on membership being premium.

### Why should an ML Engineer inspect a feature distribution?

It can reveal skewness, extreme values, unusual patterns and possible changes between datasets. These findings guide investigation and preprocessing, but should not be interpreted without business context.

## 16. Scenario questions

1. A salary dataset contains a few executive salaries much higher than the rest. Which centre may better describe a typical employee?  
   **Answer:** Median may be more representative because the extreme values affect the mean more strongly.

2. Two delivery centres have the same average delivery time, but one has much more variation. What should you compare?  
   **Answer:** Compare standard deviation or another spread measure as well as the mean.

3. A model returns a probability score of 0.90. Does it guarantee the customer will churn?  
   **Answer:** No. It is a model estimate. Its reliability depends on model performance and calibration.

4. The historical cancellation rate is 12%. Can the business promise that exactly 12% of next month's orders will cancel?  
   **Answer:** No. The rate is an estimate based on historical observations. Future conditions may differ.

5. Premium customers have a higher repeat-purchase rate than standard customers. Does that prove premium membership causes repeat purchases?  
   **Answer:** No. Other differences between the groups may explain the association.

## 17. Lesson completion checklist

- [ ] I can explain mean and median.
- [ ] I can identify when an extreme value affects the mean.
- [ ] I can explain variance and standard deviation.
- [ ] I understand sample and population conventions.
- [ ] I can explain probability in business language.
- [ ] I can calculate a simple observed event rate.
- [ ] I can explain conditional probability.
- [ ] I can identify common probability distributions.
- [ ] I avoid treating probability estimates as guarantees.
- [ ] I can explain why statistical summaries need business context.

**Next lesson:** Day 09 — Dataset Structure, Features, Target Variables and Observation Units.
