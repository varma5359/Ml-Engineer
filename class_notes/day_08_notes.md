# Day 08: Mean, Median, Variance, Standard Deviation, Probability and Distributions

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate

## Learning Objectives
- Calculate and interpret mean and median.
- Explain variance and standard deviation.
- Understand how outliers affect averages.
- Define probability and conditional probability.
- Recognize common probability distributions.
- Connect statistics with ML data analysis.

## 1. Why Statistics Matters in ML
Statistics helps us summarize data, understand variation, reason about uncertainty and evaluate patterns. Without it, developers may misinterpret averages, overlook unusual observations or make claims unsupported by data.

## 2. Mean
**Definition:** Mean is the arithmetic average: add all values and divide by the number of values.

**Formula in words:** sum of values / count of values.

```python
values = [10, 20, 30, 40]
print(sum(values) / len(values))
```
Output: 25.0.

**Three examples**
1. Average daily sales: [100, 200, 300] → 200.
2. Average marks: [60, 70, 80, 90] → 75.
3. Average delivery time: [20, 25, 30] → 25 minutes.

## 3. Median
**Definition:** Median is the middle value after sorting. For an even number of values, it is the average of the two middle values.

```python
import statistics
print(statistics.median([10, 20, 30]))
print(statistics.median([10, 20, 30, 40]))
```
Output: 20 and 25.

**Why use it?** Median is less affected by extreme values than mean.

Example: salaries [25, 30, 32, 35, 500] (in thousands). The mean is pulled upward by 500, while the median is 32.

## 4. Variance
**Definition:** Variance measures how spread out values are around the mean. It is the average squared deviation under the chosen population or sample convention.

```python
values = [2, 4, 6]
print(statistics.pvariance(values))
print(statistics.variance(values))
```
Population variance and sample variance use different denominators; choose the one appropriate to the question.

## 5. Standard Deviation
**Definition:** Standard deviation is the square root of variance. It measures spread in the same units as the original data.

```python
values = [2, 4, 6]
print(statistics.pstdev(values))
print(statistics.stdev(values))
```

**Three interpretations**
1. Low standard deviation: values are relatively close to the mean.
2. High standard deviation: values are more spread out.
3. Same mean but different standard deviation: two groups can have the same average but very different consistency.

## 6. Probability
**Definition:** Probability measures how likely an event is, from 0 (impossible) to 1 (certain).

For equally likely outcomes: probability = favourable outcomes / total possible outcomes.

**Examples**
1. Fair coin: probability of heads = 1/2 = 0.5.
2. Fair six-sided die: probability of rolling 6 = 1/6.
3. If 20 out of 100 orders are returned, observed return proportion = 0.20.

Observed proportions are estimates from data, not guarantees about the next event.

## 7. Conditional Probability
**Definition:** Conditional probability is the probability of event A when event B is already known to have occurred.

Notation in words: P(A given B) = P(A and B) / P(B), when P(B) > 0.

**Example:** Among premium customers, 30 of 100 made a repeat purchase. The observed repeat-purchase rate conditional on being premium is 30/100 = 0.30.

## 8. Common Distributions
| Distribution | Meaning | Example |
|---|---|---|
| Normal | Continuous, bell-shaped distribution | Measurement variation around a typical value |
| Binomial | Number of successes in fixed independent trials with same success probability | Number of successful deliveries in 10 trials |
| Poisson | Count of events in a fixed interval under suitable assumptions | Calls arriving per minute |
| Uniform | Outcomes have equal density/probability over a defined range | Idealized random number over an interval |

Do not assume real business data follows a normal distribution. Inspect it.

## 9. Scenario Questions
1. A salary dataset has a few very high salaries. Which centre may better represent a typical employee? Median.
2. Two stores have the same average sales but one has much more variation. Which measure helps compare consistency? Standard deviation.
3. A fraud model predicts a probability of 0.8. Does it guarantee fraud? No; it is a probability estimate that must be evaluated and interpreted.
