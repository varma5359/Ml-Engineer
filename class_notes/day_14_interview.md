# Day 14: Interview Questions — EDA, Distributions and Feature Relationships

## Beginner questions

### Q1. What is eda, distributions and feature relationships?
**Answer:** Exploratory Data Analysis is the structured inspection of data quality, distributions, segments and relationships before modelling.

### Q2. Why is it important in an ML workflow?
**Answer:** It improves the reliability of data preparation, training or evaluation. EDA reveals patterns and questions; it does not prove causation. Check sample sizes, missingness, units and time periods.

### Q3. What syntax or method would you use?
**Answer:** Use describe(), value_counts(), groupby(), histograms, box plots, scatter plots and category counts. Choose each chart to answer a specific question. I would explain the input, parameters, output and validation rather than reciting a function name.

## Practical questions

### Q4. How do you know the operation worked correctly?
**Answer:** Test a small known example, compare expected and actual output, inspect boundary cases and verify that the result still makes sense in the business domain.

### Q5. What beginner mistake should you avoid?
**Answer:** Applying a technique without checking assumptions or the meaning of the data. I would inspect source definitions, data types, units, missingness and output before using it downstream.

## Scenario-based interview questions

### Q6. An online store sees returns rise. Compare return rate by category, not just raw return count, because categories have different order volumes. Investigate before claiming a cause.
**Answer:** I would first clarify the business goal and gather examples of the issue. Then I would inspect source data, assumptions and the point in time at which each field becomes available. I would reproduce the problem on a small sample, test likely causes one at a time, select a correction based on evidence and validate it on held-out data. I would document limitations and monitor the outcome after release.

### Q7. The aggregate metric is acceptable, but one important customer segment performs poorly. What do you do?
**Answer:** I would calculate metrics and inspect data quality by segment, verify sample sizes, investigate whether the segment differs in behaviour or data capture, and discuss the error trade-off with the business owner. I would not hide the issue behind an overall average.

### Q8. A teammate suggests an immediate fix without investigating the cause. How do you respond?
**Answer:** I would ask for evidence and reproduce the failure. A quick change can mask the root cause or create a new defect. I would compare hypotheses, make the smallest safe change and add a regression test.

## Senior developer questions

### Q9. How do you choose between a simple and complex solution?
**Answer:** Compare correctness, maintainability, latency, cost, explainability, data requirements and operational risk. Start with a baseline and add complexity only when repeatable validation and business impact justify it.

### Q10. What would you monitor after deployment?
**Answer:** Input quality, missing and unknown values, output distributions, latency, errors, delayed ground-truth metrics and segment-level outcomes. Define owners and alert thresholds before release.

## Answer framework
1. Clarify the decision and constraints.
2. Inspect the data and assumptions.
3. Diagnose using evidence.
4. Choose the smallest safe correction.
5. Validate normal and edge cases.
6. Communicate trade-offs and limitations.
7. Monitor after release.


---

## Detailed Scenario and Technical Interview Expansion

### Scenario: sales fell in one region
A sales dashboard shows Region B has lower revenue. A junior analyst concludes the regional team is underperforming.

**Senior engineer response:** Check the number of active stores, number of orders, average order value, product availability, reporting completeness and time period. Compare rates and per-store metrics, not just totals. Look for missing transactions and seasonality. Present evidence and alternative explanations before recommending action.

### Scenario questions
**Q1. What is the first thing you do when receiving an unfamiliar dataset?**  
Confirm row grain, business meaning, shape, columns, types, missingness, duplicates and target definition.

**Q2. Why use a histogram instead of a bar chart for a continuous numeric variable?**  
A histogram groups numeric values into intervals and shows a distribution. A bar chart usually compares discrete categories.

**Q3. A region has the highest number of returns. Does that mean it has the worst return rate?**  
Not necessarily. Divide returns by orders or units sold and compare denominators.

**Q4. Can EDA cause data leakage?**  
Yes, if test outcomes guide feature selection or preprocessing decisions. Exploratory analysis should respect the evaluation protocol, especially when choices are tuned based on labels.

**Q5. How do you communicate EDA to a business stakeholder?**  
State the question, show a correctly labelled chart, quantify the pattern, explain uncertainty and distinguish observed association from proven cause.
