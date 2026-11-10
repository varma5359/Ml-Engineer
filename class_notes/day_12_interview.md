# Day 12: Interview Questions — Encoding Categorical Variables and Scaling Numerical Data

## Beginner questions

### Q1. What is encoding categorical variables and scaling numerical data?
**Answer:** Encoding converts categories into numeric representations. Scaling changes numeric feature magnitudes so that they are comparable.

### Q2. Why is it important in an ML workflow?
**Answer:** It improves the reliability of data preparation, training or evaluation. Arbitrary category numbers invent order, while fitting a scaler on the full dataset leaks test information. Use a ColumnTransformer and Pipeline for consistent training/inference.

### Q3. What syntax or method would you use?
**Answer:** One-hot encode unordered categories; use ordinal encoding only when a real order exists. StandardScaler uses (x - mean) / standard deviation; MinMaxScaler maps values to a chosen range. Fit transformations on training data only. I would explain the input, parameters, output and validation rather than reciting a function name.

## Practical questions

### Q4. How do you know the operation worked correctly?
**Answer:** Test a small known example, compare expected and actual output, inspect boundary cases and verify that the result still makes sense in the business domain.

### Q5. What beginner mistake should you avoid?
**Answer:** Applying a technique without checking assumptions or the meaning of the data. I would inspect source definitions, data types, units, missingness and output before using it downstream.

## Scenario-based interview questions

### Q6. A loan dataset contains city, education, age and income. City has no natural ranking; education may have a defined order. KNN may be dominated by income unless numeric features are scaled.
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

### Scenario: customer risk model uses city, education and salary
A developer maps Hyderabad=1, Vizag=2 and Chennai=3. The model may interpret Chennai as “greater” than Hyderabad even though city has no numeric order. Salary is also measured in thousands while age is measured in years, so KNN distances may be dominated by salary.

**Senior engineer response:** One-hot encode city. Use ordinal encoding for education only if the category order is real and documented. Scale continuous numeric fields for distance-based models. Put both operations in a ColumnTransformer/Pipeline so training and inference use identical transformations.

### Scenario questions
**Q1. Why is label encoding city risky?**  
It assigns arbitrary numeric order to categories that have no natural ranking. One-hot encoding is often safer for nominal features.

**Q2. Does every model require scaling?**  
No. KNN, SVM and many gradient-based models are scale-sensitive. Decision trees and random forests generally do not require standard scaling.

**Q3. Why must a scaler be fitted only on training data?**  
The test set must simulate unseen future data. Learning its mean or range leaks information from the test distribution into training.

**Q4. What if a new city appears at prediction time?**  
Configure the encoder to handle unknown categories, for example OneHotEncoder(handle_unknown="ignore"), and monitor the new category frequency.

**Q5. Why not scale the entire dataset before splitting?**  
Because the scaling statistics would include validation/test observations and make the evaluation less independent.
