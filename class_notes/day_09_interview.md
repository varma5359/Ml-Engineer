# Day 09: Interview Questions — Dataset Structure, Features, Target Variables and Observation Units

**Pathway:** Machine Learning Engineer  
**Focus:** Dataset schema, row grain, target definition and leakage awareness

## 1. Core concept questions

### Q1. What is a dataset?

**Answer:** A dataset is an organized collection of records about a subject or process. In a tabular dataset, rows represent observations and columns represent attributes or fields.

**Follow-up:** Does 10,000 rows always mean 10,000 customers?

**Answer:** No. One row may represent a transaction, customer-month, product or event. The observation unit must be confirmed.

### Q2. What is a dataset schema?

**Answer:** A schema describes the expected structure of a dataset, including column names, data types, constraints, allowed values and sometimes relationships between tables.

**Follow-up:** Can a CSV load successfully even when its schema is wrong?

**Answer:** Yes. A numeric field may be parsed as text, an ID may lose leading zeros or a required column may be missing.

### Q3. What is an observation unit or row grain?

**Answer:** It defines what one row represents. Examples include one customer, one transaction, one customer per month or one product event.

**Follow-up:** Why is row grain important for splitting data?

**Answer:** If multiple rows belong to the same customer, a random row split may place that customer in both training and test sets. That can produce overly optimistic evaluation when the goal is to generalize to unseen customers.

### Q4. What is the difference between a feature, target and identifier?

**Answer:** A feature is input information used to make a prediction. A target is the outcome a supervised model learns to predict. An identifier distinguishes records or entities and is not automatically a useful feature.

**Follow-up:** Can a customer ID ever be used as a feature?

**Answer:** It may contain useful structured information in some tasks, but it must be justified and checked for leakage and spurious patterns. Do not include it automatically.

## 2. Practical schema questions

### Q5. How do you inspect a new dataset before modelling?

**Answer:** Check its source and version, preview rows, inspect shape, column names, data types and non-null counts, review summary statistics and category values, then confirm row grain, identifiers and target definition.

**Follow-up:** Is df.head() sufficient?

**Answer:** No. It displays only a small sample. Use structural and domain-specific checks.

### Q6. A dataset has shape (10,000, 15). What can you conclude?

**Answer:** It has 10,000 rows and 15 columns. We cannot conclude that it contains 10,000 unique customers or that any column is the target.

**Follow-up:** How would you find the number of unique customers?

**Answer:** Count distinct values in the customer identifier column, after confirming the column's meaning.

### Q7. A dataset has 50,000 rows but only 8,000 unique customer IDs. What might this mean?

**Answer:** It may contain multiple records per customer, such as transactions, customer-month snapshots or support events. I would inspect timestamps, event IDs and the data definition to establish row grain.

**Follow-up:** Should the repeated customer rows be removed?

**Answer:** Not automatically. They may be legitimate observations.

### Q8. An ID column contains "00123" but is loaded as integer 123. What is the risk?

**Answer:** Leading zeros may be lost, changing the identifier's exact representation and potentially breaking joins with other systems. Read identifiers as strings when their formatting matters.

**Follow-up:** Should IDs be converted to numeric for easier modelling?

**Answer:** Not merely for convenience. An ID is usually a label, not a measured quantity.

### Q9. How do you check whether a target column has unexpected labels?

**Answer:** Inspect unique values and frequency counts, including missing values. Compare them with the approved target definition and investigate any unexpected values with the data owner.

**Follow-up:** Should you silently map an unknown label to zero?

**Answer:** No. That could create incorrect ground-truth labels.

## 3. ML prediction-time scenarios

### Q10. You are predicting churn but the dataset includes cancellation_reason. Should you use it?

**Answer:** Usually not if the reason is recorded after cancellation. It is unavailable at prediction time and can reveal the target directly. Confirm the field's timing and exclude it if it is post-outcome information.

**Follow-up:** What if cancellation_reason is known before the prediction date?

**Answer:** Then assess its definition, availability and operational reliability. The decision depends on when and how it is captured.

### Q11. A target column contains 0, 1 and "pending". What do you do?

**Answer:** Confirm whether pending means an incomplete outcome, an unknown label or a legitimate third class. Define the target policy before training. Do not automatically convert it to zero.

**Follow-up:** Why is target quality important?

**Answer:** The model learns from target labels. Incorrect labels can teach the wrong outcome and distort evaluation.

### Q12. The team wants one row per customer, but the source contains monthly records. What would you do?

**Answer:** Clarify the prediction task and time horizon. Decide whether to aggregate historical records into a customer-level snapshot or build a customer-month prediction dataset. Define the aggregation window and avoid using future information.

**Follow-up:** Could aggregation cause information loss?

**Answer:** Yes. It may remove event sequence and timing information. Choose aggregation based on the business question.

### Q13. A model will predict whether a customer renews next month. What should the target definition include?

**Answer:** It should define what renewal means, which customer/subscription is being predicted, the exact prediction timestamp, the future outcome window and how cancellations, pauses or missing outcomes are handled.

**Follow-up:** Why is the prediction timestamp important?

**Answer:** It determines which information is available to the model and helps prevent future information from leaking into features.

## 4. Code interpretation questions

### Q14. What does df.shape return?

**Answer:** A tuple containing the number of rows and columns.

**Follow-up:** Does df.shape[0] return the number of columns?

**Answer:** No. It returns the number of rows. df.shape[1] returns the number of columns.

### Q15. What does df["customer_id"].nunique() return?

**Answer:** The number of distinct non-missing customer IDs by default.

**Follow-up:** How do you include missing values in the distinct count?

**Answer:** Use `nunique(dropna=False)`.

### Q16. What does this code check?

```python
required = ["age", "city", "churn"]
missing = [col for col in required if col not in df.columns]
```

**Answer:** It creates a list of required column names that are absent from the DataFrame.

**Follow-up:** What should the pipeline do if a required column is missing?

**Answer:** Stop with a clear schema-validation error rather than continuing with incomplete inputs.

## 5. Interview answer framework

1. Define what one row represents.
2. Confirm the schema and data types.
3. Identify the target, features and identifiers.
4. Define the prediction time and target horizon.
5. Check that features are available at prediction time.
6. Validate labels, row identity and uniqueness rules.
7. Document assumptions before preprocessing.

**Senior developer principle:** Dataset shape tells you how much data you have, not what that data means. Confirm the observation unit and business definitions first.
