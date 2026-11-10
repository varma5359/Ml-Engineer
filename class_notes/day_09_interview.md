# Day 09: Interview Questions — Dataset Structure, Features, Target Variables and Observation Units

**Pathway:** Machine Learning Engineer

---

## Question 1: What is the difference between a feature and a target variable?

**Answer:** Features are input variables used to make a prediction. The target is the outcome the supervised model learns to predict.

**Example:** For house-price prediction, area and bedrooms may be features; sale price is the target.

**Follow-up:** Can a dataset have no target column?

**Answer:** Yes. It may be intended for unsupervised learning, exploration or later labelling. Ordinary supervised training requires target values for training examples.

## Question 2: What is an observation unit and why is it important?

**Answer:** It defines what one row represents, such as one transaction, one customer or one customer-month snapshot. It affects aggregation, sampling, splitting and interpretation.

**Scenario:** Customer records appear once every month. What could go wrong with a random row split?

**Answer:** The same customer's records may enter both training and testing data. The model may partly recognize customer-specific patterns, creating optimistic evaluation for the intended new-customer task.

## Question 3: How do numerical, nominal and ordinal variables differ?

**Answer:** Numerical variables represent quantities. Nominal variables are categories without an inherent order. Ordinal variables are categories with a meaningful order.

**Example:** Income is numerical; city is nominal; satisfaction level (low, medium, high) is ordinal.

**Follow-up:** Can we assign 1, 2 and 3 to nominal categories safely?

**Answer:** Not automatically. A model may interpret the numbers as ordered or as distances. One-hot or another suitable encoding may be more appropriate.

## Question 4: Is customer_id a feature?

**Answer:** It is an identifier. It is usually excluded as a direct numeric feature because its numeric magnitude has no natural relationship to the outcome. It may still be useful for joins, grouping, traceability or leakage-safe splitting.

**Follow-up:** Can an ID ever contain predictive information?

**Answer:** Sometimes the generation process encodes time, region or sequence, but using it requires strong justification and leakage/bias checks. Prefer meaningful underlying attributes when available.

## Question 5: How do you define a churn target for a predictive model?

**Answer:** Define the prediction date and a future outcome window. For example, for each active customer at the end of a month, set the target to Yes if they cancel within the next 30 days, otherwise No, using a consistent business definition.

**Follow-up:** Why is the prediction date important?

**Answer:** It determines which features were available at prediction time and helps prevent future information from entering training data.

## Question 6: A model includes `cancellation_date` as a feature while predicting churn. What is the concern?

**Answer:** It may reveal that cancellation has already occurred or will occur. If unavailable at the intended prediction time, it is target leakage and creates unrealistic performance.

**Follow-up:** How should you handle it?

**Answer:** Exclude it from features for that prediction task, audit other post-outcome columns and reconstruct features using only information available at the prediction timestamp.

## Question 7: What is a data dictionary?

**Answer:** It documents field names, definitions, types, units, allowed values and sometimes validation rules. It ensures teams interpret columns consistently.

**Follow-up:** What might happen if one team interprets `amount` as dollars and another as rupees?

**Answer:** Aggregations, features and business decisions can be wrong by a scale factor. Units must be explicit and validated.

## Question 8: How do you separate features and target in Pandas?

**Answer:** Use a DataFrame for features and a Series for the target, for example `X = df[["age", "income"]]` and `y = df["churn"]`.

**Follow-up:** Why double brackets for X?

**Answer:** Double brackets with a list of columns return a DataFrame; single brackets for one column generally return a Series.

## Question 9: The dataset contains 50,000 rows and 14 columns. Is that enough information to start training?

**Answer:** No. I need to understand the observation unit, schema, target, feature definitions, data types, missingness, label quality, prediction timestamp, duplicates, class balance and any leakage risks.

**Follow-up:** What is the first practical inspection?

**Answer:** Check `shape`, `columns`, `dtypes`, `head()`, `info()`, null counts, unique counts and a data dictionary, then confirm the business definitions.