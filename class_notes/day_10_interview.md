# Day 10: Interview Questions — Missing Values, Duplicates and Inconsistent Data

**Pathway:** Machine Learning Engineer

---

## Question 1: What is data cleaning and why is it important in ML?

**Answer:** Data cleaning identifies and handles missing, duplicate, inconsistent and invalid data. It is important because models learn from the supplied data; poor-quality records can create misleading patterns, errors and unreliable predictions.

**Follow-up:** Does cleaning mean removing all unusual records?

**Answer:** No. Unusual values may be valid. Investigate their source and meaning before deciding how to handle them.

## Question 2: How do you identify missing values in Pandas?

**Answer:** Use `df.isna().sum()` to count missing values per column. Use `df.isna().mean() * 100` to calculate their percentage.

**Follow-up:** Are empty strings always counted as missing by Pandas?

**Answer:** Not necessarily. Empty strings are strings, not automatically NA. Normalize them explicitly when the source definition says they represent missing values.

## Question 3: How do you decide between dropping and imputing missing values?

**Answer:** Consider the amount and pattern of missingness, business meaning, column importance, data distribution and impact on the target. Dropping may be suitable for very few safely removable rows. Imputation may preserve useful records, but its method must be justified.

**Follow-up:** Why might median imputation be preferred to mean imputation?

**Answer:** Median is more robust to extreme values and may better represent a skewed numerical feature.

## Question 4: Why should you not replace every missing value with zero?

**Answer:** Zero may be a valid measurement with a different meaning from unknown. Replacing missing age, income or temperature with zero can create false information and distort patterns.

**Follow-up:** What is a suitable alternative for a categorical field?

**Answer:** A mode or explicit category such as `Unknown` may be suitable, depending on what missing means and the model's needs.

## Question 5: What is the difference between exact duplicates and business-key duplicates?

**Answer:** Exact duplicates match across all columns. Business-key duplicates share the fields that the business defines as uniquely identifying a record, even if other fields differ.

**Scenario:** Two orders from the same customer have the same amount. Are they duplicates?

**Answer:** Not necessarily. They may be two legitimate purchases. Use a verified order identifier or business key to decide.

## Question 6: How do you handle inconsistent city names such as `Hyd`, `HYD` and `Hyderabad`?

**Answer:** Inspect unique values, confirm equivalence with a trusted business reference, create a documented mapping and apply it consistently. Avoid mapping based solely on assumptions.

**Follow-up:** What if `Hyd` could refer to a different field code?

**Answer:** Do not map it until the source owner or data dictionary confirms the meaning.

## Question 7: Why is fitting an imputer on the entire dataset a leakage problem?

**Answer:** The imputer learns statistics such as the mean or median from validation/test data as well as training data. That allows information from held-out data to influence training preprocessing and can make evaluation optimistic. Fit on training data and transform the held-out data using that fitted imputer.

**Follow-up:** How can you implement this safely with scikit-learn?

**Answer:** Put the imputer and other preprocessing steps inside a `Pipeline` or `ColumnTransformer` and fit that pipeline only on the training split.

## Question 8: `pd.to_numeric(..., errors="coerce")` converted some amounts to NaN. What should you do next?

**Answer:** Count the newly missing values, inspect the original raw strings, identify patterns such as currency symbols or invalid text, correct valid formats, and document truly invalid entries. Do not silently discard them.

**Follow-up:** Why preserve the original raw column or source file?

**Answer:** It supports auditability and allows the cleaning logic to be reviewed or corrected without losing the original evidence.

## Question 9: A model's training accuracy is unusually high after cleaning. How would you investigate?

**Answer:** Check for duplicates across train/test sets, target leakage, post-outcome fields, preprocessing fitted before the split, incorrect target construction, class imbalance and an overly easy or biased split. Evaluate using a suitable holdout strategy and business-relevant metrics.

**Follow-up:** Does deduplicating the full dataset before splitting always solve duplicate leakage?

**Answer:** It can help with exact duplicates, but it does not address repeated entities or time leakage by itself. The split must match the real prediction scenario, such as grouping by customer or separating by time.