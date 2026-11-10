# Day 78: Interview Questions — Capstone Data Understanding and Quality

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. What is data profiling?
**Answer:**
- Inspecting the structure, types, quality and distributions of a dataset.
- It identifies missingness, duplicates, invalid values and class balance.
- It helps establish whether the data is suitable for the task.
- It should happen before modeling.

**Follow-up:** Which profiling checks would you automate?

### Q2. How do you identify the target variable?
**Answer:**
- Refer to the business problem and label definition.
- Confirm the column represents the outcome over the agreed horizon.
- Inspect its values and missingness.
- Verify that it is not derived from future information incorrectly.

**Follow-up:** What if the dataset has multiple possible churn columns?

### Q3. Why inspect data types?
**Answer:**
- Incorrect types can cause errors or misleading analysis.
- Numeric values may be stored as strings.
- Categories may be encoded inconsistently.
- Type checks help select correct preprocessing operations.

**Follow-up:** How would you handle numeric values containing currency symbols?

### Q4. How do you investigate missing values?
**Answer:**
- Measure missing count and percentage by column.
- Check whether blanks and special strings encode missingness.
- Understand why the values are missing.
- Select imputation, exclusion or missingness indicators based on training data and business meaning.

**Follow-up:** Why can missingness itself be informative?

### Q5. Why inspect class imbalance?
**Answer:**
- It affects baseline performance and metric interpretation.
- Accuracy can appear high when the majority class dominates.
- Precision, recall, F1 and PR-AUC may be more informative.
- It influences threshold and resampling decisions.

**Follow-up:** How would you establish a baseline for an imbalanced target?

## 2. Scenario questions

### Q6. TotalCharges is stored as text. What do you do?
**Answer:**
- Inspect unique and non-numeric values.
- Identify blanks or formatting issues.
- Convert with controlled error handling.
- Quantify rows that fail conversion and decide their treatment.
- Document the transformation.

**Follow-up:** Why should you not silently coerce all errors to missing and continue?

### Q7. You find duplicate customer IDs. Are all duplicates invalid?
**Answer:**
- Not necessarily; the dataset may contain multiple snapshots per customer.
- Identify the intended row granularity.
- Check date and event columns.
- Define the correct uniqueness key.
- Deduplicate only when business rules justify it.

**Follow-up:** How would repeated monthly snapshots change the train/test split?

### Q8. A feature has extremely high correlation with churn. What do you investigate?
**Answer:**
- Check whether it is recorded after the churn event.
- Verify its business meaning and timing.
- Look for target-derived encoding or duplicated target information.
- Validate with a time-aware split.
- Exclude it if unavailable at prediction time.

**Follow-up:** Can high correlation alone prove leakage?

## 3. Project questions

### Q9. What should a dataset profile report contain?
**Answer:**
- Dataset source and version.
- Row/column counts and schema.
- Target and identifier definitions.
- Missingness, duplicates and invalid values.
- Numeric and categorical summaries.
- Target balance and potential leakage notes.
- Unresolved questions and limitations.

**Follow-up:** How would you make the report reproducible?

### Q10. Why preserve raw data?
**Answer:**
- It provides an unchanged source for audit and reproduction.
- Transformations can be rerun and compared.
- It helps debug unexpected model behavior.
- Derived clean datasets should have their own versions.

**Follow-up:** Which tool could version large datasets alongside Git?

## 4. Rapid revision
- Profiling: inspect structure and quality.
- Target: outcome to predict.
- Identifier: record reference, often not a feature.
- Missingness: absent or invalid values.
- Class balance: distribution of target labels.
- Leakage: information unavailable at prediction time.
- Data dictionary: field meaning and rules.
