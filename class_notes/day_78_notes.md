# Day 78: Capstone — Data Understanding and Quality

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Beginner to Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Dataset profiling

## 1. Learning objectives
- Load a customer churn dataset and inspect its structure.
- Identify target, features, identifiers and possible leakage columns.
- Measure missing values, duplicates and invalid values.
- Examine class balance and basic distributions.
- Produce a reproducible dataset profiling report.

## 2. Why profile data before modeling?
A model learns patterns from the data it receives. Incorrect types, missing values, duplicate records, inconsistent categories and target problems can lead to misleading results.

Data profiling answers:
- What does each row represent?
- Which column is the target?
- What are the feature types?
- Are values missing or invalid?
- Is the target distribution highly imbalanced?
- Are any fields unavailable at prediction time?

## 3. Dataset assumptions
For this capstone, use an approved customer churn dataset. A common teaching dataset contains columns such as:
- CustomerID
- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- InternetService
- Contract
- MonthlyCharges
- TotalCharges
- Churn

Actual schemas differ. Never assume these exact fields exist; inspect your selected dataset and record its source, license/usage conditions and target definition.

## 4. Load the dataset
```python
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
print(df.head())
```

Explanation:
- `read_csv` loads a comma-separated file.
- `df` is a DataFrame.
- `head()` previews the first five records.

## 5. Inspect dimensions, columns and types
```python
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.dtypes)
df.info()
```

Check:
- Number of rows and columns.
- Duplicate column names.
- Numeric and categorical types.
- Columns unexpectedly loaded as `object`.
- Identifier and target fields.

## 6. Separate target and identifier
```python
target_col = "Churn"  # Change to the real target column name.
id_candidates = ["customerID", "CustomerID", "customer_id"]

print("Target values:")
print(df[target_col].value_counts(dropna=False))
print("Possible identifiers:",
      [col for col in id_candidates if col in df.columns])
```

A customer identifier is generally not a meaningful predictive feature. Confirm its role before excluding it.

## 7. Check missingness and duplicates
```python
missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_pct.round(2)
})
print(missing_report)
print("Duplicate rows:", df.duplicated().sum())
```

A blank string or "Unknown" category may represent missingness without being a true null. Inspect category values too.

## 8. Check numeric and categorical distributions
```python
print(df.describe(include="all").T)
```

For a categorical feature:
```python
if "Contract" in df.columns:
    print(df["Contract"].value_counts(dropna=False))
```

For a numeric feature:
```python
if "MonthlyCharges" in df.columns:
    print(df["MonthlyCharges"].describe())
```

Use the actual dataset's column names. Look for impossible ranges, unexpected units and unusual category spellings.

## 9. Inspect target balance
```python
target_counts = df[target_col].value_counts(dropna=False)
target_percent = df[target_col].value_counts(normalize=True, dropna=False) * 100

print(target_counts)
print(target_percent.round(2))
```

If churn is a minority class, accuracy may hide poor positive-class detection. Record the positive label and map it consistently in later stages.

## 10. Investigate suspicious columns
Create a data dictionary containing:
- Column name.
- Business meaning.
- Data type.
- Allowed values or range.
- Missing-value meaning.
- Whether available at prediction time.
- Whether it is an identifier, target or candidate feature.

Do not remove a column only because it looks unusual. Verify meaning and timing with documentation or the data owner.

## 11. Approved deliverable — Dataset profiling

**Objective:** Create a reproducible report describing the selected churn dataset before cleaning or modeling.

### Tasks
1. Record source, download date and dataset version.
2. Load the raw file without overwriting it.
3. Inspect dimensions, types and column names.
4. Identify target, identifier and candidate features.
5. Report missing values and duplicates.
6. Inspect numeric ranges and categorical values.
7. Report target counts and percentages.
8. Flag possible leakage and unavailable-at-prediction-time fields.
9. Create a data dictionary and profiling summary.

**Suggested structure**
```text
customer_churn_capstone/
├── data/raw/customer_churn.csv
├── reports/
│   └── dataset_profile.md
├── notebooks/
│   └── 01_dataset_profiling.ipynb
└── README.md
```

**Validation:** The report must be reproducible from the raw dataset. Do not silently modify the source file. Record unresolved data questions rather than guessing.

## 12. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for identifying suspicious data types." |
| Plan | "Break dataset profiling into notebook cells and report sections." |
| Example | "Show a small example of a missing-value summary table." |
| Guided implementation | "Help me inspect the dataset one step at a time; wait after each step." |
| Explanation | "Explain what df.info and describe reveal and what they cannot prove." |
| Debugging | "Review this profiling error and help me isolate the cause." |
| Review | "Review my data dictionary and flag target leakage or ambiguous columns." |

## 13. Daily quiz

**Q1. What does df.shape return?**  
A. Only column types  
B. Row and column counts  
C. Missing values  
D. Model scores  
**Correct: B.** The tuple contains the number of rows and columns.

**Q2. Which check finds fully duplicated rows?**  
A. df.describe()  
B. df.duplicated().sum()  
C. df.shape  
D. df.columns  
**Correct: B.** It counts rows that duplicate earlier rows.

**Q3. Why inspect target balance?**  
A. To choose a UI theme  
B. To identify class imbalance and plan evaluation  
C. To rename columns  
D. To serialize the model  
**Correct: B.** Class proportions influence metric selection and modeling strategy.

**Q4. A column is stored as object but represents monthly charges. What should you do first?**  
A. Drop it immediately  
B. Inspect its values and conversion issues  
C. Use it as the target  
D. Replace all values with zero  
**Correct: B.** Determine why conversion failed before cleaning.

**Q5. What is the safest treatment of an unexplained post-cancellation field?**  
A. Use it because it improves accuracy  
B. Treat it as a potential leakage risk and investigate timing  
C. Duplicate it  
D. Convert it to an ID  
**Correct: B.** Feature availability at prediction time must be verified.

## 14. Interview preparation

### Q1. What steps do you follow when you receive a new dataset?
**Answer:** Verify source and version, inspect dimensions and types, identify target and IDs, check missingness and duplicates, inspect distributions and target balance, then document data meaning and leakage risks before transformations.

**Follow-up:** How do you verify that a field is available at prediction time?

### Q2. Why should raw data be preserved?
**Answer:** It provides an auditable source for reproducing transformations and investigating errors. Cleaning should create a derived dataset or pipeline output, not silently overwrite the original.

**Follow-up:** How would you track changes between raw dataset versions?

## 15. Completion checklist
- [ ] Load and inspect the raw dataset.
- [ ] Identify target, ID and candidate features.
- [ ] Measure missingness and duplicates.
- [ ] Inspect distributions and target balance.
- [ ] Document field meaning and leakage risks.
- [ ] Complete Dataset profiling.

**Next lesson:** Day 79 — Capstone: Preprocessing and feature engineering.
