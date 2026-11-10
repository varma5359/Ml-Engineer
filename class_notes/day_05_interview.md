# Day 05: Interview Questions — NumPy Arrays, Indexing and Slicing

**Pathway:** Machine Learning Engineer

---

## Question 1: What is NumPy and why is it important in machine learning?

**Answer:** NumPy is a Python library for efficient numerical arrays and mathematical operations. ML data is often represented as numeric vectors and matrices, and NumPy supports those operations efficiently.

**Follow-up:** What is an `ndarray`?

**Answer:** It is NumPy's n-dimensional array object, typically storing elements of a common dtype.

## Question 2: What is the difference between a Python list and a NumPy array?

**Answer:** A Python list is a general-purpose collection and can hold mixed object types. A NumPy array is designed for numerical operations, usually has a consistent dtype and supports vectorized computation and multi-dimensional shapes.

**Follow-up:** Can a NumPy array contain strings?

**Answer:** Yes, NumPy supports string and object arrays, but numeric ML calculations generally require properly prepared numerical arrays.

## Question 3: Explain `shape`, `ndim`, `size` and `dtype`.

**Answer:** `shape` gives the length along each dimension, `ndim` gives the number of dimensions, `size` gives the total number of elements, and `dtype` gives the element data type.

**Example:** For an array of shape `(100, 5)`, there are 100 observations and 5 columns, 2 dimensions and 500 elements.

**Follow-up:** Does shape `(100, 5)` tell you which column is the target?

**Answer:** No. Shape tells the dimensions only. You must know the dataset schema and target definition.

## Question 4: What is the difference between indexing and slicing?

**Answer:** Indexing selects a particular element. Slicing selects a range or subset of elements.

**Example:** `x[2]` selects one value; `x[1:4]` selects positions 1, 2 and 3.

**Follow-up:** Is the stop position included in a Python slice?

**Answer:** No. The start is included and the stop is excluded.

## Question 5: A matrix has shape `(500, 8)`. How do you select the fourth feature column?

**Answer:** `matrix[:, 3]`, assuming feature columns are zero-indexed and the fourth column is indeed the desired feature.

**Follow-up:** How do you select rows 10 through 19?

**Answer:** `matrix[10:20, :]`.

## Question 6: What is vectorization?

**Answer:** Vectorization means applying an operation to an entire array or vector rather than writing an explicit Python loop for each element. NumPy performs these operations in optimized compiled code in many cases.

**Follow-up:** Is vectorization always possible?

**Answer:** Not every algorithm or dependency is naturally vectorized. Use the clearest correct approach and measure performance where it matters.

## Question 7: What does `axis=0` mean in `array.sum(axis=0)`?

**Answer:** It reduces along axis 0, combining values down the rows and producing one sum for each column.

**Follow-up:** What does `axis=1` produce for a 2-D matrix?

**Answer:** One sum for each row, combining values across columns.

## Question 8: You get a broadcasting error when adding two arrays. How do you debug it?

**Answer:** Print both arrays' shapes. Compare dimensions from the right; corresponding dimensions must match or one must be 1. Reshape or transpose only when it matches the intended mathematical meaning.

**Follow-up:** Would you reshape randomly until the error disappears?

**Answer:** No. A shape-compatible result can still be mathematically wrong. Confirm what each axis represents before reshaping.

---

## Expanded Beginner-to-Developer Teaching Layer

# Day 05: Interview Questions — NumPy arrays, indexing and slicing

**Pathway:** Machine Learning Engineer  
**Difficulty progression:** Beginner → Practical → Scenario-based → Developer-level

---

## Section 1: Beginner Concept Questions

### Q1. Explain numpy arrays, indexing and slicing in simple terms.

**Answer:** NumPy is a Python library for efficient numerical arrays and vectorized calculations. An ndarray stores elements in a structured n-dimensional array, commonly with one data type.

**Why it matters:** Use NumPy for numerical datasets, matrix operations, statistical calculations and efficient element-wise operations.

**Follow-up:** Explain this concept to a non-technical business stakeholder in two sentences. A strong answer avoids jargon and connects the concept to the business result.

### Q2. Why is this concept required in a Machine Learning workflow?

**Answer:** Use NumPy for numerical datasets, matrix operations, statistical calculations and efficient element-wise operations. It helps a developer make a deliberate choice instead of blindly applying a library function or algorithm.

**Follow-up:** What could happen if the team skips this step?

**Answer:** undefined

### Q3. What are the important terms or building blocks?

**Answer:** Start by identifying the input, output, assumptions, parameters and validation method. Explain each term in the context of the actual dataset or business problem rather than giving only a memorized definition.

**Follow-up:** Which part is configured by the developer and which part may be learned from data? Explain for the method being discussed.

## Section 2: Practical and Syntax Questions

### Q4. How would you implement the basic operation in Python?

**Answer:** Use the syntax and small worked example in the class notes. A good interview explanation follows this order: import the required component, prepare the input, call the method, inspect the output and validate the result. Mention the important parameters and their defaults.

**Follow-up:** How would you make the code reusable and testable?

**Answer:** Put repeatable logic into a function or pipeline, validate inputs, avoid hidden global state, use clear names and add tests for normal and boundary cases.

### Q5. What common errors do beginners make?

**Answer:** undefined

**Follow-up:** How do you debug an unexpected result?

**Answer:** Reproduce the issue with a small input, inspect data types/shapes/units, check each processing step, compare expected and actual output, then add a regression test after fixing the cause.

## Section 3: Real-Time Scenario Questions

### Q6. A business team reports that the output looks reasonable on a few records but fails for many live records. How do you investigate?

**Answer:** I would not immediately change the algorithm. First, I would define the failure precisely and collect representative examples. Then I would compare live inputs with the training or development data, check missing values, category values, data types, units and feature availability. I would verify that preprocessing is identical between training and inference, and check whether the business process or data distribution changed. I would reproduce the issue, measure its frequency and impact, fix the root cause, and validate the fix on a held-out sample before release.

**Follow-up:** Would you retrain immediately?

**Answer:** Not without evidence. If the cause is a parsing bug or inconsistent preprocessing, fix that first. Retraining on corrupted or unsuitable data can make the problem worse.

### Q7. The model or process gives a good aggregate result, but one customer segment receives poor results. What would you do?

**Answer:** Break down performance by meaningful segments such as region, customer type, product or time period. Check sample sizes and data quality for each segment. Identify whether the issue comes from missing representation, a different error pattern, an unsuitable threshold or a business process difference. Discuss the trade-off with stakeholders and validate any change without using the final test set repeatedly.

**Follow-up:** Why is the overall score not sufficient?

**Answer:** An aggregate metric can hide concentrated harm or failure in a smaller but important group. Production decisions affect individual cases, not just the average.

### Q8. What would you do if the result is technically correct but not useful to the business?

**Answer:** Revisit the problem definition. Confirm that the target, prediction time, success metric and decision threshold match the actual business decision. A model can optimize the wrong target perfectly. I would involve the domain owner, quantify the cost of different errors and redefine evaluation if needed.

**Follow-up:** Who should approve the metric?

**Answer:** The engineering team should explain technical implications, but the metric and acceptable trade-offs need agreement from business owners and relevant risk/compliance stakeholders.

## Section 4: Senior Developer Thinking

### Q9. How do you decide whether to use this technique or a simpler alternative?

**Answer:** Syntax: import numpy as np; a=np.array([10,20,30]); a[0]; a[1:]; a.shape; a.ndim; a.dtype; a.mean(); a[a>20]. Indexing starts at zero; slicing stop index is excluded. I would compare correctness, maintainability, latency, cost, explainability, data requirements and operational risk. I would establish a simple baseline and adopt added complexity only when it creates measurable value.

**Follow-up:** What evidence would justify the more complex approach?

**Answer:** A repeatable improvement on appropriate validation data, meaningful business impact, acceptable error behaviour and a supportable production implementation.

### Q10. What should be monitored after deployment or operational use?

**Answer:** Monitor input quality, missing/unknown values, output distributions, latency, failures and business outcomes. For predictive models, also monitor delayed ground-truth performance, drift and segment-level errors. Define owners and alert thresholds before release.

**Follow-up:** Why can a model that performed well last month fail today?

**Answer:** Customer behaviour, products, prices, policies, data collection or the operating environment may have changed. Historical validation does not guarantee future performance.

## Section 5: Interview Answer Framework

When you receive an unfamiliar scenario, structure your answer as:

1. **Clarify:** What decision must be made and when?
2. **Inspect:** What data, assumptions and constraints exist?
3. **Diagnose:** What evidence supports the suspected cause?
4. **Act:** Choose the smallest safe corrective action.
5. **Validate:** Test normal, boundary and segment cases on appropriate data.
6. **Communicate:** Explain trade-offs, limitations and expected impact.
7. **Monitor:** Confirm the solution remains reliable after release.

**Senior-level principle:** Do not guess the cause from one metric or one example. State the evidence you need, test hypotheses and explain the impact of the decision.
