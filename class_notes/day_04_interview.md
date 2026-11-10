# Day 04: Interview Questions — Lists, Dictionaries, Tuples and Sets

**Pathway:** Machine Learning Engineer

---

## Question 1: What is the difference between a list and a tuple?

**Answer:** Both are ordered and allow duplicates. A list is mutable; a tuple is immutable.

**Example:** Use a list for customer transactions that may be updated and a tuple for a fixed image shape such as `(224, 224, 3)`.

**Follow-up:** Can a tuple contain a list?

**Answer:** Yes, but the tuple's reference to that list cannot be reassigned. The list object itself can still be mutated.

## Question 2: When would you use a dictionary?

**Answer:** Use a dictionary when values should be accessed by meaningful keys rather than numeric positions.

**Example:** A customer record can store `{"customer_id": "C10", "age": 28}`.

**Follow-up:** What is the difference between `data[key]` and `data.get(key)`?

**Answer:** Square-bracket access raises `KeyError` when the key is absent. `get()` can return a default value instead.

## Question 3: Why would you use a set in data preparation?

**Answer:** A set stores unique elements and supports fast membership checks and union/intersection/difference operations.

**Example:** Find unique product categories or compare IDs between two data extracts.

**Follow-up:** Should you use a set to preserve transaction order?

**Answer:** No. Use a list or an ordered tabular structure when order matters.

## Question 4: How can you represent many customer records using Python collections?

**Answer:** A common simple structure is a list of dictionaries. Each dictionary represents one customer and each key identifies a field.

**Follow-up:** Is this always ideal for a million-row dataset?

**Answer:** No. Pandas or a database is usually better for large tabular workloads, filtering, grouping and vectorized operations.

## Question 5: Your code accesses `customer["age"]` and raises `KeyError`. What does it indicate?

**Answer:** The dictionary does not contain the key `"age"` at that moment. Check the schema, spelling, source record and whether the field is optional. Use `get()` only when a missing key is an expected condition and handle it explicitly.

**Follow-up:** Would `customer.get("age", 0)` always be a good fix?

**Answer:** No. Zero may be a valid age and can hide a schema defect. A suitable missing value or explicit validation error is often better.

## Question 6: How do you remove duplicates from a list?

**Answer:** A set can remove duplicates when order is not important: `list(set(values))`. To preserve first-seen order, use `list(dict.fromkeys(values))` for hashable values.

**Follow-up:** Should duplicate rows always be removed from a business dataset?

**Answer:** No. First determine the business key and whether the rows represent true duplicate records or repeated legitimate events.

## Question 7: Which collection would you choose for model feature names that should not change?

**Answer:** A tuple can be appropriate because it is ordered and immutable. A list may be used when the names need to be updated during preprocessing.

**Follow-up:** Why does feature order matter?

**Answer:** A model expects input columns in the same order and meaning used during training. Incorrect order can send values into the wrong model inputs.

## Question 8: What collection design would you use for counting orders by status?

**Answer:** A dictionary can map each status to its count. For example, `{"delivered": 20, "pending": 5}`. `collections.Counter` is also designed for counting hashable values.

**Follow-up:** What happens when a new status appears?

**Answer:** The counting logic should support new keys safely, while the application should validate whether the status is allowed by the business schema.

---

## Expanded Beginner-to-Developer Teaching Layer

# Day 04: Interview Questions — Lists, dictionaries, tuples and sets

**Pathway:** Machine Learning Engineer  
**Difficulty progression:** Beginner → Practical → Scenario-based → Developer-level

---

## Section 1: Beginner Concept Questions

### Q1. Explain lists, dictionaries, tuples and sets in simple terms.

**Answer:** A list is an ordered mutable sequence; a tuple is an ordered immutable sequence; a dictionary maps unique keys to values; a set stores unique elements without positional indexing.

**Why it matters:** Choose the structure based on access pattern: ordered records, fixed grouped values, key-based lookup or uniqueness/membership checks.

**Follow-up:** Explain this concept to a non-technical business stakeholder in two sentences. A strong answer avoids jargon and connects the concept to the business result.

### Q2. Why is this concept required in a Machine Learning workflow?

**Answer:** Choose the structure based on access pattern: ordered records, fixed grouped values, key-based lookup or uniqueness/membership checks. It helps a developer make a deliberate choice instead of blindly applying a library function or algorithm.

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

**Answer:** Syntax: items = [10,20]; profile = {'name':'Asha','age':25}; point = (17.4,78.4); unique_ids = {101,102}. Use append(), get(), keys(), values(), add(), update(), discard(). I would compare correctness, maintainability, latency, cost, explainability, data requirements and operational risk. I would establish a simple baseline and adopt added complexity only when it creates measurable value.

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
