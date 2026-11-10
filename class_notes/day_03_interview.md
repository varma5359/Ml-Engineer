# Day 03: Interview Questions — Conditions, Loops and Functions

**Pathway:** Machine Learning Engineer

---

## Question 1: What is the difference between `if`, `elif` and `else`?

**Answer:** `if` checks the first condition, `elif` checks another condition when earlier conditions are false, and `else` runs when none of the previous conditions match.

**Example:** Assigning grades based on marks.

**Follow-up:** Why should conditions be ordered carefully?

**Answer:** Python executes the first matching branch. A broad condition placed before a more specific condition may prevent the specific branch from running.

## Question 2: What is the difference between `for` and `while` loops?

**Answer:** A `for` loop iterates over an iterable such as a list or range. A `while` loop repeats as long as its condition is true.

**Example:** Use `for` to process each row in a list; use `while` to repeat a process until a stopping condition is met.

**Follow-up:** What can cause an infinite `while` loop?

**Answer:** If the condition never becomes false—for example, the counter is not updated—the loop can continue indefinitely.

## Question 3: Explain `break` and `continue`.

**Answer:** `break` exits the nearest loop. `continue` skips the remaining code in the current iteration and proceeds to the next iteration.

**Follow-up:** Which would you use to stop searching after finding the first matching record?

**Answer:** `break`.

## Question 4: What is a function and why do we use it?

**Answer:** A function is a reusable named block of code designed to perform a task. It reduces duplication, improves readability and makes logic easier to test.

**Example:** A function that calculates revenue can be used for many orders.

**Follow-up:** What is the difference between a parameter and an argument?

**Answer:** A parameter is the name in the function definition; an argument is the actual value passed when calling the function.

## Question 5: What is the difference between `print()` and `return`?

**Answer:** `print()` displays information. `return` sends a value back to the caller so it can be stored or used in another calculation.

**Scenario:** A function prints a calculated average, but another function receives `None` when trying to use it. Why?

**Answer:** The first function printed the average but did not return it. A Python function without an explicit return value returns `None`.

## Question 6: How would you handle an empty list while calculating an average?

**Answer:** Check whether the list is empty before dividing by its length. Raise a meaningful error or return a documented missing result depending on the application's requirements.

**Follow-up:** Why is returning zero potentially misleading?

**Answer:** Zero is a valid average for some datasets. It is not the same as having no observations, so replacing an undefined average with zero can distort analysis.

## Question 7: You need to process 100,000 transaction records. Would you write 100,000 separate statements?

**Answer:** No. I would store or read the records in a suitable collection or dataframe and process them using reusable logic, usually vectorized dataframe/NumPy operations for numerical workloads rather than slow Python loops where appropriate.

**Follow-up:** Are Python loops always wrong for data processing?

**Answer:** No. They are useful for control flow, custom logic and small workloads. For large numerical arrays, vectorized library operations are often faster and clearer.

## Question 8: A model returns a probability for each customer. How could a function convert the score into a business action?

**Answer:** A function can apply a validated threshold and return a category such as high-priority review or normal monitoring. The threshold should be chosen using validation data and the cost of false positives and false negatives. The rule should not be described as the model itself.

**Follow-up:** Should the threshold always be 0.5?

**Answer:** No. The best threshold depends on class balance, probability calibration and business costs. It should be evaluated for the intended application.

---

## Expanded Beginner-to-Developer Teaching Layer

# Day 03: Interview Questions — Conditions, loops and functions

**Pathway:** Machine Learning Engineer  
**Difficulty progression:** Beginner → Practical → Scenario-based → Developer-level

---

## Section 1: Beginner Concept Questions

### Q1. Explain conditions, loops and functions in simple terms.

**Answer:** A condition selects a path based on True or False. A loop repeats work. A function is a named reusable block that accepts inputs and may return a result.

**Why it matters:** Use if/elif/else for decisions, for/while for repetition, and functions to organize repeated logic and make it testable.

**Follow-up:** Explain this concept to a non-technical business stakeholder in two sentences. A strong answer avoids jargon and connects the concept to the business result.

### Q2. Why is this concept required in a Machine Learning workflow?

**Answer:** Use if/elif/else for decisions, for/while for repetition, and functions to organize repeated logic and make it testable. It helps a developer make a deliberate choice instead of blindly applying a library function or algorithm.

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

**Answer:** Syntax: if condition: ... elif condition: ... else: ...; for item in items: ...; def function_name(parameter): ... return result. Example: def classify_amount(x): return 'review' if x >= 50000 else 'normal'. I would compare correctness, maintainability, latency, cost, explainability, data requirements and operational risk. I would establish a simple baseline and adopt added complexity only when it creates measurable value.

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
