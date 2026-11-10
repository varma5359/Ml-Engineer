# Day 01: Interview Questions — Machine Learning, AI, ML Categories and Real-World Use Cases

**Pathway:** Machine Learning Engineer

---

## Question 1: Explain Artificial Intelligence, Machine Learning and Deep Learning.

**Answer:** Artificial Intelligence is the broad field of building systems that perform intelligent tasks. Machine Learning is a subset of AI where a system learns patterns from data. Deep Learning is a subset of ML that uses multi-layer neural networks.

**Example:** A support bot is an AI application. A spam classifier trained from labelled emails is an ML system. A neural network that recognizes objects in images is a deep-learning system.

**Follow-up:** Can an AI system work without ML?

**Answer:** Yes. A rule-based expert system can use predefined conditions and rules without learning from training data.

## Question 2: What is the difference between supervised and unsupervised learning?

**Answer:** Supervised learning trains from examples that contain input features and correct targets. Unsupervised learning receives features without target labels and finds structure such as clusters.

**Example:** Predicting churn from past churn labels is supervised learning. Grouping customers by purchasing behaviour without predefined customer types is unsupervised learning.

**Follow-up:** Is clustering a classification problem?

**Answer:** No. Classification learns to assign known classes from labelled examples. Clustering discovers groups without given class labels.

## Question 3: A company wants to predict house prices. Is this regression or classification?

**Answer:** Regression, because the required output is a numeric amount such as ₹52 lakh.

**Follow-up:** What if the company wants to classify houses as low-cost, medium-cost and premium?

**Answer:** That is classification because the output is one of three categories.

## Question 4: A bank wants to flag suspicious transactions. Which ML approach can it use?

**Answer:** If historical transactions are labelled as fraudulent or legitimate, supervised classification can be used. If labels are unavailable or incomplete, anomaly detection or unsupervised approaches may help identify unusual patterns for investigation.

**Follow-up:** Should every flagged transaction automatically be blocked?

**Answer:** Not necessarily. A flag is a risk signal, not proof of fraud. The bank should choose thresholds based on false-positive and false-negative costs and apply suitable review or control processes.

## Question 5: Why do we divide data into training and testing sets?

**Answer:** Training data is used to fit the model. Test data provides an estimate of how well the final model performs on examples it did not learn from. Using the same examples for both can make performance look overly optimistic.

**Follow-up:** Can we repeatedly tune the model using the test set?

**Answer:** We should not. Repeatedly making decisions based on test results indirectly adapts the model to the test set. Use validation data or cross-validation for model selection and keep the test set for final evaluation.

## Question 6: Does a model with 99% accuracy always mean it is good?

**Answer:** No. If only 1% of transactions are fraudulent, a model that predicts every transaction as legitimate gets 99% accuracy but detects no fraud. We must inspect the confusion matrix, precision, recall, F1-score and business costs.

**Follow-up:** Which metric would you prioritize for detecting fraud?

**Answer:** It depends on the costs. Recall is important when missing fraud is very costly; precision matters when false alerts are costly. Often we evaluate both and choose an operating threshold with stakeholders.

## Question 7: What is the difference between a model and an algorithm?

**Answer:** An algorithm is the learning procedure. A model is the result after the algorithm has learned parameters from training data.

**Example:** Linear regression is an algorithm/model family; after fitting it to data, the learned coefficients and intercept form the fitted model.

**Follow-up:** What are hyperparameters?

**Answer:** Hyperparameters are settings selected before or around training, such as a decision tree's maximum depth. They are not the fitted coefficients learned directly from each training example.

## Question 8: When would you avoid machine learning?

**Answer:** I would avoid ML when a simple rule is adequate, data is insufficient or unreliable, success cannot be measured, the error risk cannot be managed, or the model cost exceeds the business benefit.

**Follow-up:** What would you do instead?

**Answer:** Start with a documented rule-based approach or manual process, collect reliable evidence, define metrics and revisit ML when enough data and a clear use case exist.

---

## Expanded Beginner-to-Developer Teaching Layer

# Day 01: Interview Questions — Machine Learning, AI, ML categories and real-world use cases

**Pathway:** Machine Learning Engineer  
**Difficulty progression:** Beginner → Practical → Scenario-based → Developer-level

---

## Section 1: Beginner Concept Questions

### Q1. Explain machine learning, ai, ml categories and real-world use cases in simple terms.

**Answer:** AI is the broad field of building systems that perform tasks associated with intelligence. ML is a way to learn patterns from examples. Deep learning uses multi-layer neural networks.

**Why it matters:** Choose rules when the business rule is stable and explicit; choose ML when patterns are too complex to write manually and representative data exists.

**Follow-up:** Explain this concept to a non-technical business stakeholder in two sentences. A strong answer avoids jargon and connects the concept to the business result.

### Q2. Why is this concept required in a Machine Learning workflow?

**Answer:** Choose rules when the business rule is stable and explicit; choose ML when patterns are too complex to write manually and representative data exists. It helps a developer make a deliberate choice instead of blindly applying a library function or algorithm.

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

**Answer:** If the target is a numeric amount, think regression. If it is a named class, think classification. If there are no labels and the goal is grouping, think clustering. I would compare correctness, maintainability, latency, cost, explainability, data requirements and operational risk. I would establish a simple baseline and adopt added complexity only when it creates measurable value.

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
