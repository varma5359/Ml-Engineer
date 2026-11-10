# Day 01: Machine Learning, AI, ML Categories and Real-World Use Cases

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

By the end of this lesson, you should be able to:

- Explain Artificial Intelligence, Machine Learning and Deep Learning with practical examples.
- Describe how a machine learning system learns from historical examples.
- Distinguish supervised, unsupervised and reinforcement learning.
- Identify regression, classification, clustering and anomaly-detection problems.
- Explain the major stages of a basic ML workflow.
- Recognize where ML is useful and where a simple rule is better.

## 2. What Is Artificial Intelligence?

**Artificial Intelligence (AI)** is the broader field of building computer systems that perform tasks normally associated with human intelligence, such as understanding language, recognizing images, reasoning, planning and making decisions.

Examples:

| Application | AI capability |
|---|---|
| Voice assistant | Understands a spoken request and responds |
| Navigation application | Finds a route using maps and traffic information |
| Document reader | Extracts and understands text from documents |
| Medical image support | Highlights patterns for a clinician to review |
| Chatbot | Understands a question and generates a response |

AI does not always mean machine learning. A fixed rule-based system can also be an AI application.

### Rule-based system vs machine learning

A rule-based system follows rules written by a developer. A machine learning system learns patterns from examples.

| Situation | Rule-based approach | ML approach |
|---|---|---|
| Bank transaction alert | Flag every transaction above a fixed configured limit | Learn unusual transaction patterns from historical data |
| Email filtering | Block messages containing specified words | Learn spam patterns from labelled spam and legitimate emails |
| Product recommendation | Recommend the same popular item to everyone | Learn user and item preference patterns |

**Important:** ML is not automatically better. If the rule is stable, simple and fully known, a rule-based solution may be cheaper, easier to explain and safer.

## 3. What Is Machine Learning?

Machine Learning (ML) is a branch of AI in which a computer uses data and an algorithm to learn patterns, then uses the learned pattern to make predictions or decisions on new data.

### Simple example: house price prediction

Suppose a property company has historical records:

| House area (sq. ft.) | Bedrooms | Actual price (₹ lakh) |
|---:|---:|---:|
| 800 | 2 | 35 |
| 1,000 | 2 | 43 |
| 1,200 | 3 | 55 |
| 1,500 | 3 | 68 |
| 1,800 | 4 | 82 |

The model studies how area and bedrooms relate to price. After training, it may estimate the price of a new 1,300 sq. ft. house.

The model has not memorized a guaranteed market price. It has learned an approximate relationship from the examples, and its prediction can be wrong.

### Basic ML flow

```text
Historical examples
        |
        v
Prepare input data and correct answers (when available)
        |
        v
Choose an ML algorithm
        |
        v
Train the model (learn parameters/patterns)
        |
        v
Evaluate on data not used for training
        |
        v
Use the model on new records
        |
        v
Monitor errors and changing data
```

## 4. Important ML Terminology

| Term | Simple meaning | House-price example |
|---|---|---|
| Dataset | Collection of examples | Historical house records |
| Observation / sample | One record or row | One house |
| Feature | Input information used by a model | Area, bedrooms, age |
| Target / label | Value the model must predict | House price |
| Algorithm | Learning procedure | Linear regression |
| Model | Learned result of training | Fitted regression model |
| Training | Learning from examples | Fit using historical houses |
| Prediction / inference | Applying a trained model | Estimate a new house price |
| Parameter | Value learned by the algorithm | Regression coefficient |
| Hyperparameter | Setting chosen before/during training | Tree depth, number of neighbours |
| Generalization | Performing well on unseen examples | Estimating prices for new houses |

**Feature vs target:** A feature is an input; the target is the answer the model is trying to estimate.

## 5. Machine Learning Categories

### 5.1 Supervised Learning

The training data contains both input features and the correct target values. The model learns a mapping from inputs to known answers.

Two major problem types:

**A. Regression — predict a number**

Examples:

- Predict a house price.
- Estimate monthly electricity consumption.
- Forecast product demand.
- Estimate delivery time in minutes.

**B. Classification — predict a category**

Examples:

- Spam or not spam.
- Customer will churn or will not churn.
- Loan application: approve/review/reject, subject to policy.
- Image: cat, dog or bird.

| Regression | Classification |
|---|---|
| Target is numeric | Target is a class/category |
| ₹52.4 lakh | Churn = Yes |
| 42.7 minutes | Spam = No |
| 1,250 units | Image = Dog |

A numeric-looking label is not necessarily regression. For example, predicting a postal code is usually classification because the numbers identify categories rather than a measurable quantity.

### 5.2 Unsupervised Learning

Training data has input features but no supplied answer label. The algorithm looks for structure or patterns.

Examples:

- Group customers with similar purchasing patterns.
- Identify groups of similar news articles.
- Compress many features into a smaller representation.
- Find unusual transactions for investigation.

**Clustering example:** A retailer groups customers using purchase frequency, average order value and product categories. The groups may reveal frequent high-value customers, occasional shoppers and discount-focused shoppers. The algorithm finds groups; a human interprets what they mean.

### 5.3 Reinforcement Learning

An agent learns by taking actions in an environment and receiving rewards or penalties. It aims to learn a strategy that maximizes long-term reward.

Examples:

- A game-playing agent learns which moves lead to winning.
- A simulated robot learns how to navigate without hitting obstacles.
- A traffic-control simulator learns signal policies that reduce waiting time.

| Category | Training signal | Typical task |
|---|---|---|
| Supervised | Correct labels/targets | Predict churn or price |
| Unsupervised | No supplied target label | Find customer groups |
| Reinforcement | Rewards and penalties | Learn a sequence of actions |

## 6. Deep Learning and Its Relationship to ML

Deep Learning (DL) is a subset of ML that uses neural networks with multiple processing layers. It is especially useful for complex data such as images, audio, video and language, when sufficient data and compute are available.

```text
Artificial Intelligence (AI)
          |
          v
   Machine Learning (ML)
          |
          v
    Deep Learning (DL)
```

| Traditional ML often works well for | Deep learning is often useful for |
|---|---|
| Structured business tables | Images and video |
| Smaller or medium-sized datasets | Large-scale image/audio/text tasks |
| Problems where interpretability matters | Complex learned representations |
| Fast training with limited compute | Tasks that may need substantial compute |

This is a general tendency, not a strict rule. Deep learning can work with tabular data, and traditional ML can solve some image or text tasks. Choose using validation performance, cost, data volume, latency and explainability needs.

## 7. ML Problem Types in Industry

| Business situation | ML problem | Possible output |
|---|---|---|
| Predict next month's rent | Regression | ₹18,500 |
| Identify likely-to-churn customer | Binary classification | Churn probability 0.82 |
| Categorize support ticket | Multiclass classification | Billing / Delivery / Refund |
| Group customers by behaviour | Clustering | Cluster 0, 1 or 2 |
| Find unusual card activity | Anomaly detection | Risk score / alert |
| Recommend a product | Recommendation / ranking | Ranked product list |

## 8. Industry Scenario — Customer Churn

A subscription company notices that some customers stop renewing. It wants to contact customers who may leave.

**Inputs could include:** plan type, tenure, monthly bill, usage, support calls and payment history. **Target:** whether the customer left during a defined future period.

This is supervised binary classification when historical examples include the churn label.

A responsible workflow must:

1. Define exactly what counts as churn and the prediction date.
2. Use only information available at that prediction date.
3. Split historical examples into training and evaluation sets appropriately.
4. Train a baseline model.
5. Evaluate more than accuracy if churn is rare.
6. Choose a decision threshold based on the cost of missed churners and unnecessary contacts.
7. Monitor performance and customer impact after deployment.

**Scenario lesson:** A model returning a probability of 0.82 is not saying the customer is certainly leaving. It estimates risk based on patterns in training data.

## 9. When Should We Use ML?

Use ML when:

- There are enough relevant examples to learn from.
- The pattern is difficult to express with a small set of fixed rules.
- A prediction or ranking can improve a real decision.
- The output can be evaluated against a meaningful target or outcome.
- The organization can monitor errors and handle uncertain predictions.

Avoid or delay ML when:

- A simple rule solves the problem reliably.
- The data is too limited, biased or unreliable.
- There is no clear definition of success.
- A wrong prediction could cause harm and no human review or safeguard exists.
- The model cannot be maintained as the business changes.

## 10. Common Beginner Misunderstandings

| Misunderstanding | Correct understanding |
|---|---|
| ML always gives correct answers | ML estimates patterns and can make mistakes |
| More features always improve a model | Irrelevant, noisy or leaking features can harm it |
| Accuracy proves a model is good | Metric choice must match class balance and error costs |
| Training score is enough | Evaluate on unseen data |
| AI and ML are identical | AI is broader; ML is one approach within AI |
| Deep learning replaces all ML | Model choice depends on the task and constraints |
| A model understands causality automatically | Most predictive ML learns associations, not causes |

## 11. Quick Revision

- AI is the broad field; ML is one way to build AI systems; DL is a subset of ML.
- Supervised learning uses known target values; unsupervised learning finds structure without target labels; reinforcement learning learns from reward signals.
- Regression predicts numeric quantities; classification predicts categories.
- Training creates a model; inference uses the trained model on new inputs.
- Evaluation on unseen data is essential before trusting a model.

---

## Expanded Beginner-to-Developer Teaching Layer

# Day 01: Machine Learning, AI, ML categories and real-world use cases

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner foundations  
**Learning focus:** Build understanding from first principles, implement correctly and explain engineering decisions.

---

## 1. Start Here: What Will You Learn?

By the end of this lesson, you should be able to:
- Define the key terms in simple language.
- Explain why the concept is needed and which problem it solves.
- Write and read the core syntax without memorizing blindly.
- Apply the concept to a small practical example.
- Recognize common errors and explain how to investigate them.
- Discuss a realistic business scenario and its limitations.

## 2. Beginner Definition

**Machine Learning, AI, ML categories and real-world use cases**

AI is the broad field of building systems that perform tasks associated with intelligence. ML is a way to learn patterns from examples. Deep learning uses multi-layer neural networks.

### Explain it in everyday language

Think of this as a tool in a developer's toolbox. Before using it, identify the problem, understand what information goes in, what result should come out, and what assumptions are being made. The same technique can be useful in one situation and unsuitable in another.

### Key terms

| Term | Beginner-friendly meaning | Why it matters |
|---|---|---|
| Input | Information supplied to the process | Incorrect input produces unreliable results |
| Output | Result produced by the process | Must match the business requirement |
| Parameter | A value supplied to control an operation or learned by a model, depending on context | Know whether it is configured or learned |
| Validation | Checking whether the result is correct and suitable | Prevents silent errors |
| Generalization | Ability to work on new, unseen cases | Essential for real ML systems |

## 3. Why Do We Need It?

Choose rules when the business rule is stable and explicit; choose ML when patterns are too complex to write manually and representative data exists.

**Without this concept or step:** teams may produce incorrect outputs, waste compute, make poor business decisions or trust a result that does not work on new data. The exact risk depends on the application; investigate rather than assume.

## 4. Syntax, Structure and How to Read It

Do not assume ML is always better, confuse training with prediction, or claim a model understands causation.

### Read the syntax carefully

1. Identify the library or Python feature being used.
2. Identify the input variable and its expected type/shape.
3. Identify the operation or method being called.
4. Check optional parameters and their defaults.
5. Inspect the output and verify that it matches the intended result.

### Practical example

```python
# A small, readable example related to the lesson
task = 'predict customer churn'\nlearning_type = 'supervised classification'\nprint(task)\nprint(learning_type)
```

**Expected result:** The example demonstrates the basic operation. Exact numeric output depends on the input values and library version; run it in the same environment to verify.

### Line-by-line explanation

- The import statement loads the required library or class.
- The input variable stores the data the operation will use.
- The method/function performs the requested operation.
- The print statement displays the result so that it can be checked.
- In a real application, validate inputs and handle expected exceptions rather than assuming every record is valid.

## 5. Worked Examples: From Simple to Practical

### Example A — Understand the smallest case

Start with a tiny input that can be checked manually. Predict what the output should be before running the code. This helps separate a coding error from a misunderstanding of the concept.

### Example B — Change one input

Change one value while keeping the rest constant. Observe which part of the output changes. This is a simple way to understand the effect of a parameter or feature.

### Example C — Check a boundary case

Test an empty input, a missing value, a very small value and a very large value where appropriate. Boundary testing often reveals bugs that normal examples do not show.

## 6. Industry Scenario: Think Like a Senior Developer

**Scenario:** A bank wants to flag suspicious payments. A fixed amount rule misses low-value unusual behaviour; an ML model can learn combinations of amount, time, location and customer history. A false alarm can inconvenience a genuine customer, so predictions need investigation and monitoring.

### How an experienced engineer approaches it

1. Clarify the business objective and what decision the output will support.
2. Identify the data available at the moment the prediction/operation must happen.
3. Check data quality, definitions, units and edge cases.
4. Choose the simplest method that meets the requirement.
5. Validate on representative cases, including difficult cases.
6. Explain uncertainty, limitations and the cost of mistakes.
7. Monitor the result after release; real-world data and business behaviour can change.

### What could go wrong?

undefined

### Senior developer's decision

If the target is a numeric amount, think regression. If it is a named class, think classification. If there are no labels and the goal is grouping, think clustering.

## 7. Common Beginner Mistakes and Corrections

| Mistake | Why it causes trouble | Better approach |
|---|---|---|
| Copying syntax without understanding inputs | Code may run on the wrong data | Check type, shape, units and meaning first |
| Looking only at one successful example | Edge cases remain hidden | Test normal, boundary and invalid inputs |
| Ignoring warnings or unexpected output | Errors can silently affect downstream steps | Inspect output and add assertions/checks |
| Choosing a complex method immediately | Complexity makes debugging and maintenance harder | Establish a simple baseline first |
| Reporting a metric without context | A number alone may hide business failure | Explain the metric, test data and error costs |

## 8. Quick Revision Table

| Question | Key point |
|---|---|
| What is it? | AI is the broad field of building systems that perform tasks associated with intelligence. ML is a way to learn patterns from examples. Deep learning uses multi-layer neural networks. |
| Why use it? | Choose rules when the business rule is stable and explicit; choose ML when patterns are too complex to write manually and representative data exists. |
| Main practical use | A bank wants to flag suspicious payments. A fixed amount rule misses low-value unusual behaviour; an ML model can learn combinations of amount, time, location and customer history. A false alarm can inconvenience a genuine customer, so predictions need investigation and monitoring. |
| Main risk | undefined |
| Engineering principle | Validate assumptions and unseen/boundary cases |

## 9. Check Your Understanding

Try answering without looking at the notes:
1. Define the concept in your own words.
2. What problem does it solve?
3. What are the inputs and expected outputs?
4. When would you avoid using it?
5. What would you test before trusting the result?

## 10. Interview Preparation

Use the matching `day_01_interview.md` file for detailed interview questions, model answers, real-world troubleshooting scenarios and follow-up questions.

---

**Learning standard:** Understand the meaning first, then syntax, then practical use, then engineering trade-offs. A student should be able to explain the idea without reading code from the screen.


---

## Beginner-to-Developer Expansion: More Examples and Real-World Reasoning

### A. Start with the simplest meaning
Before memorizing a function, explain the idea in your own words. Identify the input, output and the reason the operation exists. A correct syntax call on the wrong data is still a wrong solution.

### B. Additional practical scenario
A food-delivery company wants to estimate delivery minutes (regression), predict whether an order will be late (classification), and group restaurants by ordering patterns (clustering). These are different problem definitions, even if they use the same order table.

### C. Syntax and use: how to choose
Use a fixed rule when policy is explicit, such as refunding an order after a contractual delay. Consider ML when patterns are learned from many examples and the decision can be evaluated safely.

### D. What happens if this is skipped?
If the team confuses classification with regression, it may choose an unsuitable target, algorithm and metric.

### E. Senior developer review checklist
- Is the business definition clear?
- Are input types, units, shape and missing values checked?
- Can the result be verified on a small hand-checkable example?
- Have normal, boundary and invalid cases been considered?
- Is the approach simple enough to explain and maintain?
- Are limitations and possible failure costs documented?

### F. Explain it in an interview
Use this order: definition → why it is needed → syntax or method → small example → real business scenario → what can go wrong → how you validate the result. Avoid answering with only a library function name.
