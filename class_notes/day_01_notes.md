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