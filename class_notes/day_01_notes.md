# Day 01: Machine Learning, AI, ML Categories and Real-World Use Cases

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define Artificial Intelligence, Machine Learning and Deep Learning.
- Explain how an ML model learns from examples.
- Distinguish supervised, unsupervised and reinforcement learning.
- Identify classification, regression, clustering and anomaly detection tasks.
- Select an appropriate ML problem type from a business requirement.

## 1. What Is Artificial Intelligence?
**Artificial Intelligence (AI)** is the broad field of building computer systems that perform tasks that normally require human-like capabilities, such as understanding language, recognizing images, planning or making decisions.

**Examples**
1. A voice assistant understands a spoken request.
2. A camera application recognizes objects in an image.
3. A route planner selects a route based on traffic and road conditions.

AI does not always mean that a system learns from data. A rule-based system can also be an AI application.

## 2. What Is Machine Learning?
**Machine Learning (ML)** is a branch of AI in which a computer system learns patterns from data and uses those patterns to make predictions or decisions on new inputs.

Instead of writing every rule manually, we provide examples and a learning algorithm.

**Basic workflow:** Historical data → learning algorithm → trained model → new input → prediction.

**Examples**
1. An email filter learns patterns from previously labelled spam and normal emails.
2. A bank model learns from past transactions to estimate whether a new transaction is suspicious.
3. A retailer learns from sales history to estimate next month's demand.

**Without ML:** Developers may need to write many manual rules. Those rules can be difficult to maintain when patterns are complex or frequently change. ML is not automatically better: if a simple rule solves a stable problem, a rule may be safer and easier.

## 3. AI vs ML vs Deep Learning
| Term | Meaning | Example |
|---|---|---|
| AI | Broad field of intelligent computer behaviour | Route planning system |
| ML | Learns patterns from data | Customer churn prediction |
| Deep Learning | ML methods based on multi-layer neural networks | Image recognition |

Deep Learning is a subset of ML, and ML is a subset of AI.

## 4. Main Categories of Machine Learning

### 4.1 Supervised Learning
**Definition:** Learning from labelled examples. Each training record contains input features and a known target.

**Syntax / representation:** X = input features; y = known target. A model is trained using model.fit(X_train, y_train).

**Examples**
1. Predict house price from area and location: regression.
2. Predict whether a customer will churn: classification.
3. Predict whether an email is spam: classification.

**Without labels:** A standard supervised model cannot learn the intended target in the usual way. The target must be defined and reliable.

### 4.2 Unsupervised Learning
**Definition:** Finding patterns or structure in data when the training examples do not include a target label.

**Examples**
1. Group customers by purchasing behaviour: clustering.
2. Group products by similarity: clustering.
3. Find unusual transactions that differ from normal patterns: anomaly detection.

The algorithm discovers structure; the developer must interpret whether that structure is useful.

### 4.3 Reinforcement Learning
**Definition:** An agent learns which actions to take by interacting with an environment and receiving rewards or penalties.

**Examples**
1. A game-playing agent learns moves that increase its score.
2. A robot learns how to move while avoiding obstacles.
3. A traffic-control agent learns signal strategies to reduce congestion.

Reinforcement learning is not simply training on a labelled spreadsheet. It involves actions, states, rewards and interaction.

## 5. Common ML Problem Types
| Problem | What is predicted or discovered? | Example |
|---|---|---|
| Regression | A numeric value | Delivery time in minutes |
| Classification | A class/category | Fraud or legitimate |
| Clustering | Groups without supplied labels | Customer segments |
| Anomaly detection | Unusual observations | Abnormal sensor reading |

### Three examples: identify the problem
1. Predict next week's electricity usage in kWh → regression.
2. Predict whether a patient appointment will be missed → classification.
3. Group shoppers by buying patterns without pre-existing group labels → clustering.

## 6. How a Model Learns
1. Collect relevant historical examples.
2. Identify input features and, for supervised learning, the target.
3. Split data into training and evaluation sets.
4. Train an algorithm to learn patterns.
5. Evaluate predictions on data not used for fitting.
6. Deploy only if performance and operational requirements are acceptable.
7. Monitor results because data and behaviour can change.

**Important terms**
- Feature: input information used by the model.
- Target: the value/class to predict in supervised learning.
- Training: process of learning model parameters from data.
- Prediction/inference: using a trained model on new input.
- Evaluation: measuring how well predictions meet the task requirements.

## 7. Short Industry Scenarios
1. A bank wants to estimate the amount of a future loan default loss. Which problem type is this? **Regression**, if the target is a numeric loss amount.
2. A telecom company wants to identify customers likely to leave. Which problem type is this? **Classification**, if the target is churn yes/no.
3. A shop has no customer labels and wants to discover groups with similar shopping habits. Which category? **Unsupervised learning / clustering**.

## 8. Common Beginner Mistakes
- Calling every AI system machine learning.
- Assuming ML always requires deep learning.
- Confusing regression with classification.
- Evaluating a model only on training data.
- Assuming a model prediction proves cause and effect.
- Ignoring the cost of false predictions.

## 9. Quick Revision
AI is the broad field; ML learns patterns from data; Deep Learning uses multi-layer neural networks. First define the business target, then select the learning category and problem type.
