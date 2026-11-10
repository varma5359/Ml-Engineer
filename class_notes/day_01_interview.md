# Day 01: Interview Questions — Machine Learning, AI, ML Categories and Real-World Use Cases

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. Explain Artificial Intelligence, Machine Learning and Deep Learning.
**Answer:** Artificial Intelligence is the broad field of building systems that perform tasks associated with intelligence. Machine Learning is a subset of AI where systems learn patterns from data. Deep Learning is a subset of ML that uses multi-layer neural networks.
**Follow-up:** Can an AI system work without ML?  
**Answer:** Yes. A rule-based expert system can use predefined conditions without learning from training data.

### Q2. What is supervised learning?
**Answer:** Supervised learning trains from examples containing input features and known target labels or values.
**Follow-up:** Give a regression and classification example.  
**Answer:** House-price prediction is regression; churn prediction is classification.

### Q3. What is unsupervised learning?
**Answer:** It works with data that does not have target labels and discovers structure such as groups or unusual patterns.
**Follow-up:** Is clustering classification?  
**Answer:** No. Classification learns known labels; clustering discovers groups without predefined labels.

### Q4. What is reinforcement learning?
**Answer:** It trains an agent to choose actions through interaction with an environment and feedback such as rewards or penalties.
**Follow-up:** Give an example.  
**Answer:** A system learning a game strategy through repeated interaction.

### Q5. What is the difference between regression and classification?
**Answer:** Regression predicts a numerical quantity; classification predicts a category or class.
**Follow-up:** Is predicting whether a loan is approved regression?  
**Answer:** No. Approval is a category, so it is classification.

## Real-world scenarios

### Q6. A company wants to predict house prices. Which ML task is suitable?
**Answer:** Regression, because the output is a numerical price.
**Follow-up:** What if the company wants low, medium and premium categories?  
**Answer:** Classification, because the output is a category.

### Q7. A bank wants to identify suspicious transactions but has no reliable labels. What could it try?
**Answer:** Anomaly detection or unsupervised methods to identify unusual patterns for investigation. If reliable labels become available, supervised classification may also be suitable.
**Follow-up:** Should every flagged transaction automatically be blocked?  
**Answer:** Not necessarily. A flag is a risk signal, not proof. The bank should consider false alerts, missed fraud and review controls.

### Q8. When would you avoid machine learning?
**Answer:** When a simple rule is adequate, data is insufficient or unreliable, success cannot be measured, risk cannot be managed or model cost exceeds expected benefit.
**Follow-up:** What could you use instead?  
**Answer:** A documented rule-based process or manual review while collecting reliable data.

### Q9. A model reports 99% accuracy. Does that always mean it is good?
**Answer:** No. If only 1% of transactions are fraudulent, predicting every transaction as legitimate can achieve 99% accuracy while detecting no fraud.
**Follow-up:** What should you inspect?  
**Answer:** Confusion matrix, precision, recall, F1-score and the business costs of different errors.

### Q10. What is the difference between an algorithm and a model?
**Answer:** An algorithm is the learning procedure. A model is the result after that procedure has learned parameters from training data.
**Follow-up:** What are hyperparameters?  
**Answer:** Settings selected before or around training, such as a decision tree's maximum depth.

## Interview answer framework

When asked whether ML is suitable:
1. Clarify the business decision.
2. Identify the required output.
3. Check whether reliable data and labels exist.
4. Choose a suitable learning approach.
5. Define how performance and risk will be evaluated.
6. Compare against a simple baseline.

**Senior developer principle:** Choose the learning approach from the problem and data, not from the popularity of an algorithm.
