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