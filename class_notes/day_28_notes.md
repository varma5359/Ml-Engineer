# Day 28: Naive Bayes and Conditional Probability

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain Bayes' theorem at a high level.
- Define the Naive Bayes independence assumption.
- Compare common Naive Bayes variants.
- Apply Naive Bayes to simple text classification.

## 1. Definition
Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem. It uses a simplifying assumption that features are conditionally independent given the class.

Bayes' theorem in words: posterior probability is proportional to likelihood multiplied by prior probability.

## 2. Common Variants
- MultinomialNB: commonly used with non-negative word counts or frequencies.
- BernoulliNB: commonly used with binary feature indicators.
- GaussianNB: models continuous features using class-conditional Gaussian distributions.

## 3. Syntax
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_counts, y_train)
predictions = model.predict(X_test_counts)
```

## 4. Three Examples
1. Spam email classification using word counts.
2. News topic classification using term frequencies.
3. Sentiment classification using word-presence features.

## 5. Scenario Questions
1. MultinomialNB receives negative feature values. What is wrong? MultinomialNB expects non-negative count/frequency-like inputs.
2. Words in a sentence are related. Does the independence assumption hold perfectly? Usually not; it is a simplifying assumption.
3. Naive Bayes is a strong baseline but not the best model. Is it still useful? Yes, speed and simplicity can make it a valuable benchmark.
