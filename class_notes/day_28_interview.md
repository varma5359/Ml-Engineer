# Day 28: Interview Questions — Naive Bayes and Conditional Probability

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is conditional probability?
**Answer:** It describes the probability of an event when some other information is already known.
**Follow-up:** Give an ML example.  
**Answer:** The probability that an email is spam given that it contains particular words.

### Q2. What is Naive Bayes?
**Answer:** It is a family of probabilistic classifiers based on Bayes' theorem and a simplifying assumption that features are conditionally independent given the class.
**Follow-up:** Why is the assumption called naive?  
**Answer:** Real-world features are often related, so the assumption is a simplification.

### Q3. Why can Naive Bayes still work when features are not independent?
**Answer:** The simplification can still produce useful class rankings or predictions in some tasks, especially text classification. Its suitability must be evaluated empirically.
**Follow-up:** Does it always outperform other classifiers?  
**Answer:** No. Performance depends on the dataset and feature representation.

### Q4. Compare GaussianNB, MultinomialNB and BernoulliNB.
**Answer:** GaussianNB is used for continuous numerical features under a Gaussian assumption. MultinomialNB is common for non-negative counts or frequencies. BernoulliNB uses binary feature indicators.
**Follow-up:** Which is a common starting point for word-count text features?  
**Answer:** MultinomialNB.

## Scenario questions

### Q5. You are building a spam classifier from word counts. Which variant is a common starting point?
**Answer:** MultinomialNB.
**Follow-up:** What if the features only show whether each word appears?  
**Answer:** BernoulliNB may be suitable.

### Q6. GaussianNB performs poorly. What do you investigate?
**Answer:** Feature distributions, data quality, class balance, feature relevance and whether the Gaussian assumption is a reasonable approximation.
**Follow-up:** Should you immediately discard Naive Bayes?  
**Answer:** Compare it with a baseline and other suitable models using the same evaluation data.

### Q7. A spam classifier has high accuracy but misses most spam messages. What do you inspect?
**Answer:** Class distribution, confusion matrix, spam recall, threshold and the cost of missed spam.
**Follow-up:** Why is accuracy insufficient?  
**Answer:** It can hide poor performance on the class that matters most.

### Q8. A word appears frequently in both spam and legitimate messages. What should you expect?
**Answer:** That word alone may provide limited class-discriminating evidence. The classifier combines evidence from multiple features.
**Follow-up:** Does Naive Bayes understand word order?  
**Answer:** A basic bag-of-words MultinomialNB pipeline does not represent word order.

### Q9. Why fit CountVectorizer on training text only?
**Answer:** To prevent the test vocabulary from influencing preprocessing. Fit the vocabulary on training messages and transform held-out messages using that vocabulary.
**Follow-up:** What happens to a test-only word?  
**Answer:** It is not added to the learned vocabulary and does not create a new feature during transform.

## Code interpretation

### Q10. What does predict_proba() return?
**Answer:** Estimated probabilities for each class, in the order shown by classes_.
**Follow-up:** Are those probabilities guaranteed to be calibrated?  
**Answer:** No. Calibration should be assessed if probabilities drive important decisions.

### Q11. Why should MultinomialNB features be non-negative?
**Answer:** Its feature model is designed for count or frequency-like non-negative inputs. Standard centered/scaled values can contain negatives and may be unsuitable.

**Senior developer principle:** Select the Naive Bayes variant from the feature representation, then validate the simplifying assumptions against real data.
