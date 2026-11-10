# Day 28: Naive Bayes and Conditional Probability

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain conditional probability in plain language.
- Understand Bayes' theorem conceptually without memorizing formulas.
- Explain the Naive Bayes assumption.
- Distinguish Gaussian, Multinomial and Bernoulli Naive Bayes.
- Train GaussianNB for numerical features.
- Train MultinomialNB for text-count features.
- Use predict() and predict_proba().
- Understand why the independence assumption is called naive.
- Recognize strengths and limitations of Naive Bayes.
- Compare a Naive Bayes model with a baseline.

## 2. What is conditional probability?

**Conditional probability** describes the probability of an event when another piece of information is already known.

Examples:
- Probability that an email is spam given that it contains certain words.
- Probability that a customer churns given their usage pattern.
- Probability that a transaction is fraudulent given its observed characteristics.

The known information changes what we consider likely.

## 3. Bayes' theorem in plain language

Bayes' theorem provides a way to update an estimate when new evidence becomes available.

A beginner-friendly interpretation:
1. Start with how common an event is before seeing the new evidence.
2. Consider how likely the evidence is when the event is true.
3. Consider how likely the same evidence is overall.
4. Update the event probability using the evidence.

In classification, a model compares how compatible the observed features are with each possible class.

You do not need to memorize a mathematical formula for this lesson. Focus on how prior class frequency and feature evidence contribute to a prediction.

## 4. What is Naive Bayes?

**Naive Bayes** is a family of probabilistic classification algorithms based on Bayes' theorem.

It is called "naive" because it makes a simplifying assumption that features are conditionally independent given the class.

Example:
- For spam classification, the model may treat the presence of "offer" and "prize" as separate pieces of evidence once the email class is known.
- In real messages, words are not truly independent.
- Even with this simplifying assumption, Naive Bayes can work well for many text classification tasks.

## 5. Types of Naive Bayes

| Algorithm | Typical data | Example |
|---|---|---|
| GaussianNB | Continuous numerical features | Sensor readings or measurements |
| MultinomialNB | Non-negative counts or frequencies | Word-count features in text |
| BernoulliNB | Binary features | Whether a word is present or absent |
| ComplementNB | Imbalanced text classification settings | Some document classification tasks |

Choose the variant based on how the input features are represented.

## 6. Gaussian Naive Bayes for numerical features

GaussianNB is designed for continuous numerical features. It models the distribution of each feature within each class using a Gaussian assumption.

### 6.1 Import libraries

### 6.1. Import Pandas, NumPy and plotting libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

These libraries prepare and inspect the numerical dataset.



### 6.2 Create a customer risk dataset

### 6.2. Create numerical feature observations

```python
df = pd.DataFrame({
    "monthly_spend": [1200, 1500, 1800, 2100, 2400, 2800, 3200, 3600, 4000, 4500,
                      5000, 5400, 5800, 6200, 6800, 7200, 7600, 8000, 8500, 9000],
    "support_calls": [5, 4, 5, 4, 3, 4, 3, 3, 2, 2,
                      2, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    "tenure_months": [2, 3, 4, 5, 6, 8, 10, 12, 15, 18,
                      20, 24, 28, 30, 34, 36, 40, 44, 48, 54],
    "churn": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
})

print(df.head())
```

**Expected output**

```text
The first five rows of a small numerical churn dataset
```

**Explanation**

The values are illustrative and intentionally show class patterns. They are not representative of a real customer population.



### 6.3 Inspect the target classes

### 6.3. Count churn labels

```python
print(df["churn"].value_counts())
```

**Expected output**

```text
Counts for churn classes 0 and 1
```

**Explanation**

Check the number of observations in each class before training.



### 6.4 Separate features and target

### 6.4. Create X and y

```python
X = df[["monthly_spend", "support_calls", "tenure_months"]]
y = df["churn"]

print(X.shape, y.shape)
```

**Expected output**

```text
(20, 3) and (20,)
```

**Explanation**

The three numerical columns are input features and churn is the target.



## 7. Split the data

### 7.1 Import train_test_split

### 7.1. Import the split function

```python
from sklearn.model_selection import train_test_split

print("Split function ready")
```

**Expected output**

```text
Split function ready
```

**Explanation**

The function creates training and test partitions.



### 7.2 Create a stratified split

### 7.2. Split features and target

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)
print(y_train.value_counts())
print(y_test.value_counts())
```

**Expected output**

```text
Training and test shapes with class counts
```

**Explanation**

Stratification keeps class proportions approximately consistent where sample size permits.



## 8. Train Gaussian Naive Bayes

### 8.1 Import GaussianNB

### 8.1. Import GaussianNB

```python
from sklearn.naive_bayes import GaussianNB

print("GaussianNB imported")
```

**Expected output**

```text
GaussianNB imported
```

**Explanation**

GaussianNB is used for continuous numerical features under its distribution assumption.



### 8.2 Create the model

### 8.2. Initialize GaussianNB

```python
gnb = GaussianNB()

print(gnb)
```

**Expected output**

```text
GaussianNB()
```

**Explanation**

The classifier is initialized but not fitted.



### 8.3 Fit the model

### 8.3. Train GaussianNB

```python
gnb.fit(X_train, y_train)

print("GaussianNB fitted")
```

**Expected output**

```text
GaussianNB fitted
```

**Explanation**

The model learns class frequencies and feature-distribution information from training observations.



### 8.4 Predict class labels

### 8.4. Generate test predictions

```python
gnb_predictions = gnb.predict(X_test)

print(gnb_predictions)
```

**Expected output**

```text
Predicted churn labels
```

**Explanation**

The classifier combines evidence from the input features and predicts the most likely class.



### 8.5 Predict class probabilities

### 8.5. Generate class probabilities

```python
gnb_probabilities = gnb.predict_proba(X_test)

print(gnb_probabilities)
```

**Expected output**

```text
A probability matrix with one column per class
```

**Explanation**

The columns follow the order in gnb.classes_. These probabilities should be assessed for calibration if they drive decisions.



### 8.6 Inspect class order

### 8.6. Read the class labels

```python
print(gnb.classes_)
```

**Expected output**

```text
[0 1]
```

**Explanation**

This identifies which probability column corresponds to each target class.



## 9. Evaluate GaussianNB

### 9.1 Calculate accuracy

### 9.1. Calculate accuracy

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, gnb_predictions))
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

Accuracy is a basic summary. It can hide poor performance on a minority class.



### 9.2 Print class-level metrics

### 9.2. Review classification report

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    gnb_predictions,
    zero_division=0
))
```

**Expected output**

```text
Precision, recall, F1 and support for each class
```

**Explanation**

The report helps identify which class is difficult for the model.



### 9.3 Plot confusion matrix

### 9.3. Create a confusion matrix

```python
from sklearn.metrics import confusion_matrix

gnb_cm = confusion_matrix(y_test, gnb_predictions)

sns.heatmap(gnb_cm, annot=True, fmt="d")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title("GaussianNB confusion matrix")
plt.show()
```

**Expected output**

```text
A two-class confusion matrix heatmap
```

**Explanation**

The confusion matrix shows correct and incorrect class assignments.



## 10. Multinomial Naive Bayes for text classification

MultinomialNB is commonly used with non-negative feature counts or frequencies. A standard teaching example is spam detection using word counts.

### 10.1 Create a small text dataset

### 10.1. Create message text and labels

```python
messages = [
    "win a free prize now",
    "claim your cash reward",
    "limited offer click now",
    "meeting scheduled for tomorrow",
    "please review the project report",
    "team meeting starts at ten",
    "free gift claim your prize",
    "send the updated report today",
    "cash offer win now",
    "please join the project meeting"
]

labels = [
    "spam", "spam", "spam",
    "ham", "ham", "ham",
    "spam", "ham", "spam", "ham"
]

text_df = pd.DataFrame({
    "message": messages,
    "label": labels
})

print(text_df.head())
```

**Expected output**

```text
A table containing message text and spam/ham labels
```

**Explanation**

The dataset is small and constructed for teaching. Real text classifiers need much larger and representative data.



### 10.2 Inspect label counts

### 10.2. Count spam and ham examples

```python
print(text_df["label"].value_counts())
```

**Expected output**

```text
Counts for spam and ham
```

**Explanation**

Check class distribution before splitting.



### 10.3 Split text and labels

### 10.3. Create train and test text sets

```python
from sklearn.model_selection import train_test_split

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    text_df["message"],
    text_df["label"],
    test_size=0.3,
    random_state=42,
    stratify=text_df["label"]
)

print(len(X_train_text), len(X_test_text))
```

**Expected output**

```text
Seven training messages and three test messages
```

**Explanation**

The labels remain paired with their original text records.



### 10.4 Convert text to word-count features

### 10.4. Use CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train_counts = vectorizer.fit_transform(X_train_text)

print(X_train_counts.shape)
```

**Expected output**

```text
A sparse matrix shape showing training documents and learned vocabulary size
```

**Explanation**

CountVectorizer learns its vocabulary from training text and represents each document using word counts.



### 10.5 Inspect the vocabulary

### 10.5. Read learned word features

```python
print(vectorizer.get_feature_names_out())
```

**Expected output**

```text
A list of words learned from training messages
```

**Explanation**

The vocabulary is learned from training data only. Words found only in test messages are not added during transform.



### 10.6 Transform test messages

### 10.6. Convert test messages using the fitted vectorizer

```python
X_test_counts = vectorizer.transform(X_test_text)

print(X_test_counts.shape)
```

**Expected output**

```text
A sparse matrix with the same number of vocabulary columns
```

**Explanation**

Use transform() for test messages so the vocabulary is not learned from held-out text.



### 10.7 Train MultinomialNB

### 10.7. Import and fit MultinomialNB

```python
from sklearn.naive_bayes import MultinomialNB

multinomial_model = MultinomialNB()

multinomial_model.fit(X_train_counts, y_train_text)

print("MultinomialNB fitted")
```

**Expected output**

```text
MultinomialNB fitted
```

**Explanation**

The model learns class and word-count patterns from the training messages.



### 10.8 Predict test messages

### 10.8. Generate spam/ham predictions

```python
text_predictions = multinomial_model.predict(X_test_counts)

print(text_predictions)
```

**Expected output**

```text
Predicted spam or ham labels
```

**Explanation**

The classifier uses word-count evidence to estimate the most likely class.



### 10.9 Predict class probabilities

### 10.9. Generate text-class probabilities

```python
text_probabilities = multinomial_model.predict_proba(X_test_counts)

print(text_probabilities)
```

**Expected output**

```text
A probability matrix for the test messages
```

**Explanation**

The columns correspond to multinomial_model.classes_.



### 10.10 Evaluate text classification

### 10.10. Print text classification report

```python
print(classification_report(
    y_test_text,
    text_predictions,
    zero_division=0
))
```

**Expected output**

```text
Precision, recall, F1 and support for spam and ham
```

**Explanation**

A tiny sample is not enough for reliable performance claims. This demonstrates the workflow only.



## 11. Bernoulli Naive Bayes

BernoulliNB is designed for binary feature indicators, such as whether a word is present or absent.

### 11.1 Convert text to binary word-presence features

### 11.1. Use binary CountVectorizer

```python
binary_vectorizer = CountVectorizer(binary=True)

X_train_binary = binary_vectorizer.fit_transform(X_train_text)
X_test_binary = binary_vectorizer.transform(X_test_text)

print(X_train_binary.shape)
```

**Expected output**

```text
A sparse matrix with binary word-presence features
```

**Explanation**

binary=True stores whether a term appears rather than how many times it appears.



### 11.2 Train BernoulliNB

### 11.2. Fit BernoulliNB

```python
from sklearn.naive_bayes import BernoulliNB

bernoulli_model = BernoulliNB()

bernoulli_model.fit(X_train_binary, y_train_text)

print("BernoulliNB fitted")
```

**Expected output**

```text
BernoulliNB fitted
```

**Explanation**

The classifier is trained using binary feature indicators.



### 11.3 Predict test messages

### 11.3. Predict with BernoulliNB

```python
bernoulli_predictions = bernoulli_model.predict(X_test_binary)

print(bernoulli_predictions)
```

**Expected output**

```text
Spam/ham predictions
```

**Explanation**

The model uses word presence or absence as its evidence representation.



## 12. Understand the independence assumption

Naive Bayes assumes features are conditionally independent given the class.

In natural language, words often depend on one another:
- "not good" has a different meaning from "good".
- "free prize" may have a different pattern from either word alone.
- Word order can change meaning.

The independence assumption is simplified, but Naive Bayes can still be useful because:
- It is fast to train.
- It can work well with high-dimensional sparse text data.
- It provides a strong baseline for some classification tasks.
- It is relatively simple to explain.

## 13. Compare Naive Bayes with a baseline

### 13.1 Create a majority baseline

### 13.1. Fit DummyClassifier

```python
from sklearn.dummy import DummyClassifier

text_baseline = DummyClassifier(strategy="most_frequent")

text_baseline.fit(X_train_counts, y_train_text)

print("Text baseline fitted")
```

**Expected output**

```text
Text baseline fitted
```

**Explanation**

The baseline predicts the most frequent training label and provides a simple reference.



### 13.2 Compare baseline and Naive Bayes

### 13.2. Compare accuracy

```python
baseline_text_pred = text_baseline.predict(X_test_counts)

print("Baseline:", accuracy_score(y_test_text, baseline_text_pred))
print("Naive Bayes:", accuracy_score(y_test_text, text_predictions))
```

**Expected output**

```text
Two accuracy values
```

**Explanation**

Use the same test messages. On a tiny dataset, this comparison is only a teaching illustration.



## 14. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Assume all Naive Bayes variants are interchangeable | Each expects a different feature representation | Match variant to data |
| Use negative values with MultinomialNB | Standard MultinomialNB expects non-negative features | Use count/frequency representations |
| Fit text vectorizer on all data | Test vocabulary influences preprocessing | Fit on training text only |
| Assume features are truly independent | Real data often has dependencies | Treat independence as a simplifying assumption |
| Use accuracy alone | Class imbalance can hide failures | Inspect class-level metrics |
| Treat probabilities as guaranteed outcomes | Estimates may be poorly calibrated | Evaluate calibration if probabilities drive decisions |
| Use tiny examples to claim business performance | Results are unstable | Collect representative evaluation data |

## 15. Interview-ready explanations

**What is Naive Bayes?**  
It is a family of probabilistic classifiers based on Bayes' theorem and a simplifying conditional-independence assumption between features given the class.

**Why is it called naive?**  
Because the feature-independence assumption is often unrealistic in real datasets, but it makes the model simple and efficient.

**When would you use GaussianNB?**  
For continuous numerical features when the Gaussian feature-distribution assumption is a reasonable starting point.

**When would you use MultinomialNB?**  
For non-negative count or frequency features, commonly word-count vectors in text classification.

**What is the difference between MultinomialNB and BernoulliNB?**  
MultinomialNB uses counts or frequencies. BernoulliNB uses binary indicators such as whether a word is present.

## 16. Scenario questions

1. You are building a spam classifier from word counts. Which Naive Bayes variant is a common starting point?  
   **Answer:** MultinomialNB.

2. The features indicate only whether a word appears. Which variant could be suitable?  
   **Answer:** BernoulliNB.

3. Your GaussianNB model performs poorly. What do you investigate?  
   **Answer:** Feature distributions, data quality, class imbalance, feature relevance and whether the Gaussian assumption is suitable.

4. Why can Naive Bayes work even when words are not independent?  
   **Answer:** The independence assumption is simplified, but the resulting probability ranking can still be useful for some tasks.

5. A spam classifier has high accuracy but misses most spam messages. What do you inspect?  
   **Answer:** Class distribution, confusion matrix, spam recall, threshold and error costs.

## 17. Lesson completion checklist

- [ ] I can explain conditional probability conceptually.
- [ ] I understand the Naive Bayes independence assumption.
- [ ] I can distinguish Gaussian, Multinomial and Bernoulli Naive Bayes.
- [ ] I can train GaussianNB on numerical features.
- [ ] I can vectorize text using CountVectorizer.
- [ ] I can train MultinomialNB and BernoulliNB.
- [ ] I can inspect probabilities and class order.
- [ ] I can compare against a baseline.
- [ ] I understand common limitations.

**Next lesson:** Day 29 — Decision Trees and Splitting Criteria.
