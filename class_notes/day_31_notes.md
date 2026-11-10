# Day 31: Support Vector Machines and Maximum-Margin Hyperplanes

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define Support Vector Machine (SVM).
- Explain hyperplane, margin and support vectors.
- Distinguish linear and kernel SVM.
- Understand C and gamma at a beginner level.
- Recognize the importance of feature scaling.

## 1. What Is SVM?
A Support Vector Machine is a supervised learning algorithm used for classification and regression. In classification, it tries to find a decision boundary that separates classes while maximizing the margin between the boundary and the nearest training points.

- Hyperplane: the decision boundary.
- Margin: distance between the boundary and the closest points from each class.
- Support vectors: training points that help determine the boundary.

## 2. Linear and Kernel SVM
A linear SVM uses a linear boundary. A kernel allows the algorithm to model more complex boundaries by working with similarities between observations.

**Three examples**
1. Classify documents into two categories using text features.
2. Classify product measurements into quality classes.
3. Classify customers as likely/not likely to respond using numerical behaviour features.

## 3. Syntax
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

- kernel: boundary type, such as linear or rbf.
- C: controls the penalty for classification errors; larger C tends to fit training points more strictly.
- gamma: controls influence range for RBF kernel.
- Scaling is important because SVM is sensitive to feature magnitudes.

## 4. Scenario Questions
1. SVM performs poorly when income is in lakhs and age in years. What do you try? Scale numeric features.
2. Training accuracy is high but validation is low. What do you inspect? C, gamma, kernel, split and overfitting.
3. Dataset is very large and training is slow. What should you consider? Runtime cost and whether another model is more practical.
