# Day 21: Gradient Descent and Learning Rate

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define gradient descent and loss.
- Explain parameters, gradients and learning rate.
- Understand how parameter updates reduce loss.
- Recognize too-small and too-large learning rates.

## 1. What Is Gradient Descent?
Gradient descent is an optimization method that adjusts model parameters in a direction that reduces the loss function. It is used in many ML and deep-learning training procedures.

**Key terms**
- Parameter: a value the model learns, such as a weight.
- Loss: a measure of prediction error.
- Gradient: indicates how loss changes as a parameter changes.
- Learning rate: controls the size of a parameter update.

## 2. Update Rule
New parameter = old parameter - (learning rate × gradient).

```python
parameter = 2.0
gradient = 0.5
learning_rate = 0.1
parameter = parameter - learning_rate * gradient
print(parameter)
```
Output: 1.95.

## 3. Three Learning-Rate Scenarios
1. Very small learning rate: updates are tiny; training may take a long time.
2. Very large learning rate: updates can jump over a minimum and loss may become unstable.
3. Reasonable learning rate: loss may decrease steadily, depending on the model and data.

## 4. Scenario Questions
1. Training loss barely changes. What do you inspect? Learning rate, feature scale, gradients and implementation.
2. Loss increases dramatically after each update. What may be wrong? Learning rate may be too large or gradients/data may be problematic.
3. Can one learning rate work for every model and dataset? No; it must be selected and validated.
