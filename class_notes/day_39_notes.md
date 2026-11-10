# Day 39: Association Rules and Market Basket Analysis

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define market basket analysis and association rules.
- Explain support, confidence and lift.
- Interpret a rule A → B.
- Recognize that association is not causation.
- Identify business uses and limitations.

## 1. What Is Market Basket Analysis?
Market basket analysis examines which products occur together in transactions. It can identify purchasing patterns that may support product placement, bundles or recommendations.

## 2. Association Rule
A rule A → B means that transactions containing A are examined for how often they also contain B.

- Support: fraction of all transactions containing A and B.
- Confidence: among transactions containing A, fraction that also contain B.
- Lift: confidence divided by the overall support of B.

Lift greater than 1 indicates B appears with A more often than its overall frequency would suggest, under the data analysed.

## 3. Three Examples
1. Customers buying pasta also often buy pasta sauce.
2. Customers buying a printer often buy ink.
3. Customers buying a phone may also buy a case.

These are illustrative examples; actual rules must be computed from transaction data.

## 4. Scenario Questions
1. Confidence is high because B is purchased by almost everyone. What else should you inspect? Support and lift.
2. A rule says pasta buyers also buy sauce. Does pasta cause sauce purchases? No.
3. A rule has statistical strength but low margin/stock availability. Is it automatically actionable? No; check commercial constraints.
