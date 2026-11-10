# Day 29: Interview Questions — Decision Trees and Splitting Criteria

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is a Decision Tree?
**Answer:** A supervised algorithm that predicts by following a sequence of feature-based questions from a root node to a leaf.
**Follow-up:** What does a leaf contain?  
**Answer:** A final class prediction or numerical prediction, depending on the tree type.

### Q2. What are root, branch and leaf nodes?
**Answer:** The root is the first decision point, branches represent outcomes of decisions and leaves are terminal prediction nodes.
**Follow-up:** What is tree depth?  
**Answer:** The number of decision levels along a path from root to leaf.

### Q3. What is Gini impurity?
**Answer:** A measure of how mixed the class labels are within a node. A node containing only one class is pure.
**Follow-up:** What does a useful split do?  
**Answer:** It creates child groups that are more class-pure than the parent node.

### Q4. What is entropy?
**Answer:** A measure of uncertainty in the class distribution. A split can be selected to reduce uncertainty in child nodes.
**Follow-up:** Must you memorize formulas to explain it?  
**Answer:** No. Explain the idea of class mixture and uncertainty.

### Q5. What is pruning?
**Answer:** Reducing tree complexity by removing or restricting branches that do not provide enough useful predictive value.
**Follow-up:** Why prune?  
**Answer:** To reduce overfitting and improve generalization.

## Scenario questions

### Q6. Training accuracy is 100%, but test accuracy is much lower. What may be happening?
**Answer:** The tree may be too deep and overfitting training observations. Check depth, leaf size, leakage and split design.
**Follow-up:** What settings can control complexity?  
**Answer:** max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes and pruning strength.

### Q7. A tree has hundreds of leaves and is difficult to explain. What can you do?
**Answer:** Limit depth, increase minimum leaf size or prune the tree, then validate the trade-off.
**Follow-up:** Does a shallow tree always perform better?  
**Answer:** No. It may underfit. Compare on validation data.

### Q8. A feature has high impurity importance. Does it cause the target?
**Answer:** No. Importance describes how the fitted tree used the feature for splits, not a causal effect.
**Follow-up:** What is one limitation of impurity importance?  
**Answer:** It can favour features with many possible split points and can be affected by correlated features.

### Q9. A tree performs poorly on a minority class. What do you inspect?
**Answer:** Class counts, confusion matrix, class weights, split strategy and whether the features contain useful information for that class.
**Follow-up:** Is accuracy enough?  
**Answer:** No. Inspect precision, recall and business costs.

### Q10. Does a Decision Tree require feature scaling?
**Answer:** Ordinary trees are generally not sensitive to monotonic feature scaling because splits depend on feature ordering.
**Follow-up:** Does that mean data cleaning is unnecessary?  
**Answer:** No. Invalid values, leakage and inconsistent categories still need attention.

## Code interpretation

### Q11. What does max_depth=3 control?
**Answer:** It limits the maximum number of decision levels in the tree.
**Follow-up:** What is a trade-off?  
**Answer:** A shallow tree is easier to interpret but may underfit.

### Q12. What does min_samples_leaf do?
**Answer:** It sets the minimum number of training observations allowed in a terminal leaf.
**Follow-up:** Why can a larger value help?  
**Answer:** It prevents very small leaves that may represent noise or individual records.

**Senior developer principle:** A tree that explains every training row may simply have memorized the data. Validate complexity on unseen observations.
