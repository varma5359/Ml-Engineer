# Day 59: Interview Questions — Recommendation System Fundamentals

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is a recommendation system?
**Answer:**
- It ranks items that may be relevant to a user or context.
- Examples include products, films, courses and music.
- It uses item information, interactions or both.
- Its output is a ranked list, not a guarantee of preference.

### Q2. Explain popularity-based recommendation.
**Answer:**
- It ranks items using overall interactions, purchases or another signal.
- It is simple and useful when user history is missing.
- It can over-promote already popular items.
- It usually provides similar rankings to many users.

### Q3. What is content-based filtering?
**Answer:**
- It recommends items similar to those a user liked or interacted with.
- It uses item attributes such as category, description or tags.
- TF-IDF and cosine similarity are possible techniques.
- It needs useful metadata and may over-specialize.

### Q4. What is collaborative filtering?
**Answer:**
- It learns from user-item interaction patterns.
- User-based methods find similar users.
- Item-based methods find items with similar interaction patterns.
- Matrix factorization learns latent representations.

### Q5. What is a hybrid recommender?
**Answer:**
- It combines multiple recommendation strategies.
- It can reduce weaknesses of individual methods.
- Scores and candidate lists need careful combination.
- It should be evaluated against clear product goals.

### Q6. Explain cold start.
**Answer:**
- A new user or item has little or no interaction history.
- Collaborative filtering may struggle.
- Popularity, content-based methods and onboarding preferences can help.
- Hybrid systems often use fallback logic.

## 2. Data and evaluation

### Q7. What is a user-item matrix?
**Answer:**
- Rows represent users and columns represent items.
- Cells contain ratings or interactions.
- Missing cells generally mean no observed interaction.
- Such matrices are often sparse.

### Q8. What is Precision@K?
**Answer:**
- It measures the fraction of the top K recommendations that are relevant.
- It focuses on displayed top results.
- The relevance definition must be clear.
- It does not measure all relevant items retrieved.

### Q9. What is Recall@K?
**Answer:**
- It measures the fraction of all relevant items retrieved in top K.
- It complements precision.
- Higher recall may come with lower precision.
- Choose metrics based on product goals.

### Q10. Why is random interaction splitting risky?
**Answer:**
- It can allow future interactions into training.
- It may leak information unavailable at recommendation time.
- Use time-based splitting when evaluating future recommendations.
- Match offline evaluation to the serving scenario.

## 3. Practical scenarios

### Q11. A new visitor has no account. How can you recommend products?
**Answer:**
- Use popular or trending products.
- Use current session context if permitted.
- Use search or category context.
- Ask for preferences only when useful.
- Do not pretend to know personal preferences.

### Q12. A new product has no interactions. How can it be recommended?
**Answer:**
- Use product metadata and content similarity.
- Use category-level popularity.
- Consider controlled exploration or editorial placement.
- Collect interactions and evaluate.

### Q13. Users see the same popular products repeatedly. What do you investigate?
**Answer:**
- Popularity bias and feedback loops.
- Catalog coverage and diversity.
- Ranking and candidate-generation logic.
- Exposure and outcome differences.
- Controlled experiments with guardrails.

### Q14. How would you build a basic product recommender?
**Answer:**
1. Define catalog and interaction events.
2. Create a popularity baseline.
3. Prepare item descriptions.
4. Vectorize with TF-IDF.
5. Calculate cosine similarity.
6. Return top-K alternatives.
7. Exclude unavailable and selected items.
8. Evaluate with time-aware ranking metrics.

## 4. Responsible ML

### Q15. What privacy concerns exist?
**Answer:**
- Interaction history can reveal personal preferences.
- Collect only necessary data for a clear purpose.
- Protect identifiers and restrict access.
- Apply retention and deletion policies.
- Avoid unjustified sensitive inferences.

### Q16. Can recommendation systems create feedback loops?
**Answer:**
- Yes. Items shown more often may receive more clicks.
- The system may interpret clicks as stronger preference.
- Less-exposed items may not get a fair chance.
- Monitor exposure, diversity and outcomes; consider controlled exploration.

## 5. Rapid revision
- Popularity: generally popular items.
- Content-based: item attributes.
- Collaborative: interaction patterns.
- Hybrid: combination of methods.
- Cold start: insufficient interaction history.
- Precision@K and Recall@K: ranking metrics.
- Time-based evaluation: protects against future interaction leakage.
