# Day 59: Recommendation System Fundamentals

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Beginner to Intermediate  
**Time:** 2–3 hours  
**Approved mini-project:** Product Recommendation Prototype

## 1. Learning objectives
- Explain recommendation systems.
- Compare popularity, content-based and collaborative filtering.
- Build a popularity recommender.
- Calculate item similarity using TF-IDF and cosine similarity.
- Explain cold-start and ranking evaluation.

## 2. What is a recommendation system?
A recommendation system ranks items that may be relevant to a user or context. Examples include products, films, music, courses and news.

Typical flow:
User/context + item data → candidate generation → ranking → recommendations → feedback.

A recommendation is not a guarantee that the user will like or buy an item.

## 3. Main approaches

| Approach | How it works | Example | Limitation |
|---|---|---|---|
| Popularity | Recommends generally popular items | Top-selling products | Similar list for everyone |
| Content-based | Matches item attributes to preferences | Similar product category | Needs useful item features |
| Collaborative | Uses user-item interaction patterns | Similar users/items | Sparse data and cold start |
| Hybrid | Combines methods | Content + collaborative | More complex |

## 4. User-item matrix
| User | Phone | Headphones | Laptop |
|---|---:|---:|---:|
| U1 | 1 | 1 | 0 |
| U2 | 1 | 0 | 1 |
| U3 | 0 | 1 | 1 |

A value of 1 might represent a purchase in this example. Real systems may use clicks, ratings, views or carts. Missing interactions do not automatically mean dislike.

## 5. Create interaction data
```python
import pandas as pd

interactions = pd.DataFrame({
    "user_id": ["U1", "U1", "U2", "U2", "U3", "U3", "U4", "U4"],
    "item": ["Phone", "Audio", "Phone", "Laptop", "Audio", "Laptop", "Phone", "Audio"],
    "rating": [5, 4, 4, 5, 5, 4, 3, 4]
})
print(interactions)
```

This small table is only for learning.

## 6. Popularity recommender
```python
popular_items = (
    interactions.groupby("item")
    .agg(interactions=("rating", "count"), average_rating=("rating", "mean"))
    .sort_values(["interactions", "average_rating"], ascending=False)
)
print(popular_items)
```

This ranks by interaction count, then average rating. Production ranking may also consider recency, minimum support and exposure bias.

## 7. Content-based recommendation

### 7.1 Create item descriptions and vectors
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

items = pd.DataFrame({
    "item": ["Phone", "Tablet", "Headphones", "Laptop"],
    "description": [
        "mobile device communication",
        "portable touchscreen device",
        "wireless audio music",
        "computer productivity device"
    ]
})
vectorizer = TfidfVectorizer()
item_vectors = vectorizer.fit_transform(items["description"])
similarity = cosine_similarity(item_vectors)
print(similarity.round(2))
```

TF-IDF converts text into numeric vectors. Cosine similarity compares vector directions. Similarity reflects descriptions, not user satisfaction.

### 7.2 Recommend similar items
```python
selected_index = 0
scores = list(enumerate(similarity[selected_index]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)

for index, score in scores[1:3]:
    print(items.loc[index, "item"], round(score, 3))
```

**Expected output:** Two highest-scoring items other than the selected item. Exact order depends on descriptions.

## 8. Collaborative filtering
Collaborative filtering uses interaction patterns:
- User-based: identify users with similar histories.
- Item-based: identify items interacted with by similar users.
- Matrix factorization: learn latent user and item representations.

```python
user_item = interactions.pivot_table(
    index="user_id", columns="item", values="rating"
)
print(user_item)
```

Missing cells mean no recorded rating, not necessarily a negative rating.

## 9. Cold-start problem
- New user: no interaction history.
- New item: no interaction history.
- New platform: little interaction data.

Possible solutions include popularity fallback, onboarding preferences, content-based similarity and carefully chosen context.

## 10. Evaluation
- Precision@K: fraction of top K recommendations that are relevant.
- Recall@K: fraction of all relevant items retrieved in top K.
- MAP/NDCG: ranking quality.
- Coverage: how much of the catalog can be recommended.

Use time-aware evaluation: train on earlier interactions and evaluate on later ones. Online experiments need careful design and guardrails.

## 11. Approved mini-project — Product Recommendation Prototype

**Scenario:** An online store wants to suggest products to visitors.

**Tasks**
1. Create/load a product catalog with descriptions and categories.
2. Create user-item interaction data.
3. Build popularity ranking.
4. Convert item descriptions to TF-IDF vectors.
5. Calculate cosine similarity.
6. Return top-K similar items, excluding the selected item.
7. Explain cold-start behavior.
8. Define a time-aware evaluation approach.

**Suggested structure**
```text
product_recommendation/
├── data/products.csv
├── data/interactions.csv
├── src/popularity.py
├── src/content_recommender.py
├── notebooks/
└── README.md
```

**Validation:** Exclude self-recommendations, handle empty descriptions, avoid future interaction leakage and explain toy-data limitations.

## 12. Responsible recommendation design
- Protect interaction history.
- Minimize collection and retention.
- Avoid unjustified sensitive inferences.
- Consider popularity feedback loops and catalog diversity.
- Provide appropriate user controls.
- Monitor quality across users and item categories.

## 13. Completion checklist
- [ ] Explain popularity, content-based and collaborative filtering.
- [ ] Build popularity ranking.
- [ ] Generate content similarity.
- [ ] Explain cold start and ranking metrics.
- [ ] Complete Product Recommendation Prototype.

**Next lesson:** Day 60 — Model pipelines and preprocessing integration.
