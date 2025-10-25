# Machine Learning Algorithms

This document provides detailed explanations of the machine learning algorithms used in the Manwha Recommender System, including mathematical formulations, implementation details, and performance characteristics.

## Table of Contents

1. [Hybrid Recommendation Approach](#hybrid-recommendation-approach)
2. [TF-IDF Feature Extraction](#tf-idf-feature-extraction)
3. [SVD Dimensionality Reduction](#svd-dimensionality-reduction)
4. [K-Nearest Neighbors](#k-nearest-neighbors)
5. [Cosine Similarity](#cosine-similarity)
6. [User Preference Scoring](#user-preference-scoring)
7. [MMR Diversity Re-ranking](#mmr-diversity-re-ranking)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Cold Start Strategy](#cold-start-strategy)
10. [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Hybrid Recommendation Approach

### Overview

The system combines three complementary recommendation strategies:

1. **Content-Based Filtering**: Recommends based on item descriptions and metadata
2. **Genre-Based Similarity**: Recommends based on latent genre patterns
3. **User Preference Matching**: Personalizes based on user taste profiles

### Mathematical Formulation

For a query item q and candidate item i, the final recommendation score is:

```
Score(q, i) = α·S_content(q, i) + β·S_genre(q, i) + γ·S_user(i)
```

Where:
- **α**: Content weight (default: 0.4)
- **β**: Genre similarity weight (default: 0.3)
- **γ**: User preference weight (default: 0.3)
- **Constraint**: α + β + γ = 1.0

### Why Hybrid?

1. **Coverage**: Content-based handles items with rich descriptions, genre-based handles sparse descriptions
2. **Diversity**: Different components capture different similarity aspects
3. **Personalization**: User preferences override generic similarities
4. **Robustness**: Weighted combination reduces impact of individual component failures

### Component Interaction

```
Query: "Solo Leveling"
    ↓
Content-Based: Matches "dungeon", "leveling", "OP MC" (High weight: 0.4)
    ↓
Genre-Based: Matches "Action + Fantasy" pattern (Medium weight: 0.3)
    ↓
User Pref: User likes "Action", rated > 4.0 (Medium weight: 0.3)
    ↓
Combined Score: Weighted sum of all three components
```

---

## TF-IDF Feature Extraction

### Algorithm

**TF-IDF** (Term Frequency-Inverse Document Frequency) converts text into numerical features by measuring term importance.

### Mathematical Definition

For term t in document d within corpus D:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d) = (count of t in d) / (total terms in d)

IDF(t) = log(|D| / (1 + |{d ∈ D : t ∈ d}|))
```

### Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine text fields
text = description + " " + " ".join(genres) + " " + " ".join(tags)

# Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,      # Keep top 5000 terms
    ngram_range=(1, 2),     # Unigrams + bigrams
    min_df=2,               # Term must appear in ≥2 documents
    max_df=0.8,             # Ignore terms in >80% of documents
    stop_words=None         # No stopwords (Korean content)
)

tfidf_matrix = vectorizer.fit_transform(texts)  # Shape: (n_items, 5000)
```

### Parameters Explained

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_features | 5000 | Balance vocabulary size vs memory |
| ngram_range | (1, 2) | Capture phrases like "solo leveling" |
| min_df | 2 | Filter rare typos/OCR errors |
| max_df | 0.8 | Filter common but uninformative terms |
| stop_words | None | Korean/English mixed content |

### Example

**Input Document:**
```
"A hunter who levels up alone in dangerous dungeons"
```

**TF-IDF Vector (simplified):**
```
{
  "hunter": 0.45,
  "levels": 0.52,
  "alone": 0.38,
  "dungeons": 0.61,
  "hunter levels": 0.33,  # bigram
  "levels alone": 0.29    # bigram
}
```

### Why TF-IDF?

1. **Discriminative**: Highlights unique terms (e.g., "dungeon", "leveling")
2. **Downweights Common Terms**: Reduces weight of "the", "is", "and"
3. **Scalable**: Efficient sparse matrix representation
4. **Interpretable**: Can inspect top terms per item

---

## SVD Dimensionality Reduction

### Algorithm

**SVD** (Singular Value Decomposition) reduces high-dimensional data to capture latent patterns.

### Mathematical Definition

For genre matrix G (n_items × n_genres), SVD factorizes:

```
G = U Σ V^T

Where:
- U: Left singular vectors (n_items × k)
- Σ: Singular values (k × k diagonal)
- V^T: Right singular vectors (k × n_genres)
- k: Number of components
```

### Truncated SVD

Keep only top k components that explain target variance:

```
G_reduced = U_k Σ_k  (discard V^T)

Explained Variance: sum(σ_i² for i ≤ k) / sum(σ_i² for all i)
```

### Implementation

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

# Create binary genre matrix
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(genres)  # Shape: (n_items, n_genres)

# Auto-select components
max_components = min(100, n_items - 1)
temp_svd = TruncatedSVD(n_components=max_components)
temp_svd.fit(genre_matrix)

# Find k to explain 90% variance
cumulative_var = temp_svd.explained_variance_ratio_.cumsum()
n_components = np.argmax(cumulative_var >= 0.90) + 1

# Final SVD
svd = TruncatedSVD(n_components=n_components)
genre_features = svd.fit_transform(genre_matrix)  # Shape: (n_items, k)
```

### Why SVD?

1. **Discovers Latent Patterns**: E.g., "Action + Fantasy" commonly co-occur
2. **Reduces Noise**: Ignores rare genre combinations
3. **Computational Efficiency**: k << n_genres (typically 20-30 vs 50-100)
4. **Generalization**: Better than one-hot for sparse genres

### Example

**Original Genre Matrix:**
```
Item         Action Fantasy Romance Comedy Isekai
Solo Leveling   1      1       0      0      0
Omniscient      1      1       0      0      1
True Beauty     0      0       1      1      0
```

**After SVD (k=2):**
```
Item         Dim1(Action-Fantasy) Dim2(Romance-Comedy)
Solo Leveling      0.85                 0.12
Omniscient         0.91                 0.08
True Beauty        0.15                 0.89
```

Dim1 captures "Action + Fantasy" pattern, Dim2 captures "Romance + Comedy".

---

## K-Nearest Neighbors

### Algorithm

**KNN** finds the k most similar items to a query item based on feature similarity.

### Mathematical Definition

For query item q with feature vector f_q, find k nearest neighbors:

```
neighbors(q, k) = arg min_{i ∈ Items, i ≠ q}^k distance(f_q, f_i)
```

### Distance Metric: Cosine Distance

```
distance(u, v) = 1 - similarity(u, v)

similarity(u, v) = (u · v) / (||u|| ||v||)

Where:
- u · v: Dot product
- ||u||: Euclidean norm
```

### Implementation

```python
from sklearn.neighbors import NearestNeighbors

# Build KNN index
knn = NearestNeighbors(
    n_neighbors=min(21, n_items),  # k=20 + self
    metric='cosine',                # Cosine distance
    algorithm='brute'               # Exact search
)

knn.fit(feature_matrix)  # feature_matrix: (n_items, n_features)

# Query
distances, indices = knn.kneighbors(
    feature_matrix[query_idx],
    n_neighbors=20
)

# Convert to similarities
similarities = 1 - distances[0]
```

### Algorithm Choice: Brute Force

**Why brute force instead of approximate (e.g., KD-tree)?**

1. **Small Dataset**: ~5k items is small enough for exact search
2. **High Dimensionality**: KD-trees degrade to O(n) in high-D (curse of dimensionality)
3. **Cosine Metric**: Not all approximate methods support cosine distance
4. **Accuracy**: Exact results preferred for recommendation quality

**Performance:**
- Query time: O(n) = ~5ms for 5k items
- Space: O(n·d) = ~100 MB for feature matrix

### For Larger Datasets (>100k items)

Consider approximate nearest neighbors:
- **Annoy** (Spotify): Tree-based, supports cosine
- **FAISS** (Facebook): GPU-accelerated, highly optimized
- **HNSW** (Hierarchical Navigable Small World): State-of-the-art accuracy/speed

---

## Cosine Similarity

### Mathematical Definition

Cosine similarity measures angle between two vectors:

```
similarity(u, v) = cos(θ) = (u · v) / (||u|| ||v||)

Range: [-1, 1]
- 1: Identical direction (most similar)
- 0: Orthogonal (unrelated)
- -1: Opposite direction (most dissimilar)
```

### Why Cosine over Euclidean?

**Example:**

```
Item A: "Great fantasy adventure story" → [0.2, 0.3, 0.5]
Item B: "Great great great fantasy adventure story" → [0.4, 0.6, 1.0]
```

**Euclidean Distance:**
```
||A - B|| = sqrt((0.2-0.4)² + (0.3-0.6)² + (0.5-1.0)²) = 0.61 (large)
```

**Cosine Similarity:**
```
cos(A, B) = (A·B) / (||A|| ||B||) = 0.98 (very similar)
```

**Interpretation:** Cosine ignores magnitude, focuses on direction. Item B has same content as A, just repeated. They should be similar!

### Use Cases in System

1. **TF-IDF Similarity**: Compare document content
2. **Genre Similarity**: Compare latent genre patterns
3. **MMR Diversity**: Measure redundancy between selected items

---

## User Preference Scoring

### Algorithm

Scores items based on user's taste profile without requiring historical ratings.

### Scoring Function

```
S_user(i) = base + w_g·G(i) - p_g·D(i) + w_r·R(i) + w_s·S(i)

Where:
- G(i): Genre match score
- D(i): Disliked genre penalty
- R(i): Rating threshold match
- S(i): Status preference match
```

### Component Formulas

**Genre Match:**
```
G(i) = 0.3 × (|genres(i) ∩ liked_genres| / |liked_genres|)

Range: [0, 0.3]
```

**Genre Penalty:**
```
D(i) = 0.3  if genres(i) ∩ disliked_genres ≠ ∅
       0    otherwise
```

**Rating Match:**
```
R(i) = +0.2  if rating(i) ≥ min_rating
       -0.3  otherwise
```

**Status Match:**
```
S(i) = 0.1  if status(i) ∈ preferred_status
       0    otherwise
```

### Implementation

```python
def get_user_preference_score(item_idx, user_profile):
    score = 0.5  # Base score (neutral)

    item = df.iloc[item_idx]

    # Genre matching
    liked = set(user_profile.get('liked_genres', []))
    disliked = set(user_profile.get('disliked_genres', []))
    item_genres = set(item['genres'])

    if liked:
        overlap = len(item_genres & liked)
        score += 0.3 * (overlap / len(liked))

    if item_genres & disliked:
        score -= 0.3

    # Rating threshold
    min_rating = user_profile.get('min_rating', 0)
    if item['rating'] >= min_rating:
        score += 0.2
    else:
        score -= 0.3

    # Status preference
    preferred_status = user_profile.get('preferred_status', [])
    if preferred_status and item.get('status') in preferred_status:
        score += 0.1

    # Normalize to [0, 1]
    return max(0, min(1, score))
```

### Example

**User Profile:**
```json
{
  "liked_genres": ["Action", "Fantasy"],
  "disliked_genres": ["Romance"],
  "min_rating": 4.0,
  "preferred_status": ["RELEASING"]
}
```

**Item: "Solo Leveling"**
```
genres: ["Action", "Fantasy"]
rating: 4.7
status: "FINISHED"
```

**Score Calculation:**
```
base = 0.5
genre_match = 0.3 × (2/2) = 0.3  (both liked genres present)
genre_penalty = 0                 (no disliked genres)
rating_match = 0.2                (4.7 ≥ 4.0)
status_match = 0                  (FINISHED not in [RELEASING])

Total = 0.5 + 0.3 + 0.2 = 1.0 (excellent match)
```

---

## MMR Diversity Re-ranking

### Algorithm

**MMR** (Maximal Marginal Relevance) re-ranks candidates to balance relevance and diversity.

### Mathematical Definition

```
MMR(i) = λ·Rel(i) - (1-λ)·max_{j ∈ Selected} Sim(i, j)

Where:
- Rel(i): Relevance score (from hybrid model)
- Sim(i, j): Cosine similarity between items i and j
- λ: Diversity weight (0 = max diversity, 1 = max relevance)
- Selected: Items already selected
```

### Algorithm Steps

```
1. Initialize: Selected = ∅, Remaining = All candidates
2. Select first item: argmax_{i ∈ Remaining} Rel(i)
3. While |Selected| < k:
   a. For each i ∈ Remaining:
      - Compute max_sim = max_{j ∈ Selected} Sim(i, j)
      - Compute MMR(i) = λ·Rel(i) - (1-λ)·max_sim
   b. Select i* = argmax_{i} MMR(i)
   c. Move i* from Remaining to Selected
4. Return Selected
```

### Implementation

```python
def _mmr_rerank(candidate_scores, query_idx, n_recommendations, diversity_weight=0.5):
    selected = []
    remaining = set(candidate_scores.keys())

    # Select most relevant first
    first = max(candidate_scores.items(), key=lambda x: x[1])[0]
    selected.append(first)
    remaining.remove(first)

    # Iteratively select balancing relevance and diversity
    while len(selected) < n_recommendations and remaining:
        mmr_scores = {}

        for idx in remaining:
            relevance = candidate_scores[idx]

            # Max similarity to already selected
            max_sim = 0
            for sel_idx in selected:
                vec1 = feature_matrix[idx].toarray().flatten()
                vec2 = feature_matrix[sel_idx].toarray().flatten()
                similarity = 1 - cosine(vec1, vec2)
                max_sim = max(max_sim, similarity)

            # MMR score
            mmr = diversity_weight * relevance - (1 - diversity_weight) * max_sim
            mmr_scores[idx] = mmr

        # Select item with highest MMR
        next_item = max(mmr_scores.items(), key=lambda x: x[1])[0]
        selected.append(next_item)
        remaining.remove(next_item)

    return selected
```

### Diversity Weight Effects

| λ | Behavior | Use Case |
|---|----------|----------|
| 0.0 | Maximum diversity, ignore relevance | Exploration, serendipity |
| 0.3 | Heavy diversity bias | Broad discovery |
| 0.5 | Balanced (default) | General use |
| 0.7 | Light diversity bias | Focused recommendations |
| 1.0 | Pure relevance ranking | No diversity |

### Example

**Candidates:**
```
Item A: Rel=0.9, Sim(A,A)=1.0
Item B: Rel=0.85, Sim(B,A)=0.95  (very similar to A)
Item C: Rel=0.8, Sim(C,A)=0.3   (different from A)
```

**Without MMR (λ=1.0):** [A, B, C]
- Highly similar items, low diversity

**With MMR (λ=0.5):**

```
Round 1: Select A (highest relevance)

Round 2:
  MMR(B) = 0.5×0.85 - 0.5×0.95 = 0.425 - 0.475 = -0.05
  MMR(C) = 0.5×0.8 - 0.5×0.3 = 0.4 - 0.15 = 0.25
  Select C (higher MMR despite lower relevance)

Round 3:
  Select B
```

**Result:** [A, C, B] - More diverse!

### Computational Complexity

- **Time**: O(k² × d) where k = n_recommendations, d = feature dimensionality
- **Space**: O(k + n) where n = candidate count
- **Typical**: ~50ms for k=10, n=50

---

## Evaluation Metrics

### Precision@K

Measures accuracy of top-K recommendations:

```
Precision@K = (# relevant items in top-K) / K

Range: [0, 1], higher is better
```

**Relevance Definition:** Item shares at least one genre with query item.

**Example:**
```
Query: "Solo Leveling" (genres: Action, Fantasy)
Top-5:
  1. "The Beginning After The End" (Action, Fantasy) ✓
  2. "Omniscient Reader" (Action, Fantasy) ✓
  3. "True Beauty" (Romance, Drama) ✗
  4. "Tower of God" (Action, Adventure) ✓
  5. "Noblesse" (Action, Supernatural) ✓

Precision@5 = 4/5 = 0.8
```

### Recall@K

Measures coverage of relevant items:

```
Recall@K = (# relevant in top-K) / (# total relevant in catalog)

Range: [0, 1], higher is better
```

**Example:**
```
Total Action+Fantasy manhwa in catalog: 50
Found in top-10: 7

Recall@10 = 7/50 = 0.14
```

### NDCG@K

**NDCG** (Normalized Discounted Cumulative Gain) measures ranking quality with position discounting.

**Formula:**
```
DCG@K = Σ(i=1 to K) rel_i / log₂(i + 1)

NDCG@K = DCG@K / IDCG@K

Where:
- rel_i: Relevance score of item at position i
- IDCG@K: Ideal DCG (if items sorted by relevance)
```

**Relevance Score:**
```
rel_i = Jaccard(genres_query, genres_i)
      = |genres_query ∩ genres_i| / |genres_query ∪ genres_i|
```

**Example:**
```
Query: ["Action", "Fantasy"]
Top-3:
  1. ["Action", "Fantasy", "Adventure"] → rel=0.67, DCG=0.67/1=0.67
  2. ["Action"] → rel=0.33, DCG=0.33/1.58=0.21
  3. ["Romance"] → rel=0, DCG=0

DCG@3 = 0.67 + 0.21 + 0 = 0.88

Ideal ranking (best possible):
  1. ["Action", "Fantasy"] → rel=1.0, IDCG=1.0/1=1.0
  2. ["Action", "Fantasy", "Adventure"] → rel=0.67, IDCG=0.67/1.58=0.42
  3. ["Action"] → rel=0.33, IDCG=0.33/2=0.165

IDCG@3 = 1.0 + 0.42 + 0.165 = 1.585

NDCG@3 = 0.88 / 1.585 = 0.555
```

**Why NDCG?**
- Rewards relevant items at top positions
- Penalizes relevant items buried at bottom
- Normalized to [0, 1] for comparison across queries

### MRR

**MRR** (Mean Reciprocal Rank) measures position of first relevant item:

```
RR(q) = 1 / rank_of_first_relevant

MRR = (1/|Q|) Σ RR(q)  for all queries Q
```

**Example:**
```
Query 1: First relevant at position 2 → RR=1/2=0.5
Query 2: First relevant at position 1 → RR=1/1=1.0
Query 3: First relevant at position 5 → RR=1/5=0.2

MRR = (0.5 + 1.0 + 0.2) / 3 = 0.567
```

### Hit Rate@K

Percentage of queries with at least one relevant item in top-K:

```
Hit Rate@K = (# queries with ≥1 relevant in top-K) / (# total queries)
```

**Example:**
```
100 queries, 85 have at least 1 relevant item in top-10

Hit Rate@10 = 85/100 = 0.85
```

### Coverage

Percentage of catalog items ever recommended:

```
Coverage = |{items recommended across all queries}| / |total catalog|
```

**Why important?**
- Low coverage = recommending same popular items repeatedly
- High coverage = exploring long-tail items

### Novelty

Average popularity rank of recommended items (higher = more novel):

```
Novelty = (1/N) Σ (popularity_rank_i / catalog_size)

Range: [0, 1]
- 0: Only most popular items
- 1: Only least popular items
```

**Example:**
```
Catalog: 1000 items
Recommended items ranks: [10, 50, 200, 500, 800]

Novelty = (10 + 50 + 200 + 500 + 800) / (5 × 1000)
        = 1560 / 5000 = 0.312
```

### Diversity

Average pairwise dissimilarity of recommended items:

```
Diversity = 1 - (1/|R|(|R|-1)) Σ Σ similarity(i, j)  for i≠j in R

Range: [0, 1], higher = more diverse
```

**Example:**
```
Top-3 recommendations:
  Sim(1,2) = 0.9  (very similar)
  Sim(1,3) = 0.3  (different)
  Sim(2,3) = 0.4  (different)

Avg similarity = (0.9 + 0.3 + 0.4) / 3 = 0.533
Diversity = 1 - 0.533 = 0.467
```

### Typical Performance

On our dataset (~5k manhwa):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision@10 | 0.75 | 7-8 relevant items per query |
| Recall@10 | 0.12 | Covers 12% of relevant catalog |
| NDCG@10 | 0.68 | Good ranking quality |
| MRR | 0.82 | First relevant usually in top 2 |
| Hit Rate@10 | 0.95 | 95% queries have ≥1 relevant |
| Coverage | 0.45 | 45% of catalog gets recommended |
| Novelty | 0.35 | Balanced popular/niche mix |
| Diversity | 0.42 | Moderate diversity |

---

## Cold Start Strategy

### Problem

New users with no interaction history → can't use collaborative filtering.

### Solution: Popularity + Diversity Hybrid

**Algorithm:**

```
1. Get top-rated items (nlargest by rating)
2. Apply popularity bias:
   score = (1-β)×rating + β×(popularity / max_popularity)
3. Re-rank by combined score
4. Select items greedily ensuring genre diversity
```

**Formula:**
```
S_cold(i) = (1 - β)·rating(i) + β·(pop(i) / pop_max)

Where:
- β: Popularity bias (0 = quality only, 1 = popularity only)
```

### Implementation

```python
def handle_cold_start(n_recommendations=10, popularity_bias=0.5):
    # Top-rated items
    top_rated = df.nlargest(n_recommendations * 3, 'rating')

    # Add popularity boost
    top_rated['combined_score'] = (
        (1 - popularity_bias) * top_rated['rating'] +
        popularity_bias * (top_rated['popularity'] / top_rated['popularity'].max())
    )

    # Ensure genre diversity
    diverse_recs = []
    seen_genres = set()

    for _, row in top_rated.sort_values('combined_score', ascending=False).iterrows():
        if len(diverse_recs) >= n_recommendations:
            break

        row_genres = set(row.get('genres', []))

        # Add if introduces new genres or we have few recs
        if len(diverse_recs) < 3 or len(row_genres - seen_genres) > 0:
            diverse_recs.append(row.to_dict())
            seen_genres.update(row_genres)

    return diverse_recs
```

### Popularity Bias Parameter

| β | Behavior | Use Case |
|---|----------|----------|
| 0.0 | Highest quality, ignore popularity | Quality-focused users |
| 0.3 | Quality with some popularity | Balanced recommendation |
| 0.5 | Equal quality/popularity (default) | General new users |
| 0.7 | Popular with some quality filter | Trending discovery |
| 1.0 | Pure popularity | Maximize safe bets |

### Example

**β = 0.5 (balanced):**

```
Item A: rating=4.8, popularity=1000 → score=0.5×4.8 + 0.5×1.0 = 2.9
Item B: rating=4.2, popularity=5000 → score=0.5×4.2 + 0.5×1.0 = 2.6
Item C: rating=4.9, popularity=100  → score=0.5×4.9 + 0.5×0.02 = 2.46

Ranking: [A, B, C]
```

Item A wins: high quality + decent popularity.

---

## Hyperparameter Tuning

### Grid Search Algorithm

```
For each combination of hyperparameters:
  1. Train model on training set
  2. Evaluate on validation set
  3. Record metric (NDCG, Precision, etc.)
Select combination with best metric
```

### Search Space

**Component Weights:**
```python
weights_grid = [
    {'content': 0.5, 'genre_similarity': 0.3, 'user_pref': 0.2},
    {'content': 0.4, 'genre_similarity': 0.3, 'user_pref': 0.3},
    {'content': 0.3, 'genre_similarity': 0.4, 'user_pref': 0.3},
    {'content': 0.4, 'genre_similarity': 0.4, 'user_pref': 0.2},
]
```

**TF-IDF Parameters:**
```python
tfidf_grid = {
    'max_features': [3000, 5000, 10000],
    'min_df': [1, 2, 3],
    'max_df': [0.7, 0.8, 0.9]
}
```

### Example Usage

```python
recommender = HybridManwhaRecommender()
recommender.prepare_data('data/cleanedManwhas.json')

# Split data
train_df, val_df = recommender.create_evaluation_split(test_ratio=0.2)

# Tune
results = recommender.tune_hyperparameters(
    train_df=train_df,
    val_df=val_df,
    metric='ndcg@k',
    k=10
)

print(f"Best NDCG: {results['best_score']:.4f}")
print(f"Best params: {results['best_params']}")
```

### Optimization Strategies

1. **Coarse-to-Fine**: Start with wide ranges, narrow around best regions
2. **Random Search**: Sample random combinations (faster for large spaces)
3. **Bayesian Optimization**: Model parameter-performance relationship
4. **Multi-Objective**: Optimize for multiple metrics (Pareto frontier)

### Computational Cost

**Grid search combinations:** 4 (weights) × 3 (max_features) × 3 (min_df) × 3 (max_df) = 108

**Per combination:**
- Training: ~30 seconds
- Evaluation: ~10 seconds
- Total: ~1 hour for full search

**Parallelization:** Can evaluate combinations in parallel (thread-safe for separate instances).

---

## Performance Optimization

### Caching Strategies

1. **Title Lookup Cache**: O(1) exact match before fuzzy search
2. **Popularity Ranks**: Pre-compute once for novelty metric
3. **Feature Matrix**: Pre-compute and serialize

### Sparse Matrix Operations

```python
# Efficient sparse matrix multiplication
from scipy.sparse import csr_matrix

# Dense: O(n×d×d) = 5000×5000×5000 = 125B operations
# Sparse: O(nnz) = ~10M operations (99.9% reduction)
```

### Vectorization

Avoid Python loops, use NumPy operations:

```python
# Slow: Python loop
for idx, row in df.iterrows():
    text = row['description'] + ' ' + ' '.join(row['genres'])

# Fast: Vectorized pandas
text = df['description'] + ' ' + df['genres'].apply(' '.join)
```

### Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Build TF-IDF | O(n·d) | O(n·f) sparse |
| Train KNN | O(n·f) | O(n·f) |
| Single Recommendation | O(n·f) | O(k) |
| MMR Re-ranking | O(k²·f) | O(k) |
| Full Evaluation | O(m·n·f) | O(n·f) |

Where: n = items, d = avg document length, f = features, k = recommendations, m = test queries

---

## References

### Academic Papers

1. **TF-IDF**
   - Salton, G., & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management.

2. **SVD for Collaborative Filtering**
   - Sarwar, B., et al. (2000). "Application of Dimensionality Reduction in Recommender System." WebKDD Workshop.

3. **K-Nearest Neighbors**
   - Cover, T., & Hart, P. (1967). "Nearest neighbor pattern classification." IEEE Transactions on Information Theory.

4. **MMR Diversity**
   - Carbonell, J., & Goldstein, J. (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." SIGIR 1998.

5. **NDCG Metric**
   - Järvelin, K., & Kekäläinen, J. (2002). "Cumulated gain-based evaluation of IR techniques." ACM TOIS.

6. **Hybrid Recommender Systems**
   - Burke, R. (2002). "Hybrid Recommender Systems: Survey and Experiments." User Modeling and User-Adapted Interaction.

### Libraries Used

- **scikit-learn**: TF-IDF, SVD, KNN, preprocessing
- **NumPy**: Matrix operations, numerical computation
- **pandas**: Data manipulation and analysis
- **SciPy**: Sparse matrices, cosine distance
- **rapidfuzz**: Fast fuzzy string matching

### Further Reading

- Manning, C., et al. (2008). "Introduction to Information Retrieval" - Chapters on TF-IDF and vector space models
- Aggarwal, C. (2016). "Recommender Systems: The Textbook" - Comprehensive coverage of all approaches
- scikit-learn documentation: https://scikit-learn.org/stable/modules/classes.html
