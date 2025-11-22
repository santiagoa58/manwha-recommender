# System Architecture

## Overview

The Manwha Recommender System is a hybrid recommendation engine that combines multiple machine learning approaches to provide personalized manhwa (Korean comics) recommendations. The system consists of three main layers:

1. **Data Collection & Processing Layer** - Collects and deduplicates data from multiple sources
2. **Machine Learning Layer** - Builds and trains recommendation models
3. **Application Layer** - Provides CLI interface for user interaction

## Architecture Diagram (Text-Based)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Sources                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ AniList  │  │   MAL    │  │  MangaU  │  │  AnimePl │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
└───────┼─────────────┼─────────────┼─────────────┼─────────────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Data Collectors           │
        │  - AniListCollector         │
        │  - JikanCollector (MAL)     │
        │  - MangaUpdatesCollector    │
        │  - AnimePlanetCollector     │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │   Deduplicator              │
        │  - Fuzzy matching           │
        │  - Blocking strategy        │
        │  - Multi-source merging     │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │   Unified Catalog           │
        │  (cleanedManwhas.json)      │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │   HybridRecommender         │
        │  ┌────────────────────────┐ │
        │  │ Content-Based (TF-IDF) │ │
        │  ├────────────────────────┤ │
        │  │ Genre Similarity (SVD) │ │
        │  ├────────────────────────┤ │
        │  │ User Preferences       │ │
        │  └────────────────────────┘ │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │   Serialized Models         │
        │  (models/ directory)        │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │   CLI Interface             │
        │  - Interactive prompts      │
        │  - Recommendations display  │
        └─────────────────────────────┘
```

## Component Details

### 1. Data Collection Layer

#### Data Collectors

Each collector fetches data from a specific API/source and normalizes it to a common schema:

**AniListCollector** (`src/data_collection/anilist_collector.py`)
- Source: AniList GraphQL API
- Features: Comprehensive metadata, high-quality descriptions, genre tags
- Rate limiting: 90 requests/minute
- Async: Yes (batched GraphQL queries)

**JikanCollector** (`src/data_collection/jikan_collector.py`)
- Source: Jikan REST API (MyAnimeList unofficial)
- Features: User ratings, popularity metrics, MAL IDs
- Rate limiting: 3 requests/second
- Async: Yes (sequential with delays)

**MangaUpdatesCollector** (`src/data_collection/mangaupdates_collector.py`)
- Source: MangaUpdates API
- Features: Publication status, author info, detailed tags
- Rate limiting: No strict limit (respectful delays)
- Async: Yes

**AnimePlanetCollector** (`src/data_collection/animeplanet_collector.py`)
- Source: Anime-Planet web scraping
- Features: Alternative recommendations, user reviews
- Rate limiting: Respectful delays between requests
- Async: Yes

**Common Schema:**
```json
{
  "id": "source_uniqueid",
  "name": "Title",
  "altName": "Alternative Title",
  "description": "Synopsis",
  "genres": ["Genre1", "Genre2"],
  "tags": ["Tag1", "Tag2"],
  "rating": 4.5,
  "popularity": 1000,
  "status": "RELEASING",
  "source": "SourceName"
}
```

#### Deduplicator

**ManwhaDeduplicator** (`src/data_processing/deduplicator.py`)

Merges entries from multiple sources using:

1. **Blocking Strategy**: Groups titles by normalized prefixes to reduce O(n²) comparisons
   - Creates blocks using first 2 characters and first significant word
   - Reduces comparison complexity from O(n²) to O(n*k) where k is avg block size

2. **Fuzzy Matching**: Uses rapidfuzz library with token_sort_ratio scorer
   - Title similarity threshold: 85%
   - Alternative title threshold: 80%
   - Handles variations: "Solo Leveling" ≈ "Solo Leveling (Manhwa)"

3. **Multi-Source Merging**:
   - Source priority: MangaUpdates > AniList > MyAnimeList > Anime-Planet
   - Weighted rating average based on vote counts
   - Union of genres and tags
   - Longest description selected
   - Preserves all source IDs for cross-referencing

**Memory Implications:**
- Entries duplicated across 2-3 blocks on average
- Memory usage: ~2-3x input data size
- Acceptable for <100k entries (typical: ~5k manhwa)

### 2. Machine Learning Layer

#### HybridManwhaRecommender

**Core Components:**

1. **Content-Based Filtering** (40% weight default)
   - TF-IDF vectorization on descriptions, genres, and tags
   - Configurable parameters: max_features (5000), ngram_range (1-2), min_df (2)
   - K-Nearest Neighbors (KNN) with cosine similarity
   - Feature matrix: sparse CSR matrix combining TF-IDF + normalized ratings + popularity

2. **Genre-Based Similarity** (30% weight default)
   - Binary genre matrix (one-hot encoding)
   - Truncated SVD for dimensionality reduction
   - Auto-selects components to explain 90% variance (typically 20-30 components)
   - Cosine similarity in latent genre space

3. **User Preference Scoring** (30% weight default)
   - Genre matching: +0.3 for liked genres, -0.3 for disliked
   - Rating filter: +0.2 if meets threshold, -0.3 otherwise
   - Status preference: +0.1 for preferred status
   - Base score: 0.5 (neutral)

**Recommendation Pipeline:**

```
Input Title
    ↓
Find Index (with fuzzy matching cache)
    ↓
Generate Candidates (5x requested amount)
    ├→ Content-Based Recommendations (KNN)
    ├→ Genre Similarity Recommendations (SVD)
    └→ User Preference Scores
    ↓
Combine Weighted Scores
    ↓
Apply Filters (genre, rating, status, etc.)
    ↓
Optional: MMR Re-ranking (diversity)
    ↓
Top-N Results
```

**Evaluation Metrics:**

- **Precision@K**: Accuracy of top K recommendations
- **Recall@K**: Coverage of relevant items in top K
- **NDCG@K**: Ranking quality with position discounting
- **MRR**: Mean Reciprocal Rank of first relevant item
- **Hit Rate@K**: % of queries with at least 1 relevant result
- **Coverage**: % of catalog items ever recommended
- **Novelty**: Average popularity rank (higher = more novel)
- **Diversity**: Intra-list diversity (1 - avg pairwise similarity)

### 3. Application Layer

#### CLI Interface (`src/cli/main.py`)

Interactive command-line interface with:
- Title search with autocomplete
- Personalized filtering options
- Formatted recommendation display
- Error handling and fuzzy matching

#### Model Persistence

**Saved Components:**
- `content_model.pkl`: Trained KNN model
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer
- `genre_model.pkl`: Trained SVD model
- `feature_matrix.pkl`: Pre-computed feature matrix
- `rating_scaler.pkl`: Min-max scaler for ratings
- `popularity_scaler.pkl`: Min-max scaler for popularity
- `manhwa_catalog.pkl`: Processed catalog dataframe
- `recommender_config.json`: Hyperparameters and metadata

**Model Versioning:**
- Current version: 2.0.0
- Backward compatible with 2.x.x models
- Breaking changes increment major version

## Data Flow

### Build Pipeline (Offline)

```
1. Data Collection
   ├─ Run collectors (async, parallel)
   ├─ Save raw JSON files (data/anilist.json, jikan.json, etc.)
   └─ Typically takes 5-15 minutes for ~5k titles

2. Deduplication
   ├─ Load all source files
   ├─ Normalize and block titles
   ├─ Fuzzy match within blocks
   ├─ Merge duplicate groups
   └─ Save unified catalog (data/cleanedManwhas.json)

3. Model Training
   ├─ Load unified catalog
   ├─ Build TF-IDF features
   ├─ Train KNN model
   ├─ Build genre SVD features
   ├─ Optional: Evaluation on test set
   └─ Save serialized models (models/)

Total build time: 10-20 minutes
```

### Runtime Pipeline (Online)

```
1. Load Model
   ├─ Deserialize models from disk
   ├─ Validate version compatibility
   └─ ~2-5 seconds startup time

2. User Request
   ├─ Input: manhwa title
   ├─ Optional: user preferences, filters
   └─ Fuzzy match to catalog entry

3. Generate Recommendations
   ├─ Query KNN for content similarity
   ├─ Query SVD for genre similarity
   ├─ Score with user preferences
   ├─ Combine weighted scores
   ├─ Apply filters
   └─ Optional: MMR diversity re-ranking

4. Return Results
   └─ Recommendation latency: ~100-500ms for 10 items

```

## Configuration and Customization

### Hyperparameter Tuning

**Tunable Parameters:**

```python
# Component weights (must sum to 1.0)
weights = {
    'content': 0.4,           # TF-IDF similarity
    'genre_similarity': 0.3,  # Genre latent features
    'user_pref': 0.3          # User preference scoring
}

# TF-IDF parameters
tfidf_params = {
    'max_features': 5000,     # Vocabulary size
    'ngram_range': (1, 2),    # Unigrams + bigrams
    'min_df': 2,              # Minimum document frequency
    'max_df': 0.8             # Maximum document frequency (filter common terms)
}

# SVD parameters
svd_params = {
    'explained_variance_threshold': 0.90  # Auto-select components
}

# KNN parameters
knn_params = {
    'metric': 'cosine',       # Distance metric
    'algorithm': 'brute'      # Exact search (small dataset)
}

# User preference weights
user_pref_weights = {
    'base_score': 0.5,        # Neutral starting point
    'genre_match': 0.3,       # Liked genre boost
    'genre_penalty': 0.3,     # Disliked genre penalty
    'rating_boost': 0.2,      # Meets rating threshold
    'rating_penalty': 0.3,    # Below rating threshold
    'status_boost': 0.1       # Preferred status
}
```

**Grid Search:**

Use `HybridManwhaRecommender.tune_hyperparameters()` to automatically search parameter space and optimize for a target metric (NDCG, Precision, etc.).

### Cold Start Handling

**Strategy:**
1. Get top-rated items (nlargest by rating)
2. Apply popularity bias (configurable 0-1)
3. Ensure genre diversity (select items with new genres)
4. Fill remaining with top-rated items

**Parameters:**
- `popularity_bias`: 0.0 = quality only, 1.0 = popularity only
- Recommended: 0.3-0.5 for balanced exploration/exploitation

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Data Collection | O(n) API calls | 5-15 min for 5k items |
| Deduplication | O(n*k) fuzzy matching | 1-2 min for 20k entries |
| Model Training | O(n*d) TF-IDF + O(n²) KNN | 30-60 sec for 5k items |
| Single Recommendation | O(log n) KNN query | 100-500 ms |
| Batch Recommendations | O(k log n) for k queries | Linear scaling |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Raw Data | ~5-10 MB | Per source (4 sources) |
| Unified Catalog | ~10-20 MB | Deduplicated entries |
| TF-IDF Matrix | ~50-100 MB | Sparse matrix (5k×5k features) |
| Feature Matrix | ~60-120 MB | TF-IDF + metadata |
| Genre Features | ~1-5 MB | Dense matrix (5k×30 components) |
| KNN Model | ~1 MB | Lightweight index |
| Total Models | ~150-250 MB | All serialized components |

### Scalability

**Current Design:**
- Optimized for 1k-50k manhwa catalog
- In-memory operation (requires 1-2 GB RAM)
- Single-machine deployment

**Scaling Considerations:**
- For >100k items: Consider approximate KNN (Annoy, FAISS)
- For distributed: Shard by genre, use distributed TF-IDF
- For real-time updates: Incremental learning or periodic retraining

## Thread Safety Considerations

**WARNING: Not Thread-Safe**

The `HybridManwhaRecommender` class contains mutable state that is modified without synchronization:

- `df`: Pandas DataFrame (modified during training/evaluation)
- `user_preferences`: Dict of user profiles
- `_all_recommended_items`: Set tracking coverage
- `_title_cache`: Dict for title lookups
- `_popularity_ranks`: Dict of popularity rankings

**Implications:**
- Do NOT share single instance across threads
- Do NOT call training methods concurrently
- Safe: Multiple threads calling `recommend()` after training (read-only)
- Unsafe: Concurrent calls to `prepare_data()`, `build_content_features()`, etc.

**Solutions:**
1. **Separate Instances**: Create one recommender instance per thread
2. **External Locking**: Use threading.Lock() around all method calls
3. **Process-Based**: Use multiprocessing instead of threading
4. **Immutable Models**: Load model once, only call read-only methods

## Deployment Patterns

### Local CLI Application

```bash
# Build models
python -m scripts.build

# Run interactive CLI
python -m src.cli.main
```

### Web Service (Future)

Recommended architecture:
- Load model at service startup
- Handle requests in separate threads (with proper locking)
- Cache recommendations per user session
- Periodic model updates (offline training)

### Batch Processing

```python
recommender = HybridManwhaRecommender()
recommender.load_model("models/")

# Process batch of user queries
for title in user_titles:
    recs = recommender.recommend(title, n_recommendations=10)
    # Store or return results
```

## Error Handling and Logging

**Logging Levels:**
- INFO: Normal operations, progress updates
- WARNING: Data quality issues, fuzzy matches, fallback behaviors
- ERROR: Failed API calls, model loading errors

**Common Errors:**

1. **FileNotFoundError**: Missing catalog or model files
   - Solution: Run `scripts/build.py` to generate models

2. **ValueError**: Invalid catalog format or corrupted data
   - Solution: Re-run data collection and deduplication

3. **KeyError**: Missing required fields in catalog entries
   - Solution: Update collector schemas or add default values

4. **MemoryError**: Dataset too large for available RAM
   - Solution: Reduce max_features, use smaller dataset, or add more RAM

## Extension Points

### Adding New Data Sources

1. Create collector class inheriting from base pattern
2. Implement async fetch and normalization
3. Add to deduplication pipeline
4. Update source priority in deduplicator

### Custom Recommendation Algorithms

1. Create new recommendation method in HybridManwhaRecommender
2. Add weight to self.weights dict
3. Include scores in recommend() method
4. Validate weights sum to 1.0

### Alternative Ranking Strategies

1. Modify _mmr_rerank() for different diversity metrics
2. Implement learning-to-rank with user feedback
3. Add temporal features (trending, recent releases)

## References

- TF-IDF: Salton & Buckley (1988)
- K-Nearest Neighbors: Cover & Hart (1967)
- SVD: Golub & Reinsch (1970)
- MMR: Carbonell & Goldstein (1998)
- NDCG: Järvelin & Kekäläinen (2002)
