# Manwha Recommender System

<img width="1422" alt="Screenshot 2023-09-09 at 8 09 27 PM" src="https://github.com/santiagoa58/manwha-recommender/assets/24705975/789ffcca-59b6-4375-a7f4-8178cd3db98a">

## Demo

https://github.com/santiagoa58/manwha-recommender/assets/24705975/510501e4-4b7d-4cc6-adf4-71783b5681cc

## Overview

The Manwha Recommender System is an advanced **hybrid recommendation engine** that combines multiple machine learning approaches to provide personalized manhwa (Korean comics) recommendations. The system uses:

- **Content-Based Filtering**: TF-IDF vectorization on descriptions, genres, and tags
- **Genre-Based Similarity**: SVD dimensionality reduction to discover latent genre patterns
- **User Preference Learning**: Personalized scoring based on user taste profiles
- **Diversity Re-ranking**: MMR (Maximal Marginal Relevance) algorithm for diverse recommendations

## Key Features

- **Multi-Source Data Collection**: Aggregates data from AniList, MyAnimeList, MangaUpdates, and Anime-Planet
- **Intelligent Deduplication**: Fuzzy matching with blocking strategy to merge entries across sources
- **Configurable Hyperparameters**: Tune TF-IDF, SVD, and weighting parameters for optimal performance
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG, MRR, Coverage, Novelty, and Diversity metrics
- **Cold Start Handling**: Quality-popularity hybrid strategy for new users
- **Fast Recommendations**: ~100-500ms per query with efficient caching
- **Model Versioning**: Backward-compatible model serialization

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)**: System design, data flow, and component interactions
- **[Algorithm Details](docs/ALGORITHMS.md)**: Mathematical formulations and implementation details

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Building the Model](#building-the-model)
- [Using the Recommender](#using-the-recommender)
- [Advanced Usage](#advanced-usage)
- [Performance Characteristics](#performance-characteristics)
- [Thread Safety](#thread-safety)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/santiagoa58/manwha-recommender.git
   cd manwha-recommender
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```bash
# Build models (first-time setup, ~10-20 minutes)
python -m scripts.build

# Get recommendations
python -m src.cli.main
```

## Building the Model

The build process collects data, deduplicates entries, trains models, and evaluates performance.

```bash
python -m scripts.build
```

This will:
1. Collect data from multiple sources (AniList, MAL, MangaUpdates, Anime-Planet)
2. Deduplicate and merge entries using fuzzy matching
3. Train content-based (TF-IDF + KNN) and genre-based (SVD) models
4. Evaluate on held-out test set
5. Save serialized models to `models/` directory

**Build Time**: 10-20 minutes for ~5k manhwa titles

## Using the Recommender

### Interactive CLI

```bash
python -m src.cli.main
```

Follow the prompts to:
- Enter a manhwa title (with fuzzy matching support)
- Optionally filter by genres, rating, status
- Adjust diversity level (0 = pure relevance, 1 = maximum diversity)

### Programmatic Usage

```python
from src.recommender.hybrid_recommender import HybridManwhaRecommender

# Load pre-trained model
recommender = HybridManwhaRecommender()
recommender.load_model("models/")

# Get recommendations
recs = recommender.recommend(
    manhwa_title="Solo Leveling",
    n_recommendations=10,
    diversity=0.3  # Add some diversity
)

for rec in recs:
    print(f"{rec['name']}: {rec['recommendation_score']:.2f}")
```

## Advanced Usage

### Custom User Preferences

```python
user_profile = {
    'liked_genres': ['Action', 'Fantasy'],
    'disliked_genres': ['Romance'],
    'min_rating': 4.0,
    'preferred_status': ['RELEASING', 'FINISHED']
}

recs = recommender.recommend(
    manhwa_title="Solo Leveling",
    n_recommendations=10,
    user_profile=user_profile
)
```

### Hyperparameter Tuning

```python
# Create train/validation split
train_df, val_df = recommender.create_evaluation_split(test_ratio=0.2)

# Define search space
param_grid = {
    'weights': [
        {'content': 0.5, 'genre_similarity': 0.3, 'user_pref': 0.2},
        {'content': 0.4, 'genre_similarity': 0.3, 'user_pref': 0.3},
    ],
    'tfidf_max_features': [3000, 5000, 10000],
    'tfidf_min_df': [1, 2, 3]
}

# Optimize for NDCG@10
results = recommender.tune_hyperparameters(
    train_df=train_df,
    val_df=val_df,
    param_grid=param_grid,
    metric='ndcg@k',
    k=10
)

print(f"Best NDCG: {results['best_score']:.4f}")
print(f"Best params: {results['best_params']}")
```

### Cold Start Recommendations

```python
# For new users with no history
cold_start_recs = recommender.handle_cold_start(
    n_recommendations=10,
    popularity_bias=0.5  # 0 = quality only, 1 = popularity only
)
```

### Custom Weighting

```python
# Emphasize content similarity over genres
recommender = HybridManwhaRecommender(
    weights={
        'content': 0.6,         # Increase content weight
        'genre_similarity': 0.2, # Decrease genre weight
        'user_pref': 0.2
    }
)
recommender.prepare_data("data/cleanedManwhas.json")
recommender.build_content_features()
recommender.build_genre_similarity_features()
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Data Collection | O(n) API calls | 5-15 min for 5k items |
| Deduplication | O(n×k) fuzzy matching | 1-2 min for 20k entries |
| Model Training | O(n×d) TF-IDF + O(n²) KNN | 30-60 sec for 5k items |
| Single Recommendation | O(log n) KNN query | 100-500 ms |
| With MMR Re-ranking | O(k²×d) additional | +50-100 ms |

### Space Requirements

- **Models**: ~150-250 MB (serialized)
- **Runtime Memory**: 1-2 GB RAM
- **Dataset**: Optimized for 1k-50k items

### Scalability

- **Current**: 5k manhwa, single-machine
- **For >100k items**: Consider approximate KNN (Annoy, FAISS)
- **For distributed**: Shard by genre, use distributed TF-IDF

## Thread Safety

**⚠️ WARNING: The recommender class is NOT thread-safe.**

Mutable state (`df`, `user_preferences`, `_title_cache`, etc.) is modified without locking.

**Safe Usage:**
```python
# ✓ Separate instances per thread
thread1_recommender = HybridManwhaRecommender()
thread1_recommender.load_model("models/")

thread2_recommender = HybridManwhaRecommender()
thread2_recommender.load_model("models/")

# ✓ Read-only after training
recommender.load_model("models/")
# Now safe to call recommend() from multiple threads
```

**Unsafe Usage:**
```python
# ✗ Shared instance with concurrent training
# ✗ Concurrent calls to prepare_data(), build_content_features()
```

**Solutions:**
1. Use separate instances per thread
2. Add external synchronization (threading.Lock)
3. Use process-based parallelism (multiprocessing)

## Project Structure

```
manwha-recommender/
├── data/                       # Data storage
│   ├── anilist.json           # Raw AniList data
│   ├── jikan.json             # Raw MyAnimeList data
│   ├── mangaupdates.json      # Raw MangaUpdates data
│   ├── animeplanet.json       # Raw Anime-Planet data
│   └── cleanedManwhas.json    # Unified, deduplicated catalog
├── models/                     # Serialized models
│   ├── content_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── genre_model.pkl
│   ├── feature_matrix.pkl
│   └── recommender_config.json
├── src/                        # Source code
│   ├── data_collection/       # Data collectors
│   │   ├── anilist_collector.py
│   │   ├── jikan_collector.py
│   │   ├── mangaupdates_collector.py
│   │   └── animeplanet_collector.py
│   ├── data_processing/       # Data processing
│   │   └── deduplicator.py
│   ├── recommender/           # Recommendation engine
│   │   └── hybrid_recommender.py
│   └── cli/                   # CLI interface
│       └── main.py
├── scripts/                   # Build and utility scripts
│   └── build.py              # Main build script
├── tests/                     # Unit and integration tests
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md       # System architecture
│   └── ALGORITHMS.md         # Algorithm details
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Data Sources**: AniList, MyAnimeList (via Jikan), MangaUpdates, Anime-Planet
- **Libraries**: scikit-learn, pandas, NumPy, SciPy, rapidfuzz
- **Algorithms**: Based on research by Salton & Buckley (TF-IDF), Carbonell & Goldstein (MMR), Järvelin & Kekäläinen (NDCG)
