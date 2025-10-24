"""Fixtures for deduplication tests."""

import pytest


@pytest.fixture
def duplicate_manhwa_entries():
    """Sample entries with duplicates for testing deduplication."""
    return [
        {
            "id": "anilist_1",
            "name": "Solo Leveling",
            "altName": "Na Honjaman Level-Up",
            "rating": 4.7,
            "popularity": 50000,
            "genres": ["Action", "Fantasy"],
            "tags": ["Dungeon"],
            "source": "AniList",
            "description": "Short description"
        },
        {
            "id": "mal_1",
            "name": "Solo Leveling",
            "altName": "Only I Level Up",
            "rating": 4.8,
            "popularity": 100000,
            "genres": ["Action", "Adventure"],
            "tags": ["Game"],
            "source": "MyAnimeList",
            "description": "Different description"
        },
        {
            "id": "mu_1",
            "name": "Solo Leveling",
            "rating": 4.9,
            "popularity": 200000,
            "genres": ["Action", "Fantasy", "Supernatural"],
            "tags": [],
            "source": "MangaUpdates",
            "description": "Most detailed description here"
        },
        {
            "id": "test_4",
            "name": "Tower of God",
            "rating": 4.5,
            "genres": ["Action"],
            "tags": [],
            "source": "AniList",
            "description": "Tower story"
        }
    ]


@pytest.fixture
def sample_manhwa_list():
    """List of sample manhwa entries for testing."""
    return [
        {
            "id": "test_1",
            "name": "Solo Leveling",
            "altName": "Na Honjaman Level-Up",
            "description": "Hunter story",
            "rating": 4.8,
            "popularity": 250000,
            "genres": ["Action", "Fantasy"],
            "tags": ["Dungeon", "OP MC"],
            "status": "FINISHED",
            "source": "AniList"
        },
        {
            "id": "test_2",
            "name": "Tower of God",
            "altName": "Sinui Tap",
            "description": "Tower climbing story",
            "rating": 4.5,
            "popularity": 200000,
            "genres": ["Action", "Adventure", "Fantasy"],
            "tags": ["Tower", "Game"],
            "status": "RELEASING",
            "source": "MyAnimeList"
        },
        {
            "id": "test_3",
            "name": "The Beginning After The End",
            "altName": "TBATE",
            "description": "Reincarnation story",
            "rating": 4.7,
            "popularity": 180000,
            "genres": ["Action", "Fantasy", "Drama"],
            "tags": ["Reincarnation", "Magic"],
            "status": "RELEASING",
            "source": "MangaUpdates"
        }
    ]
