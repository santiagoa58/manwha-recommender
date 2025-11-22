"""
Shared pytest fixtures and configuration.
"""

import pytest
import json
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_manhwa_entry():
    """Sample manhwa entry for testing."""
    return {
        "id": "test_123",
        "name": "Solo Leveling",
        "altName": "Na Honjaman Level-Up",
        "description": "E-class hunter Jinwoo Sung is the weakest of them all...",
        "rating": 4.8,
        "popularity": 250000,
        "favourites": 50000,
        "genres": ["Action", "Fantasy", "Adventure"],
        "tags": ["Dungeon", "OP MC", "Leveling System"],
        "format": "Manhwa",
        "status": "FINISHED",
        "chapters": 200,
        "volumes": 45,
        "years": "2018 - 2023",
        "imageURL": "https://example.com/image.jpg",
        "country": "KR",
        "source": "AniList"
    }


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
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_catalog_file(temp_data_dir, sample_manhwa_list):
    """Create a temporary catalog file for testing."""
    catalog_path = temp_data_dir / "test_catalog.json"
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(sample_manhwa_list, f)
    return catalog_path


@pytest.fixture
def mock_anilist_response():
    """Mock AniList GraphQL API response."""
    return {
        "data": {
            "Page": {
                "pageInfo": {
                    "total": 100,
                    "currentPage": 1,
                    "lastPage": 2,
                    "hasNextPage": True,
                    "perPage": 50
                },
                "media": [
                    {
                        "id": 123456,
                        "idMal": 789,
                        "title": {
                            "romaji": "Solo Leveling",
                            "english": "Solo Leveling",
                            "native": "나 혼자만 레벨업"
                        },
                        "description": "E-class hunter story",
                        "format": "MANGA",
                        "status": "FINISHED",
                        "startDate": {"year": 2018, "month": 3, "day": 4},
                        "endDate": {"year": 2023, "month": 1, "day": 1},
                        "chapters": 200,
                        "volumes": 45,
                        "countryOfOrigin": "KR",
                        "isLicensed": True,
                        "coverImage": {
                            "large": "https://example.com/image.jpg"
                        },
                        "genres": ["Action", "Fantasy"],
                        "tags": [
                            {"name": "Dungeon", "rank": 80},
                            {"name": "OP MC", "rank": 75}
                        ],
                        "meanScore": 94,
                        "averageScore": 94,
                        "popularity": 250000,
                        "favourites": 50000,
                        "staff": {
                            "edges": [
                                {
                                    "node": {"name": {"full": "Chugong"}},
                                    "role": "Story"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }


@pytest.fixture
def mock_jikan_response():
    """Mock Jikan API response."""
    return {
        "data": [
            {
                "mal_id": 123,
                "title": "Solo Leveling",
                "titles": [
                    {"type": "Default", "title": "Solo Leveling"},
                    {"type": "English", "title": "Solo Leveling"},
                    {"type": "Japanese", "title": "俺だけレベルアップな件"}
                ],
                "type": "Manhwa",
                "score": 9.4,
                "scored_by": 100000,
                "members": 250000,
                "favorites": 50000,
                "synopsis": "E-class hunter story",
                "status": "Finished",
                "chapters": 200,
                "volumes": 45,
                "published": {
                    "from": "2018-03-04T00:00:00+00:00",
                    "to": "2023-01-01T00:00:00+00:00"
                },
                "genres": [
                    {"name": "Action"},
                    {"name": "Fantasy"}
                ],
                "themes": [
                    {"name": "Survival"}
                ],
                "demographics": [
                    {"name": "Shounen"}
                ],
                "authors": [
                    {"name": "Chugong"}
                ],
                "serializations": [
                    {"name": "Kakao Page"}
                ],
                "images": {
                    "jpg": {
                        "large_image_url": "https://example.com/image.jpg"
                    }
                },
                "rank": 42,
                "url": "https://myanimelist.net/manga/123"
            }
        ],
        "pagination": {
            "has_next_page": False,
            "last_visible_page": 1
        }
    }


@pytest.fixture
def mock_mangaupdates_response():
    """Mock MangaUpdates API response."""
    return {
        "series_id": 456,
        "title": "Solo Leveling",
        "associated": [
            {"title": "Na Honjaman Level-Up"}
        ],
        "type": "Manhwa",
        "description": "E-class hunter story",
        "bayesian_rating": 9.8,
        "rating_votes": 50000,
        "year": "2018",
        "status": "Complete",
        "latest_chapter": 200,
        "genres": [
            {"genre": "Action"},
            {"genre": "Fantasy"}
        ],
        "categories": [
            {"category": "Dungeons"},
            {"category": "Overpowered MC"}
        ],
        "image": {
            "url": {
                "original": "https://example.com/image.jpg"
            }
        },
        "authors": [
            {"name": "Chugong", "type": "Story"}
        ],
        "publishers": [
            {"publisher_name": "Kakao", "type": "Original"}
        ],
        "recommendations": [
            {"series_name": "Tower of God"}
        ],
        "rank": {
            "position": 12
        },
        "url": "https://www.mangaupdates.com/series/456",
        "forum_id": 789
    }


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and speed tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer than 5 seconds"
    )
    config.addinivalue_line(
        "markers", "api: Tests that make real API calls"
    )
