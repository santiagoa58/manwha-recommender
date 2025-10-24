"""Fixtures for hybrid recommender tests."""

import pytest
import json
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_catalog_file(temp_data_dir):
    """Create a temporary catalog file for testing."""
    catalog_data = [
        {
            "name": "Solo Leveling",
            "altName": "Na Honjaman Level-Up",
            "description": "E-class hunter Jinwoo Sung becomes the world's only Solo Leveler",
            "rating": 4.8,
            "popularity": 250000,
            "genres": ["Action", "Fantasy", "Adventure"],
            "tags": ["Dungeon", "OP MC", "Leveling System"],
            "status": "FINISHED",
            "source": "AniList"
        },
        {
            "name": "Tower of God",
            "altName": "Sinui Tap",
            "description": "Twenty-Fifth Bam climbs a mysterious tower to find his friend",
            "rating": 4.5,
            "popularity": 200000,
            "genres": ["Action", "Adventure", "Fantasy"],
            "tags": ["Tower", "Game", "Mystery"],
            "status": "RELEASING",
            "source": "MyAnimeList"
        },
        {
            "name": "The Beginning After The End",
            "altName": "TBATE",
            "description": "King Grey reincarnates in a world of magic and monsters",
            "rating": 4.7,
            "popularity": 180000,
            "genres": ["Action", "Fantasy", "Drama"],
            "tags": ["Reincarnation", "Magic", "OP MC"],
            "status": "RELEASING",
            "source": "MangaUpdates"
        },
        {
            "name": "Omniscient Reader",
            "altName": "Omniscient Reader's Viewpoint",
            "description": "Dokja Kim's favorite novel becomes reality",
            "rating": 4.6,
            "popularity": 150000,
            "genres": ["Action", "Fantasy", "Drama"],
            "tags": ["Game", "Survival"],
            "status": "RELEASING",
            "source": "AniList"
        },
        {
            "name": "Return of the Mount Hua Sect",
            "altName": "Return of the Blossoming Blade",
            "description": "Chung Myung returns 100 years after his death",
            "rating": 4.9,
            "popularity": 120000,
            "genres": ["Action", "Martial Arts", "Fantasy"],
            "tags": ["Reincarnation", "Murim", "OP MC"],
            "status": "RELEASING",
            "source": "MangaUpdates"
        },
        {
            "name": "Romance Novel",
            "altName": "Love Story",
            "description": "A romantic story",
            "rating": 4.0,
            "popularity": 50000,
            "genres": ["Romance", "Drama"],
            "tags": ["School Life", "Love"],
            "status": "FINISHED",
            "source": "AniList"
        }
    ]

    catalog_path = temp_data_dir / "test_catalog.json"
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog_data, f)

    return catalog_path
