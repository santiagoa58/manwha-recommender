"""
Data deduplication system using fuzzy matching to merge entries from multiple sources.
"""

import json
from typing import List, Dict, Optional, Tuple
from rapidfuzz import fuzz, process
import logging
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManwhaDeduplicator:
    """Deduplicates and merges manhwa entries from multiple data sources."""

    # Similarity threshold for considering entries as duplicates
    TITLE_SIMILARITY_THRESHOLD = 85
    ALT_TITLE_SIMILARITY_THRESHOLD = 80

    def __init__(self):
        self.duplicate_groups = []
        self.merged_entries = []

    def normalize_title(self, title: str) -> str:
        """Normalize title for better matching."""
        if not title:
            return ""

        # Convert to lowercase
        title = title.lower()

        # Remove common patterns that cause mismatches
        title = re.sub(r'\s*\(.*?\)\s*', '', title)  # Remove parentheses content
        title = re.sub(r'\s*\[.*?\]\s*', '', title)  # Remove bracket content
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        title = title.strip()

        # Remove common suffixes
        suffixes = [
            'manhwa', 'webtoon', 'part 1', 'part 2', 'part 3',
            'season 1', 'season 2', 'season 3', 's1', 's2', 's3'
        ]
        for suffix in suffixes:
            if title.endswith(suffix):
                title = title[:-len(suffix)].strip()

        return title

    def find_duplicates(self, entries: List[Dict]) -> List[List[Dict]]:
        """
        Find duplicate entries based on title similarity.

        Args:
            entries: List of manhwa entries from various sources

        Returns:
            List of duplicate groups (each group is a list of duplicate entries)
        """
        logger.info(f"Finding duplicates among {len(entries)} entries...")

        # Create index of normalized titles
        title_to_entries = defaultdict(list)
        processed = set()
        duplicate_groups = []

        # Index entries by normalized title for exact matches
        for entry in entries:
            norm_title = self.normalize_title(entry.get('name', ''))
            title_to_entries[norm_title].append(entry)

        # Group exact matches
        for norm_title, group in title_to_entries.items():
            if len(group) > 1:
                duplicate_groups.append(group)
                for entry in group:
                    entry_id = self._get_entry_id(entry)
                    processed.add(entry_id)

        # Find fuzzy matches for remaining entries
        unprocessed = [e for e in entries if self._get_entry_id(e) not in processed]

        logger.info(f"Found {len(duplicate_groups)} exact match groups")
        logger.info(f"Fuzzy matching {len(unprocessed)} remaining entries...")

        fuzzy_groups = self._fuzzy_match_entries(unprocessed)
        duplicate_groups.extend(fuzzy_groups)

        logger.info(f"Total duplicate groups found: {len(duplicate_groups)}")

        self.duplicate_groups = duplicate_groups
        return duplicate_groups

    def _fuzzy_match_entries(self, entries: List[Dict]) -> List[List[Dict]]:
        """Find duplicates using fuzzy string matching."""
        if not entries:
            return []

        fuzzy_groups = []
        processed = set()

        # Create list of titles and their entries
        title_entry_map = {}
        for entry in entries:
            entry_id = self._get_entry_id(entry)
            title = entry.get('name', '')
            title_entry_map[entry_id] = (title, entry)

        # For each entry, find similar titles
        for entry_id, (title, entry) in title_entry_map.items():
            if entry_id in processed:
                continue

            # Find similar titles
            choices = {eid: t for eid, (t, _) in title_entry_map.items() if eid not in processed}
            if not choices:
                continue

            matches = process.extract(
                title,
                choices,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.TITLE_SIMILARITY_THRESHOLD,
                limit=10
            )

            if matches:
                # Create group with current entry and matches
                group = [entry]
                processed.add(entry_id)

                for match_title, score, match_id in matches:
                    if match_id != entry_id:
                        group.append(title_entry_map[match_id][1])
                        processed.add(match_id)

                if len(group) > 1:
                    fuzzy_groups.append(group)

        return fuzzy_groups

    def _get_entry_id(self, entry: Dict) -> str:
        """Get unique identifier for an entry."""
        return entry.get('id', entry.get('name', str(id(entry))))

    def merge_duplicates(self, duplicate_groups: List[List[Dict]]) -> List[Dict]:
        """
        Merge duplicate entries, combining data from multiple sources.

        Args:
            duplicate_groups: List of duplicate groups

        Returns:
            List of merged entries
        """
        logger.info(f"Merging {len(duplicate_groups)} duplicate groups...")

        merged_entries = []

        for group in duplicate_groups:
            merged = self._merge_group(group)
            merged_entries.append(merged)

        # Add entries that weren't duplicates
        logger.info(f"Created {len(merged_entries)} merged entries")

        self.merged_entries = merged_entries
        return merged_entries

    def _merge_group(self, group: List[Dict]) -> Dict:
        """Merge a group of duplicate entries into a single entry."""

        # Source priority: MangaUpdates > AniList > MyAnimeList > Anime-Planet > Reddit
        source_priority = {
            'MangaUpdates': 4,
            'AniList': 3,
            'MyAnimeList': 2,
            'Anime-Planet': 1,
            'Reddit': 0
        }

        # Sort by source priority
        sorted_group = sorted(
            group,
            key=lambda x: source_priority.get(x.get('source', ''), 0),
            reverse=True
        )

        # Start with highest priority entry
        merged = sorted_group[0].copy()

        # Collect all sources
        merged['sources'] = [entry.get('source') for entry in group]
        merged['source_count'] = len(set(merged['sources']))

        # Merge IDs from all sources
        merged['ids'] = {
            'primary': merged.get('id'),
            'anilist': None,
            'mal': None,
            'mangaupdates': None,
        }

        for entry in group:
            if 'anilist_' in entry.get('id', ''):
                merged['ids']['anilist'] = entry.get('id')
            if 'mal_' in entry.get('id', ''):
                merged['ids']['mal'] = entry.get('id')
                merged['ids']['mal_id'] = entry.get('mal_id')
            if 'mu_' in entry.get('id', ''):
                merged['ids']['mangaupdates'] = entry.get('id')
                merged['ids']['mangaupdates_id'] = entry.get('mangaupdates_id')

        # Merge titles - collect all alternative names
        alt_names = set()
        for entry in group:
            name = entry.get('name')
            if name and name != merged['name']:
                alt_names.add(name)

            alt_name = entry.get('altName', '')
            if alt_name:
                for alt in alt_name.split(','):
                    alt = alt.strip()
                    if alt and alt != merged['name']:
                        alt_names.add(alt)

        merged['altName'] = ', '.join(sorted(alt_names)[:5])  # Limit to 5 most relevant

        # Merge ratings - weighted average based on vote count
        ratings = []
        weights = []

        for entry in group:
            rating = entry.get('rating')
            if rating:
                # Weight by popularity/votes
                weight = entry.get('rated_by', entry.get('scored_by', entry.get('popularity', 1)))
                if weight and weight > 0:
                    ratings.append(rating)
                    weights.append(weight)

        if ratings:
            weighted_rating = sum(r * w for r, w in zip(ratings, weights)) / sum(weights)
            merged['rating'] = round(weighted_rating, 2)
            merged['rating_sources'] = len(ratings)

        # Merge descriptions - prefer longest/most detailed
        descriptions = [e.get('description', '') for e in group if e.get('description')]
        if descriptions:
            merged['description'] = max(descriptions, key=len)

        # Merge genres and tags - union of all
        all_genres = set()
        all_tags = set()

        for entry in group:
            all_genres.update(entry.get('genres', []))
            all_tags.update(entry.get('tags', []))

        merged['genres'] = sorted(list(all_genres))
        merged['tags'] = sorted(list(all_tags))

        # Merge popularity - use max
        popularity_values = [e.get('popularity', 0) for e in group if e.get('popularity')]
        if popularity_values:
            merged['popularity'] = max(popularity_values)

        # Merge favourites - use max
        fav_values = [e.get('favourites', 0) for e in group if e.get('favourites')]
        if fav_values:
            merged['favourites'] = max(fav_values)

        # Prefer non-Unknown values for chapters/volumes
        for entry in group:
            if merged.get('chapters') in [None, 'Unknown'] and entry.get('chapters') not in [None, 'Unknown']:
                merged['chapters'] = entry.get('chapters')
            if merged.get('volumes') in [None, 'Unknown'] and entry.get('volumes') not in [None, 'Unknown']:
                merged['volumes'] = entry.get('volumes')

        # Prefer better image URLs
        for entry in group:
            img = entry.get('imageURL')
            if img and ('original' in img or 'large' in img):
                merged['imageURL'] = img
                break

        # Use most recent status
        status_priority = ['RELEASING', 'FINISHED', 'HIATUS', 'CANCELLED', 'NOT_YET_RELEASED']
        for status in status_priority:
            for entry in group:
                if entry.get('status') == status:
                    merged['status'] = status
                    break
            if merged.get('status') == status:
                break

        return merged

    def process_all_sources(
        self,
        anilist_data: List[Dict],
        jikan_data: List[Dict],
        mangaupdates_data: List[Dict],
        animeplanet_data: List[Dict]
    ) -> List[Dict]:
        """
        Process and merge data from all sources.

        Args:
            anilist_data: Entries from AniList
            jikan_data: Entries from Jikan/MAL
            mangaupdates_data: Entries from MangaUpdates
            animeplanet_data: Entries from Anime-Planet

        Returns:
            List of deduplicated and merged entries
        """
        # Combine all entries
        all_entries = []
        all_entries.extend(anilist_data)
        all_entries.extend(jikan_data)
        all_entries.extend(mangaupdates_data)
        all_entries.extend(animeplanet_data)

        logger.info(f"Processing {len(all_entries)} total entries from all sources")
        logger.info(f"  - AniList: {len(anilist_data)}")
        logger.info(f"  - MyAnimeList: {len(jikan_data)}")
        logger.info(f"  - MangaUpdates: {len(mangaupdates_data)}")
        logger.info(f"  - Anime-Planet: {len(animeplanet_data)}")

        # Find duplicates
        duplicate_groups = self.find_duplicates(all_entries)

        # Merge duplicates
        merged = self.merge_duplicates(duplicate_groups)

        # Add non-duplicate entries
        all_entry_ids = {self._get_entry_id(e) for group in duplicate_groups for e in group}
        unique_entries = [e for e in all_entries if self._get_entry_id(e) not in all_entry_ids]

        logger.info(f"Adding {len(unique_entries)} unique (non-duplicate) entries")

        final_entries = merged + unique_entries

        logger.info(f"Final catalog size: {len(final_entries)} manhwa entries")

        return final_entries


def main():
    """Test deduplication system."""
    # Sample test data
    test_entries = [
        {
            'id': 'anilist_1',
            'name': 'Solo Leveling',
            'altName': 'Na Honjaman Level-Up',
            'rating': 4.7,
            'source': 'AniList',
            'description': 'Long description...',
            'genres': ['Action', 'Fantasy'],
            'tags': ['Dungeon', 'OP MC']
        },
        {
            'id': 'mal_1',
            'name': 'Solo Leveling',
            'altName': 'Only I Level Up',
            'rating': 4.8,
            'source': 'MyAnimeList',
            'description': 'Different description...',
            'genres': ['Action', 'Adventure'],
            'tags': ['Game']
        },
        {
            'id': 'mu_1',
            'name': 'Solo Leveling',
            'rating': 4.9,
            'source': 'MangaUpdates',
            'description': 'Most detailed description...',
            'genres': ['Action', 'Fantasy', 'Supernatural'],
            'tags': []
        },
    ]

    dedup = ManwhaDeduplicator()
    duplicate_groups = dedup.find_duplicates(test_entries)
    merged = dedup.merge_duplicates(duplicate_groups)

    print(f"\nFound {len(duplicate_groups)} duplicate group(s)")
    print(f"Merged into {len(merged)} unique entry(ies)\n")

    for entry in merged:
        print(f"Title: {entry['name']}")
        print(f"Sources: {', '.join(entry['sources'])}")
        print(f"Rating: {entry['rating']}/5.0 (from {entry['rating_sources']} sources)")
        print(f"Genres: {', '.join(entry['genres'])}")
        print()


if __name__ == "__main__":
    main()
