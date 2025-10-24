"""
Jikan API client for collecting manhwa data from MyAnimeList.
Jikan is an unofficial REST API for MyAnimeList.
Rate limit: 3 requests/second, 60 requests/minute
"""

import asyncio
from typing import List, Dict, Optional
import logging
from src.data_collectors.base_collector import (
    BaseAPICollector,
    TransformationError,
    logger
)

logger = logging.getLogger(__name__)


class JikanCollector(BaseAPICollector):
    """
    Collects manhwa data from Jikan (MyAnimeList) API.

    Jikan provides unofficial access to MyAnimeList data with comprehensive
    metadata including ratings, genres, themes, authors, and reviews.

    Rate limit: 3 requests/second (~0.35s delay between requests)
    API: https://api.jikan.moe/v4

    Usage:
        async with JikanCollector() as collector:
            manhwa_list = await collector.collect_all_manhwa(max_pages=3)
    """

    BASE_URL = "https://api.jikan.moe/v4"
    RATE_LIMIT_DELAY = 0.35  # ~2.8 requests/second to stay under 3/sec limit

    async def search_manhwa(self, page: int = 1, query: str = "") -> tuple[List[Dict], Dict]:
        """
        Search for manhwa/manga entries.

        Args:
            page: Page number (1-indexed)
            query: Search query (empty for all)

        Returns:
            Tuple of (transformed entries, pagination info dict)
        """
        params = {
            "page": page,
            "limit": 25,  # Jikan default max
            "type": "manhwa",  # Filter for manhwa only
            "order_by": "members",  # Sort by popularity
            "sort": "desc"
        }

        if query:
            params["q"] = query

        try:
            data = await self._get("manga", params)
            entries = data.get("data", [])
            pagination = data.get("pagination", {})

            logger.info(f"Retrieved page {page} with {len(entries)} entries")

            return self._transform_entries(entries), pagination

        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}")
            return [], {}

    async def get_manga_details(self, mal_id: int) -> Optional[Dict]:
        """
        Get detailed information for a specific manga/manhwa.

        Args:
            mal_id: MyAnimeList ID

        Returns:
            Transformed entry data or None if not found
        """
        try:
            data = await self._get(f"manga/{mal_id}/full")
            entry = data.get("data")

            if entry:
                return self._transform_entry(entry)
            return None

        except Exception as e:
            logger.error(f"Error fetching manga {mal_id}: {e}")
            return None

    def _transform_entry(self, entry: Dict) -> Optional[Dict]:
        """
        Transform a single Jikan entry to unified schema.

        Args:
            entry: Raw Jikan API entry

        Returns:
            Transformed entry or None if transformation fails
        """
        try:
            # Calculate rating (MAL uses 0-10 scale, convert to 0-5)
            rating = None
            if entry.get("score"):
                score = self._safe_float(entry["score"], 0)
                if 0 <= score <= 10:
                    rating = round(score / 2, 2)  # 10 -> 5.0

            # Extract genres and themes
            genres = [g["name"] for g in entry.get("genres", [])]
            themes = [t["name"] for t in entry.get("themes", [])]
            demographics = [d["name"] for d in entry.get("demographics", [])]
            all_tags = list(set(genres + themes + demographics))

            # Get alternative titles
            alt_titles = []
            title_obj = entry.get("titles", [])
            for title_entry in title_obj:
                if title_entry.get("type") in ["English", "Japanese", "Synonym"]:
                    alt_titles.append(title_entry["title"])

            # Remove duplicates and primary title
            primary_title = entry.get("title", "Unknown")
            alt_titles = [t for t in alt_titles if t != primary_title]
            alt_name_str = ", ".join(alt_titles[:3]) if alt_titles else ""  # Limit to 3

            # Format publication dates
            published = entry.get("published", {})
            from_date = published.get("from", "").split("T")[0] if published.get("from") else None
            to_date = published.get("to", "").split("T")[0] if published.get("to") else None

            if from_date and to_date:
                years = f"{from_date[:4]} - {to_date[:4]}"
            elif from_date:
                years = f"{from_date[:4]} - Ongoing"
            else:
                years = "Unknown"

            # Get authors
            authors = []
            for author in entry.get("authors", []):
                authors.append(author["name"])

            # Status mapping
            status_map = {
                "Publishing": "RELEASING",
                "Finished": "FINISHED",
                "On Hiatus": "HIATUS",
                "Discontinued": "CANCELLED",
                "Not yet published": "NOT_YET_RELEASED"
            }
            status = status_map.get(entry.get("status"), entry.get("status"))

            transformed_entry = {
                "id": f"mal_{entry['mal_id']}",
                "mal_id": entry["mal_id"],
                "name": primary_title,
                "altName": alt_name_str,
                "description": entry.get("synopsis", ""),
                "rating": rating,
                "popularity": entry.get("members", 0),  # Members count as popularity
                "favourites": entry.get("favorites", 0),
                "tags": all_tags,
                "genres": genres,
                "themes": themes,
                "demographics": demographics,
                "format": entry.get("type"),  # Manga, Manhwa, Manhua, etc.
                "status": status,
                "chapters": entry.get("chapters", "Unknown"),
                "volumes": entry.get("volumes", "Unknown"),
                "years": years,
                "start_date": from_date,
                "end_date": to_date,
                "imageURL": entry.get("images", {}).get("jpg", {}).get("large_image_url"),
                "country": "KR" if entry.get("type") == "Manhwa" else "Unknown",
                "source": "MyAnimeList",
                "authors": authors,
                "serializations": [s["name"] for s in entry.get("serializations", [])],
                "score": entry.get("score"),
                "scored_by": entry.get("scored_by", 0),
                "rank": entry.get("rank"),
                "url": entry.get("url"),
            }

            return transformed_entry

        except Exception as e:
            logger.error(f"Error transforming entry {entry.get('mal_id')}: {e}")
            return None

    async def collect_all_manhwa(self, max_pages: Optional[int] = None) -> List[Dict]:
        """
        Collect all manhwa from MyAnimeList via Jikan.

        Args:
            max_pages: Maximum number of pages to fetch (None for all)

        Returns:
            List of all manhwa entries
        """
        all_manhwa = []
        page = 1

        logger.info("Starting Jikan (MAL) manhwa collection...")

        while True:
            if max_pages and page > max_pages:
                logger.info(f"Reached max_pages limit: {max_pages}")
                break

            entries, pagination = await self.search_manhwa(page=page)

            if not entries:
                logger.info("No more entries found")
                break

            all_manhwa.extend(entries)

            # Check if there are more pages
            has_next_page = pagination.get("has_next_page", False)
            if not has_next_page:
                logger.info("Reached last page")
                break

            page += 1

            # Progress update
            if page % 10 == 0:
                logger.info(f"Collected {len(all_manhwa)} manhwa entries so far...")

        logger.info(f"Collection complete! Total entries: {len(all_manhwa)}")
        return all_manhwa

    async def get_manhwa_stats(self, mal_id: int) -> Optional[Dict]:
        """
        Get statistics for a specific manhwa.

        Args:
            mal_id: MyAnimeList ID

        Returns:
            Statistics data or None if not found
        """
        try:
            data = await self._get(f"manga/{mal_id}/statistics")
            return data.get("data", {})
        except Exception as e:
            logger.error(f"Error fetching stats for {mal_id}: {e}")
            return None

    async def get_manhwa_reviews(self, mal_id: int, page: int = 1) -> List[Dict]:
        """
        Get user reviews for a specific manhwa.

        Args:
            mal_id: MyAnimeList ID
            page: Page number (1-indexed)

        Returns:
            List of review entries
        """
        try:
            data = await self._get(f"manga/{mal_id}/reviews", {"page": page})
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Error fetching reviews for {mal_id}: {e}")
            return []


async def main():
    """Test the Jikan collector."""
    async with JikanCollector() as collector:
        # Test: Get first 3 pages (75 entries max)
        manhwa_list = await collector.collect_all_manhwa(max_pages=3)

        print(f"\n{'='*60}")
        print(f"Collected {len(manhwa_list)} manhwa entries from MyAnimeList")
        print(f"{'='*60}\n")

        # Show sample entries
        for i, manhwa in enumerate(manhwa_list[:5], 1):
            print(f"{i}. {manhwa['name']}")
            print(f"   MAL ID: {manhwa['mal_id']}")
            print(f"   Rating: {manhwa['rating']}/5.0 (Score: {manhwa.get('score', 'N/A')}/10)")
            print(f"   Genres: {', '.join(manhwa['genres'][:3])}")
            print(f"   Status: {manhwa['status']}")
            print(f"   Members: {manhwa['popularity']:,}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
