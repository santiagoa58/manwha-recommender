"""
Jikan API client for collecting manhwa data from MyAnimeList.
Jikan is an unofficial REST API for MyAnimeList.
Rate limit: 3 requests/second, 60 requests/minute
"""

import asyncio
import time
from typing import List, Dict, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# REVIEW: [HIGH] Code duplication with AniListCollector
# Recommendation: Extract common rate limiting and HTTP client logic to base class
# Location: JikanCollector class
class JikanCollector:
    """Collects manhwa data from Jikan (MyAnimeList) API."""

    BASE_URL = "https://api.jikan.moe/v4"
    RATE_LIMIT_DELAY = 0.35  # ~2.8 requests/second to stay under 3/sec limit

    def __init__(self):
        self.last_request_time = 0

    # REVIEW: [HIGH] Using time.sleep() in async context blocks event loop
    # Recommendation: Use asyncio.sleep() instead
    # Location: _rate_limit function
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Execute a GET request with retry logic."""
        self._rate_limit()

        # REVIEW: [MEDIUM] Same client recreation issue as AniListCollector
        # Recommendation: Reuse client instance
        # Location: Lines 39-45
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/{endpoint}",
                params=params or {}
            )
            response.raise_for_status()
            return response.json()

    async def search_manhwa(self, page: int = 1, query: str = "") -> tuple[List[Dict], Dict]:
        """
        Search for manhwa/manga entries.

        Args:
            page: Page number (1-indexed)
            query: Search query (empty for all)

        Returns:
            Tuple of (entries, pagination info)
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
            Detailed entry data
        """
        try:
            data = await self._get(f"manga/{mal_id}/full")
            entry = data.get("data")

            if entry:
                return self._transform_entries([entry])[0]
            return None

        except Exception as e:
            logger.error(f"Error fetching manga {mal_id}: {e}")
            return None

    def _transform_entries(self, entries: List[Dict]) -> List[Dict]:
        """Transform Jikan entries to unified schema."""
        transformed = []

        for entry in entries:
            try:
                # REVIEW: [LOW] No validation of score range
            # Recommendation: Validate score is between 0-10 before conversion
            # Location: Lines 110-113
            # Calculate rating (MAL uses 0-10 scale, convert to 0-5)
                rating = None
                if entry.get("score"):
                    rating = round(entry["score"] / 2, 2)  # 10 -> 5.0

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

                transformed.append(transformed_entry)

            except Exception as e:
                logger.error(f"Error transforming entry {entry.get('mal_id')}: {e}")
                continue

        return transformed

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
        """Get statistics for a specific manhwa."""
        try:
            data = await self._get(f"manga/{mal_id}/statistics")
            return data.get("data", {})
        except Exception as e:
            logger.error(f"Error fetching stats for {mal_id}: {e}")
            return None

    async def get_manhwa_reviews(self, mal_id: int, page: int = 1) -> List[Dict]:
        """Get user reviews for a specific manhwa."""
        try:
            data = await self._get(f"manga/{mal_id}/reviews", {"page": page})
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Error fetching reviews for {mal_id}: {e}")
            return []


async def main():
    """Test the Jikan collector."""
    collector = JikanCollector()

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
