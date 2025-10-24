"""
MangaUpdates API client for collecting manhwa data.
Official API at https://api.mangaupdates.com/v1
Most comprehensive manga/manhwa database with detailed metadata.
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


class MangaUpdatesCollector(BaseAPICollector):
    """
    Collects manhwa data from MangaUpdates official API.

    MangaUpdates provides the most comprehensive manga/manhwa database
    with detailed metadata including bayesian ratings, categories,
    publishers, and user recommendations.

    Rate limit: Conservative 0.5s delay between requests
    API: https://api.mangaupdates.com/v1

    Usage:
        async with MangaUpdatesCollector() as collector:
            manhwa_list = await collector.collect_all_manhwa(max_entries=150)
    """

    BASE_URL = "https://api.mangaupdates.com/v1"
    RATE_LIMIT_DELAY = 0.5  # Conservative rate limiting

    async def search_manhwa(
        self,
        page: int = 1,
        per_page: int = 50,
        query: str = "",
        type_filter: str = "Manhwa"
    ) -> tuple[List[Dict], int]:
        """
        Search for manhwa entries.

        Args:
            page: Page number (1-indexed)
            per_page: Results per page
            query: Search query
            type_filter: Type filter (Manhwa, Manga, Manhua, etc.)

        Returns:
            Tuple of (transformed entries, total_count)
        """
        params = {
            "page": page,
            "perpage": per_page,
            "orderby": "rating",  # Order by rating
        }

        if query:
            params["search"] = query

        if type_filter:
            params["type"] = type_filter

        try:
            # MangaUpdates search endpoint
            data = await self._get("series/search", params)

            results = data.get("results", [])
            total_hits = data.get("total_hits", 0)

            logger.info(f"Retrieved page {page} with {len(results)} entries (Total: {total_hits})")

            # Get full details for each result in parallel using asyncio.gather()
            # This fixes N+1 query problem by fetching all details concurrently
            series_ids = [
                result.get("record", {}).get("series_id")
                for result in results
                if result.get("record", {}).get("series_id")
            ]

            if not series_ids:
                return [], total_hits

            # Fetch all details in parallel
            tasks = [self.get_series_details(series_id) for series_id in series_ids]
            detailed_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out None results and exceptions
            detailed_entries = [
                entry for entry in detailed_results
                if entry is not None and not isinstance(entry, Exception)
            ]

            # Log any errors
            errors = [r for r in detailed_results if isinstance(r, Exception)]
            if errors:
                logger.warning(f"Failed to fetch {len(errors)}/{len(series_ids)} series details")

            return detailed_entries, total_hits

        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}")
            return [], 0

    async def get_series_details(self, series_id: int) -> Optional[Dict]:
        """
        Get detailed information for a specific series.

        Args:
            series_id: MangaUpdates series ID

        Returns:
            Transformed entry data or None if not found
        """
        try:
            data = await self._get(f"series/{series_id}")
            return self._transform_entry(data)

        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {e}")
            return None

    def _transform_entry(self, entry: Dict) -> Optional[Dict]:
        """
        Transform a single MangaUpdates entry to unified schema.

        Args:
            entry: Raw MangaUpdates API entry

        Returns:
            Transformed entry or None if transformation fails
        """
        try:
            # Calculate rating (MangaUpdates uses bayesian average 0-10, convert to 0-5)
            rating = None
            bayesian_rating = entry.get("bayesian_rating")
            if bayesian_rating:
                rating_value = self._safe_float(bayesian_rating)
                if rating_value is not None:
                    rating = round(rating_value / 2, 2)

            # Extract genres and categories
            genres = [g["genre"] for g in entry.get("genres", [])]
            categories = [c["category"] for c in entry.get("categories", [])]
            all_tags = list(set(genres + categories))

            # Get alternative titles
            alt_titles = []
            for assoc_name in entry.get("associated", []):
                title = assoc_name.get("title")
                if title:
                    alt_titles.append(title)

            alt_name_str = ", ".join(alt_titles[:3]) if alt_titles else ""

            # Primary title
            primary_title = entry.get("title", "Unknown")

            # Get authors and artists
            authors = []
            for author in entry.get("authors", []):
                name = author.get("name")
                author_type = author.get("type", "Author")
                if name:
                    authors.append(f"{name} ({author_type})")

            # Get publishers
            publishers = []
            for pub in entry.get("publishers", []):
                pub_name = pub.get("publisher_name")
                pub_type = pub.get("type", "")
                if pub_name:
                    publishers.append(f"{pub_name} ({pub_type})")

            # Format year
            year = entry.get("year")
            status = entry.get("status", "Unknown")
            if year and status == "Complete":
                years = f"{year} - Complete"
            elif year:
                years = f"{year} - Ongoing"
            else:
                years = "Unknown"

            # Status mapping
            status_map = {
                "Complete": "FINISHED",
                "Ongoing": "RELEASING",
                "Hiatus": "HIATUS",
                "Cancelled": "CANCELLED"
            }
            mapped_status = status_map.get(status, status)

            # Get recommendations (related series)
            recommendations = []
            for rec in entry.get("recommendations", []):
                rec_title = rec.get("series_name")
                if rec_title:
                    recommendations.append(rec_title)

            transformed = {
                "id": f"mu_{entry['series_id']}",
                "mangaupdates_id": entry["series_id"],
                "name": primary_title,
                "altName": alt_name_str,
                "description": entry.get("description", ""),
                "rating": rating,
                "bayesian_rating": bayesian_rating,
                "rating_votes": entry.get("rating_votes", 0),
                "tags": all_tags,
                "genres": genres,
                "categories": categories,
                "format": entry.get("type"),  # Manhwa, Manga, etc.
                "status": mapped_status,
                "original_status": status,
                "years": years,
                "year": year,
                "latest_chapter": entry.get("latest_chapter"),
                "imageURL": entry.get("image", {}).get("url", {}).get("original"),
                "country": "KR" if entry.get("type") == "Manhwa" else "Unknown",
                "source": "MangaUpdates",
                "authors": authors,
                "publishers": publishers,
                "serialization": entry.get("url"),
                "anime": entry.get("anime", {}).get("start") is not None,  # Has anime adaptation
                "forum_id": entry.get("forum_id"),
                "recommendations": recommendations[:10],  # Limit to 10
                "rank": entry.get("rank", {}).get("position"),
            }

            return transformed

        except Exception as e:
            logger.error(f"Error transforming entry {entry.get('series_id')}: {e}")
            return None

    async def collect_all_manhwa(self, max_entries: Optional[int] = None) -> List[Dict]:
        """
        Collect all manhwa from MangaUpdates.

        Args:
            max_entries: Maximum number of entries to fetch (None for all available)

        Returns:
            List of all manhwa entries
        """
        all_manhwa = []
        page = 1
        per_page = 50

        logger.info("Starting MangaUpdates manhwa collection...")

        while True:
            if max_entries and len(all_manhwa) >= max_entries:
                logger.info(f"Reached max_entries limit: {max_entries}")
                all_manhwa = all_manhwa[:max_entries]
                break

            entries, total_hits = await self.search_manhwa(
                page=page,
                per_page=per_page,
                type_filter="Manhwa"
            )

            if not entries:
                logger.info("No more entries found")
                break

            all_manhwa.extend(entries)

            # Check if we've collected all available entries
            if len(all_manhwa) >= total_hits:
                logger.info(f"Collected all available entries: {total_hits}")
                break

            page += 1

            # Progress update
            if page % 5 == 0:
                logger.info(f"Collected {len(all_manhwa)}/{total_hits} manhwa entries...")

        logger.info(f"Collection complete! Total entries: {len(all_manhwa)}")
        return all_manhwa

    async def get_series_comments(self, series_id: int, page: int = 1) -> List[Dict]:
        """
        Get user comments/reviews for a specific series.

        Args:
            series_id: MangaUpdates series ID
            page: Page number (1-indexed)

        Returns:
            List of comment entries
        """
        try:
            data = await self._get(f"series/{series_id}/comments", {"page": page})
            return data.get("comments", [])
        except Exception as e:
            logger.error(f"Error fetching comments for {series_id}: {e}")
            return []


async def main():
    """Test the MangaUpdates collector."""
    async with MangaUpdatesCollector() as collector:
        # Test: Get first 3 pages (150 entries max)
        manhwa_list = await collector.collect_all_manhwa(max_entries=150)

        print(f"\n{'='*60}")
        print(f"Collected {len(manhwa_list)} manhwa entries from MangaUpdates")
        print(f"{'='*60}\n")

        # Show sample entries
        for i, manhwa in enumerate(manhwa_list[:5], 1):
            print(f"{i}. {manhwa['name']}")
            print(f"   MU ID: {manhwa['mangaupdates_id']}")
            print(f"   Rating: {manhwa['rating']}/5.0 (Bayesian: {manhwa.get('bayesian_rating', 'N/A')}/10)")
            print(f"   Genres: {', '.join(manhwa['genres'][:3])}")
            print(f"   Status: {manhwa['status']}")
            print(f"   Year: {manhwa['year']}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
