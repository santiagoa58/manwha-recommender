"""
AniList GraphQL API client for collecting manhwa data.
AniList has 500k+ entries and explicitly supports manhwa.
Rate limit: 90 requests/minute
"""

import asyncio
import time
from typing import List, Dict, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AniListCollector:
    """Collects manhwa data from AniList GraphQL API."""

    BASE_URL = "https://graphql.anilist.co"
    RATE_LIMIT_DELAY = 0.7  # ~85 requests/minute to stay under 90/min limit

    def __init__(self):
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _query(self, query: str, variables: Dict) -> Dict:
        """Execute a GraphQL query with retry logic."""
        self._rate_limit()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.BASE_URL,
                json={"query": query, "variables": variables}
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                raise Exception(f"GraphQL errors: {data['errors']}")

            return data.get("data", {})

    async def search_manhwa(self, page: int = 1, per_page: int = 50) -> List[Dict]:
        """
        Search for manhwa entries.

        Args:
            page: Page number (1-indexed)
            per_page: Results per page (max 50)

        Returns:
            List of manhwa entries with full metadata
        """
        query = """
        query ($page: Int, $perPage: Int, $type: MediaType, $countryOfOrigin: String) {
          Page(page: $page, perPage: $perPage) {
            pageInfo {
              total
              currentPage
              lastPage
              hasNextPage
              perPage
            }
            media(type: $type, countryOfOrigin: $countryOfOrigin, sort: POPULARITY_DESC) {
              id
              idMal
              title {
                romaji
                english
                native
              }
              description(asHtml: false)
              format
              status
              startDate {
                year
                month
                day
              }
              endDate {
                year
                month
                day
              }
              chapters
              volumes
              countryOfOrigin
              isLicensed
              source
              coverImage {
                large
                medium
              }
              bannerImage
              genres
              tags {
                id
                name
                category
                rank
              }
              averageScore
              meanScore
              popularity
              favourites
              trending
              rankings {
                id
                rank
                type
                year
                season
                allTime
              }
              staff {
                edges {
                  node {
                    id
                    name {
                      full
                    }
                  }
                  role
                }
              }
              studios {
                nodes {
                  id
                  name
                }
              }
            }
          }
        }
        """

        variables = {
            "page": page,
            "perPage": per_page,
            "type": "MANGA",
            "countryOfOrigin": "KR"  # Korea for manhwa
        }

        try:
            data = await self._query(query, variables)
            page_info = data.get("Page", {}).get("pageInfo", {})
            media_list = data.get("Page", {}).get("media", [])

            logger.info(f"Retrieved page {page}/{page_info.get('lastPage', '?')} "
                       f"with {len(media_list)} entries")

            return self._transform_entries(media_list), page_info

        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}")
            return [], {}

    def _transform_entries(self, media_list: List[Dict]) -> List[Dict]:
        """Transform AniList entries to unified schema."""
        transformed = []

        for media in media_list:
            try:
                # Calculate rating (AniList uses 0-100 scale, convert to 0-5)
                rating = None
                if media.get("meanScore"):
                    rating = round(media["meanScore"] / 20, 2)  # 100 -> 5.0
                elif media.get("averageScore"):
                    rating = round(media["averageScore"] / 20, 2)

                # Format dates
                start_date = self._format_date(media.get("startDate"))
                end_date = self._format_date(media.get("endDate"))
                years = f"{start_date} - {end_date}" if start_date else "Unknown"

                # Extract genres and tags
                genres = media.get("genres", [])
                tags = [tag["name"] for tag in media.get("tags", [])
                       if tag.get("rank", 0) >= 60]  # Only high-confidence tags
                all_tags = list(set(genres + tags))

                # Get alternative titles
                titles = media.get("title", {})
                alt_names = []
                if titles.get("english"):
                    alt_names.append(titles["english"])
                if titles.get("native"):
                    alt_names.append(titles["native"])
                alt_name_str = ", ".join(alt_names) if alt_names else ""

                # Primary title (prefer English, fallback to romaji)
                primary_title = titles.get("english") or titles.get("romaji") or "Unknown"

                # Get staff info
                authors = []
                for edge in media.get("staff", {}).get("edges", []):
                    role = edge.get("role", "").lower()
                    if "story" in role or "art" in role or "author" in role:
                        name = edge.get("node", {}).get("name", {}).get("full")
                        if name:
                            authors.append(f"{name} ({edge.get('role')})")

                entry = {
                    "id": f"anilist_{media['id']}",
                    "mal_id": media.get("idMal"),
                    "name": primary_title,
                    "altName": alt_name_str,
                    "description": media.get("description", "").replace("<br>", "\n").replace("<i>", "").replace("</i>", ""),
                    "rating": rating,
                    "popularity": media.get("popularity", 0),
                    "favourites": media.get("favourites", 0),
                    "tags": all_tags,
                    "genres": genres,
                    "format": media.get("format"),  # MANGA, NOVEL, ONE_SHOT
                    "status": media.get("status"),  # FINISHED, RELEASING, etc.
                    "chapters": media.get("chapters", "Unknown"),
                    "volumes": media.get("volumes", "Unknown"),
                    "years": years,
                    "start_date": start_date,
                    "end_date": end_date,
                    "imageURL": media.get("coverImage", {}).get("large"),
                    "banner_image": media.get("bannerImage"),
                    "country": media.get("countryOfOrigin", "KR"),
                    "source": "AniList",
                    "is_licensed": media.get("isLicensed", False),
                    "authors": authors,
                    "source_material": media.get("source"),  # ORIGINAL, MANGA, etc.
                }

                transformed.append(entry)

            except Exception as e:
                logger.error(f"Error transforming entry {media.get('id')}: {e}")
                continue

        return transformed

    def _format_date(self, date_obj: Optional[Dict]) -> Optional[str]:
        """Format AniList date object to string."""
        if not date_obj:
            return None

        year = date_obj.get("year")
        if not year:
            return None

        month = date_obj.get("month")
        day = date_obj.get("day")

        if month and day:
            return f"{year}-{month:02d}-{day:02d}"
        elif month:
            return f"{year}-{month:02d}"
        else:
            return str(year)

    async def collect_all_manhwa(self, max_pages: Optional[int] = None) -> List[Dict]:
        """
        Collect all manhwa from AniList.

        Args:
            max_pages: Maximum number of pages to fetch (None for all)

        Returns:
            List of all manhwa entries
        """
        all_manhwa = []
        page = 1
        has_next_page = True

        logger.info("Starting AniList manhwa collection...")

        while has_next_page:
            if max_pages and page > max_pages:
                logger.info(f"Reached max_pages limit: {max_pages}")
                break

            entries, page_info = await self.search_manhwa(page=page, per_page=50)
            all_manhwa.extend(entries)

            has_next_page = page_info.get("hasNextPage", False)
            page += 1

            # Progress update
            if page % 10 == 0:
                logger.info(f"Collected {len(all_manhwa)} manhwa entries so far...")

        logger.info(f"Collection complete! Total entries: {len(all_manhwa)}")
        return all_manhwa

    async def get_manhwa_by_id(self, anilist_id: int) -> Optional[Dict]:
        """Get a specific manhwa by AniList ID."""
        query = """
        query ($id: Int) {
          Media(id: $id, type: MANGA) {
            id
            idMal
            title {
              romaji
              english
              native
            }
            description
            genres
            averageScore
          }
        }
        """

        try:
            data = await self._query(query, {"id": anilist_id})
            media = data.get("Media")
            if media:
                return self._transform_entries([media])[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching manhwa {anilist_id}: {e}")
            return None


async def main():
    """Test the AniList collector."""
    collector = AniListCollector()

    # Test: Get first 3 pages (150 entries)
    manhwa_list = await collector.collect_all_manhwa(max_pages=3)

    print(f"\n{'='*60}")
    print(f"Collected {len(manhwa_list)} manhwa entries from AniList")
    print(f"{'='*60}\n")

    # Show sample entries
    for i, manhwa in enumerate(manhwa_list[:5], 1):
        print(f"{i}. {manhwa['name']}")
        print(f"   Rating: {manhwa['rating']}/5.0")
        print(f"   Genres: {', '.join(manhwa['genres'][:3])}")
        print(f"   Status: {manhwa['status']}")
        print(f"   Popularity: {manhwa['popularity']:,}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
