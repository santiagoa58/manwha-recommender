"""
Master data collection script that orchestrates gathering data from all sources.
This script:
1. Collects data from AniList, Jikan, MangaUpdates, and Anime-Planet
2. Deduplicates and merges entries
3. Saves to unified catalog
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collectors.anilist_collector import AniListCollector
from src.data_collectors.jikan_collector import JikanCollector
from src.data_collectors.mangaupdates_collector import MangaUpdatesCollector
from src.data_processing.deduplicator import ManwhaDeduplicator
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# REVIEW: [MEDIUM] Class does too much - violates Single Responsibility Principle
# Recommendation: Split into separate classes for collection, persistence, and orchestration
# Location: DataCollectionOrchestrator class
class DataCollectionOrchestrator:
    """Orchestrates data collection from multiple sources."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # REVIEW: [LOW] All collectors instantiated even if not used
        # Recommendation: Use lazy initialization or dependency injection
        # Location: Lines 38-41
        self.anilist_collector = AniListCollector()
        self.jikan_collector = JikanCollector()
        self.mangaupdates_collector = MangaUpdatesCollector()
        self.deduplicator = ManwhaDeduplicator()

    async def collect_from_anilist(self, max_pages: int = None) -> list:
        """Collect data from AniList."""
        logger.info("=" * 60)
        logger.info("COLLECTING FROM ANILIST")
        logger.info("=" * 60)

        data = await self.anilist_collector.collect_all_manhwa(max_pages=max_pages)

        # Save raw data
        output_file = self.data_dir / "raw_anilist_manhwa.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data)} entries to {output_file}")
        return data

    async def collect_from_jikan(self, max_pages: int = None) -> list:
        """Collect data from Jikan/MyAnimeList."""
        logger.info("=" * 60)
        logger.info("COLLECTING FROM MYANIMELIST (via Jikan)")
        logger.info("=" * 60)

        data = await self.jikan_collector.collect_all_manhwa(max_pages=max_pages)

        # Save raw data
        output_file = self.data_dir / "raw_mal_manhwa.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data)} entries to {output_file}")
        return data

    async def collect_from_mangaupdates(self, max_entries: int = None) -> list:
        """Collect data from MangaUpdates."""
        logger.info("=" * 60)
        logger.info("COLLECTING FROM MANGAUPDATES")
        logger.info("=" * 60)

        data = await self.mangaupdates_collector.collect_all_manhwa(max_entries=max_entries)

        # Save raw data
        output_file = self.data_dir / "raw_mangaupdates_manhwa.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data)} entries to {output_file}")
        return data

    def load_animeplanet_data(self) -> list:
        """Load existing Anime-Planet data."""
        logger.info("=" * 60)
        logger.info("LOADING EXISTING ANIME-PLANET DATA")
        logger.info("=" * 60)

        input_file = self.data_dir / "cleanedManwhas.json"

        if not input_file.exists():
            logger.warning(f"Anime-Planet data not found at {input_file}")
            return []

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} entries from {input_file}")
        return data

    async def collect_all(
        self,
        anilist_max_pages: int = None,
        jikan_max_pages: int = None,
        mangaupdates_max_entries: int = None,
        skip_sources: list = None,
    ) -> dict:
        """
        Collect data from all sources.

        Args:
            anilist_max_pages: Max pages from AniList (None = all)
            jikan_max_pages: Max pages from Jikan (None = all)
            mangaupdates_max_entries: Max entries from MangaUpdates (None = all)
            skip_sources: List of sources to skip ['anilist', 'jikan', 'mangaupdates', 'animeplanet']

        Returns:
            Dictionary with data from each source
        """
        skip_sources = skip_sources or []

        logger.info("\n" + "=" * 60)
        logger.info("STARTING COMPREHENSIVE DATA COLLECTION")
        logger.info("=" * 60 + "\n")

        results = {"anilist": [], "jikan": [], "mangaupdates": [], "animeplanet": []}

        # Collect from each source
        # REVIEW: [MEDIUM] Bare except catches all exceptions too broadly
        # Recommendation: Catch specific exceptions (httpx.HTTPError, etc.) or re-raise critical ones
        # Location: Lines 142-146
        if "anilist" not in skip_sources:
            try:
                results["anilist"] = await self.collect_from_anilist(max_pages=anilist_max_pages)
            except Exception as e:
                logger.error(f"Error collecting from AniList: {e}")

        if "jikan" not in skip_sources:
            try:
                results["jikan"] = await self.collect_from_jikan(max_pages=jikan_max_pages)
            except Exception as e:
                logger.error(f"Error collecting from Jikan: {e}")

        if "mangaupdates" not in skip_sources:
            try:
                results["mangaupdates"] = await self.collect_from_mangaupdates(
                    max_entries=mangaupdates_max_entries
                )
            except Exception as e:
                logger.error(f"Error collecting from MangaUpdates: {e}")

        if "animeplanet" not in skip_sources:
            try:
                results["animeplanet"] = self.load_animeplanet_data()
            except Exception as e:
                logger.error(f"Error loading Anime-Planet data: {e}")

        return results

    # REVIEW: [MEDIUM] No rollback mechanism if process fails mid-way
    # Recommendation: Use atomic writes (write to temp file, then rename)
    # Location: deduplicate_and_merge function
    def deduplicate_and_merge(self, all_data: dict) -> list:
        """Deduplicate and merge data from all sources."""
        logger.info("\n" + "=" * 60)
        logger.info("DEDUPLICATING AND MERGING DATA")
        logger.info("=" * 60 + "\n")

        # REVIEW: [LOW] No error handling if deduplication fails
        # Recommendation: Wrap in try-except and save partial results
        # Location: Lines 174-179
        merged_data = self.deduplicator.process_all_sources(
            anilist_data=all_data["anilist"],
            jikan_data=all_data["jikan"],
            mangaupdates_data=all_data["mangaupdates"],
            animeplanet_data=all_data["animeplanet"],
        )

        # REVIEW: [MEDIUM] No atomic write pattern - file corruption risk
        # Recommendation: Write to temp file then rename: tempfile.NamedTemporaryFile()
        # Location: Lines 182-184
        # Save merged catalog
        output_file = self.data_dir / "master_manhwa_catalog.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved merged catalog to {output_file}")

        # Save metadata
        metadata = {
            "collection_date": datetime.now().isoformat(),
            "total_entries": len(merged_data),
            "source_counts": {
                "anilist": len(all_data["anilist"]),
                "jikan": len(all_data["jikan"]),
                "mangaupdates": len(all_data["mangaupdates"]),
                "animeplanet": len(all_data["animeplanet"]),
            },
            "deduplication_stats": {
                "total_raw_entries": sum(len(v) for v in all_data.values()),
                "merged_entries": len(merged_data),
                "duplicate_groups": len(self.deduplicator.duplicate_groups),
            },
        }

        metadata_file = self.data_dir / "collection_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_file}")

        return merged_data

    def generate_statistics(self, merged_data: list):
        """Generate and display statistics about the collected data."""
        logger.info("\n" + "=" * 60)
        logger.info("DATA COLLECTION STATISTICS")
        logger.info("=" * 60 + "\n")

        total = len(merged_data)
        logger.info(f"Total unique manhwa entries: {total}")

        # Rating distribution
        with_ratings = [m for m in merged_data if m.get("rating")]
        logger.info(
            f"Entries with ratings: {len(with_ratings)} ({len(with_ratings)/total*100:.1f}%)"
        )

        if with_ratings:
            avg_rating = sum(m["rating"] for m in with_ratings) / len(with_ratings)
            logger.info(f"Average rating: {avg_rating:.2f}/5.0")

        # Status distribution
        status_counts = {}
        for m in merged_data:
            status = m.get("status", "Unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        logger.info("\nStatus distribution:")
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {status}: {count} ({count/total*100:.1f}%)")

        # Genre distribution
        genre_counts = {}
        for m in merged_data:
            for genre in m.get("genres", []):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        logger.info("\nTop 10 genres:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {genre}: {count} ({count/total*100:.1f}%)")

        # Multi-source entries
        multi_source = [m for m in merged_data if m.get("source_count", 1) > 1]
        logger.info(
            f"\nEntries from multiple sources: {len(multi_source)} ({len(multi_source)/total*100:.1f}%)"
        )

        logger.info("\n" + "=" * 60 + "\n")


async def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect manhwa data from all sources")
    parser.add_argument("--anilist-pages", type=int, help="Max pages from AniList (default: all)")
    parser.add_argument("--jikan-pages", type=int, help="Max pages from Jikan (default: all)")
    parser.add_argument(
        "--mangaupdates-entries", type=int, help="Max entries from MangaUpdates (default: all)"
    )
    parser.add_argument(
        "--skip", nargs="+", help="Sources to skip (anilist, jikan, mangaupdates, animeplanet)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode - collect only a few pages from each"
    )

    args = parser.parse_args()

    # Test mode - quick collection for testing
    if args.test:
        logger.info("Running in TEST MODE - collecting limited data")
        args.anilist_pages = 2
        args.jikan_pages = 2
        args.mangaupdates_entries = 100

    orchestrator = DataCollectionOrchestrator()

    # Collect from all sources
    all_data = await orchestrator.collect_all(
        anilist_max_pages=args.anilist_pages,
        jikan_max_pages=args.jikan_pages,
        mangaupdates_max_entries=args.mangaupdates_entries,
        skip_sources=args.skip or [],
    )

    # Deduplicate and merge
    merged_data = orchestrator.deduplicate_and_merge(all_data)

    # Generate statistics
    orchestrator.generate_statistics(merged_data)

    logger.info("Data collection complete!")
    logger.info(f"Master catalog saved to: data/master_manhwa_catalog.json")
    logger.info(f"Total unique manhwa: {len(merged_data)}")


if __name__ == "__main__":
    asyncio.run(main())
