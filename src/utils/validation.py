"""Data validation for manwha data structures."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ManwhaData:
    """Validated manwha data structure.

    Attributes:
        name: Manwha title (required)
        rating: Rating from 0-5 (required)
        tags: List of genre/theme tags (required)
        description: Plot description (required)
        altName: Alternative title (optional)
        chapters: Chapter information (optional)
    """

    name: str
    rating: float
    tags: List[str]
    description: str
    altName: Optional[str] = None
    chapters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate data after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Manwha name cannot be empty")

        if not isinstance(self.rating, (int, float)):
            raise ValueError(f"Rating must be numeric, got {type(self.rating)}")

        if not 0 <= self.rating <= 5:
            raise ValueError(f"Rating must be 0-5, got {self.rating}")

        if not isinstance(self.tags, list):
            raise ValueError(f"Tags must be a list, got {type(self.tags)}")

        if not self.description:
            logger.warning(f"Manwha '{self.name}' has no description")

    @classmethod
    def from_dict(cls, data: dict) -> "ManwhaData":
        """Create from dictionary with validation.

        Args:
            data: Dictionary containing manwha data

        Returns:
            Validated ManwhaData instance

        Raises:
            ValueError: If required fields missing or invalid
        """
        try:
            return cls(
                name=data["name"],
                rating=float(data.get("rating", 0)),
                tags=data.get("tags", []),
                description=data.get("description", ""),
                altName=data.get("altName"),
                chapters=data.get("chapters"),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid data format: {e}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "rating": self.rating,
            "tags": self.tags,
            "description": self.description,
            "altName": self.altName,
            "chapters": self.chapters,
        }


def validate_manwha_list(data: List[dict]) -> List[ManwhaData]:
    """Validate a list of manwha dictionaries.

    Args:
        data: List of manwha dictionaries

    Returns:
        List of validated ManwhaData objects

    Raises:
        ValueError: If data is not a list or contains invalid entries
    """
    if not isinstance(data, list):
        raise ValueError(f"Expected list of manwhas, got {type(data)}")

    validated = []
    errors = []

    for i, item in enumerate(data):
        try:
            validated.append(ManwhaData.from_dict(item))
        except ValueError as e:
            errors.append(f"Item {i}: {e}")
            logger.error(f"Validation error for item {i}: {e}")

    if errors:
        logger.warning(f"Found {len(errors)} validation errors out of {len(data)} items")

    return validated
