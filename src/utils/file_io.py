"""File I/O utilities for loading and saving data."""

import json
import logging
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    """Load JSON file with error handling.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON as dict

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {path}")
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        raise ValueError(f"Invalid JSON in {path}: {e}")


def save_json(path: Path, data: Any, indent: int = 4) -> None:
    """Save data to JSON file.

    Args:
        path: Path to save JSON file
        data: Data to serialize
        indent: Indentation level (default 4)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    logger.info(f"Saved JSON to {path}")


def load_text_lines(path: Path) -> List[str]:
    """Load text file as list of non-empty lines.

    Args:
        path: Path to text file

    Returns:
        List of non-empty, stripped lines

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(lines)} lines from {path}")
            return lines
    except FileNotFoundError:
        logger.error(f"Text file not found: {path}")
        raise FileNotFoundError(f"Text file not found: {path}")
