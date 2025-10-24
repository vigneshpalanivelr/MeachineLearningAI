from typing import List, Dict, Any, Optional, Callable
import logging
import json
import math
import statistics
import sys

logger = logging.getLogger(__name__)


def load_json(path: str) -> List[Dict[str, Any]]:
    """
    Load JSON from file and return list of dicts.
    Raises FileNotFoundError or json.JSONDecodeError.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array (list of records).")

    return data


def safe_load_json(path) -> List[Dict[str, Any]]:
    """Safely load JSON with error handling and graceful exit."""
    try:
        return load_json(path)
    except FileNotFoundError:
        logger.error("File not found: %s", path)
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in: %s", path)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error while loading JSON: %s", e)
        sys.exit(1)


def _to_numbers(values: List[Any]) -> List[float]:
    """Filter list to numeric values (int/float), skipping NaN or non-numeric strings."""
    nums: List[float] = []
    for v in values:
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            nums.append(float(v))
    return nums


def summary_stats(data: List[Dict[str, Any]], field: str) -> Dict[str, Optional[float]]:
    """
    Compute simple summary statistics for a numeric field.
    Returns dict: {count, mean, median, mode, min, max}. Missing or non-numeric values are ignored.
    """
    values = [item.get(field) for item in data if field in item]
    nums = _to_numbers(values)

    if not nums:
        return {"count": 0, "mean": None, "median": None, "mode": None, "min": None, "max": None}

    result = {
            "count": len(nums),
            "mean": statistics.mean(nums),
            "median": statistics.median(nums),
            "min": min(nums),
            "max": max(nums),
        }
    try:
        result["mode"] = statistics.mode(nums)
    except statistics.StatisticsError:
        result["mode"] = None
    return result


def filter_items(items: List[Dict[str, Any]], key: str, predicate: Callable[[Any], bool]) -> List[Dict[str, Any]]:
    """Return records where key exists and predicate(value) is True."""
    results = []
    for it in items:
        if key in it:
            try:
                if predicate(it.get(key)):
                    results.append(it)
            except Exception as e:
                logger.debug("Predicate raised for record %s: %s", it, e)
    return results