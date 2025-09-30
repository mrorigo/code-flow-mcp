"""Utility functions for basic Python project."""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def greet_user(name: str, greeting: str = "Hello") -> str:
    """Greet a user with a customizable greeting.

    Args:
        name: Name of the user
        greeting: Type of greeting to use

    Returns:
        Formatted greeting string
    """
    return f"{greeting}, {name}!"


def calculate_total(items: List[float], discount: float = 0.0) -> float:
    """Calculate total price of items with optional discount.

    Args:
        items: List of item prices
        discount: Discount percentage (0.0 to 1.0)

    Returns:
        Total price after discount
    """
    subtotal = sum(items)
    discount_amount = subtotal * discount
    return subtotal - discount_amount


def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process dictionary data with validation.

    Args:
        data: Input dictionary to process

    Returns:
        Processed data with metadata

    Raises:
        ValueError: If data is invalid
    """
    if not data:
        raise ValueError("Data cannot be empty")

    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")

    processed = {
        "original": data,
        "keys": list(data.keys()),
        "count": len(data),
        "processed_at": "timestamp_here"  # Would use datetime in real app
    }

    return processed


def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read and parse JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """Write data to JSON file.

    Args:
        file_path: Path where to save JSON file
        data: Data to write
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def filter_by_key_value(
    items: List[Dict[str, Any]],
    key: str,
    value: Any
) -> List[Dict[str, Any]]:
    """Filter list of dictionaries by key-value pair.

    Args:
        items: List of dictionaries to filter
        key: Key to check
        value: Value to match

    Returns:
        Filtered list of dictionaries
    """
    return [item for item in items if item.get(key) == value]


def sort_by_key(items: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """Sort list of dictionaries by key.

    Args:
        items: List of dictionaries to sort
        key: Key to sort by
        reverse: Whether to sort in reverse order

    Returns:
        Sorted list of dictionaries
    """
    try:
        return sorted(items, key=lambda x: x[key], reverse=reverse)
    except (KeyError, TypeError):
        # Return original order if sorting fails
        return items


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary (later dicts override earlier ones)
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def safe_divide(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Division result or None if division by zero
    """
    try:
        return a / b
    except ZeroDivisionError:
        return None