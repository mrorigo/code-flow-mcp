"""
Basic Python function for testing.
"""

def greet_user(name: str) -> str:
    """Greets a user by name."""
    return f"Hello, {name}!"

def calculate_total(items: list) -> float:
    """Calculate total of items."""
    return sum(items)

def process_data(data: dict) -> str:
    """Process dictionary data."""
    if not data:
        return "empty"

    result = []
    for key, value in data.items():
        result.append(f"{key}: {value}")

    return " | ".join(result)