"""
Class example for testing.
"""

class User:
    """User class for testing."""

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        self.active = True

    def get_display_name(self) -> str:
        """Get display name."""
        return self.name.title()

    def deactivate(self) -> None:
        """Deactivate user."""
        self.active = False

class Product:
    """Product class for testing."""

    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price
        self.in_stock = 0

    def add_stock(self, quantity: int) -> None:
        """Add stock."""
        self.in_stock += quantity

    def get_total_value(self) -> float:
        """Get total value."""
        return self.price * self.in_stock