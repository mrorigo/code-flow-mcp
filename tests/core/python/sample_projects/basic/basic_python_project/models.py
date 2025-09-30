"""Data models for basic Python project."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class User:
    """User data model."""

    name: str
    email: str
    created_at: Optional[datetime] = None
    is_active: bool = True
    preferences: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default created_at if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()

    def get_display_name(self) -> str:
        """Get formatted display name."""
        return self.name.title()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
            "preferences": self.preferences
        }

    def update_preferences(self, **kwargs) -> None:
        """Update user preferences."""
        self.preferences.update(kwargs)


@dataclass
class Product:
    """Product data model."""

    name: str
    price: float
    description: str = ""
    in_stock: int = 0
    tags: List[str] = field(default_factory=list)

    def get_total_value(self) -> float:
        """Calculate total value of stock."""
        return self.price * self.in_stock

    def add_stock(self, quantity: int) -> None:
        """Add items to stock."""
        self.in_stock += quantity

    def remove_stock(self, quantity: int) -> bool:
        """Remove items from stock."""
        if quantity > self.in_stock:
            return False
        self.in_stock -= quantity
        return True

    def is_available(self) -> bool:
        """Check if product is available."""
        return self.in_stock > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "price": self.price,
            "description": self.description,
            "in_stock": self.in_stock,
            "tags": self.tags
        }


@dataclass
class Order:
    """Order data model."""

    user_id: str
    items: List[Dict[str, Any]]
    order_date: Optional[datetime] = None
    status: str = "pending"
    total_amount: float = 0.0

    def __post_init__(self):
        """Set default order_date if not provided."""
        if self.order_date is None:
            self.order_date = datetime.now()

    def calculate_total(self) -> float:
        """Calculate total order amount."""
        return sum(item.get("price", 0) * item.get("quantity", 0) for item in self.items)

    def add_item(self, product: Product, quantity: int) -> None:
        """Add product to order."""
        self.items.append({
            "product_name": product.name,
            "price": product.price,
            "quantity": quantity
        })
        self.total_amount = self.calculate_total()

    def get_summary(self) -> Dict[str, Any]:
        """Get order summary."""
        return {
            "user_id": self.user_id,
            "item_count": len(self.items),
            "total_amount": self.total_amount,
            "status": self.status,
            "order_date": self.order_date.isoformat() if self.order_date else None
        }


class UserManager:
    """Manager class for user operations."""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.next_id = 1

    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        user_id = str(self.next_id)
        self.next_id += 1

        user = User(name=name, email=email)
        self.users[user_id] = user
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_all_users(self) -> List[User]:
        """Get all users."""
        return list(self.users.values())

    def get_active_users(self) -> List[User]:
        """Get only active users."""
        return [user for user in self.users.values() if user.is_active]


class ProductCatalog:
    """Manager class for product operations."""

    def __init__(self):
        self.products: Dict[str, Product] = {}
        self.categories: Dict[str, List[str]] = {}

    def add_product(self, product: Product, category: str = "general") -> None:
        """Add product to catalog."""
        self.products[product.name] = product

        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(product.name)

    def get_product(self, name: str) -> Optional[Product]:
        """Get product by name."""
        return self.products.get(name)

    def get_products_by_category(self, category: str) -> List[Product]:
        """Get all products in a category."""
        product_names = self.categories.get(category, [])
        return [self.products[name] for name in product_names if name in self.products]

    def search_products(self, query: str) -> List[Product]:
        """Search products by name or tags."""
        query = query.lower()
        results = []

        for product in self.products.values():
            if (query in product.name.lower() or
                any(query in tag.lower() for tag in product.tags)):
                results.append(product)

        return results


def serialize_to_json(obj: Any) -> str:
    """Serialize object to JSON string."""
    if hasattr(obj, 'to_dict'):
        data = obj.to_dict()
    else:
        data = obj

    return json.dumps(data, indent=2, default=str)


def deserialize_from_json(json_str: str, target_class: type) -> Any:
    """Deserialize JSON string to object."""
    data = json.loads(json_str)

    if hasattr(target_class, '__dataclass_fields__'):
        # Handle dataclass
        return target_class(**data)
    else:
        # Handle regular class
        return target_class(**data)