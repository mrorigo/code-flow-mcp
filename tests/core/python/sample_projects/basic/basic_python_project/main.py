"""Main module for basic Python project."""

import click
from typing import List, Optional

from .utils import greet_user, calculate_total
from .models import User, Product


@click.command()
@click.option("--name", default="World", help="Name to greet")
@click.option("--items", multiple=True, type=float, help="Items to calculate total")
def main(name: str, items: List[float]) -> None:
    """Main CLI entry point."""
    # Use utility functions
    greeting = greet_user(name)
    click.echo(greeting)

    if items:
        total = calculate_total(list(items))
        click.echo(f"Total: {total}")

    # Use classes
    user = User(name, f"{name.lower()}@example.com")
    click.echo(f"User: {user.get_display_name()}")

    product = Product("Test Item", 29.99)
    product.add_stock(5)
    click.echo(f"Product value: ${product.get_total_value()}")


if __name__ == "__main__":
    main()