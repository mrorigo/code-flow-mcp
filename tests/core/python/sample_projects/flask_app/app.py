"""Flask application for testing."""

from flask import Flask, jsonify, request, render_template_string
from typing import Dict, List, Optional, Any
import json

app = Flask(__name__)


# Sample data storage (in production, use a database)
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
]

products = [
    {"id": 1, "name": "Laptop", "price": 999.99, "category": "electronics"},
    {"id": 2, "name": "Book", "price": 19.99, "category": "books"},
    {"id": 3, "name": "Coffee Mug", "price": 9.99, "category": "home"}
]


@app.route('/')
def home():
    """Home page."""
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flask Test App</title>
        </head>
        <body>
            <h1>Welcome to Flask Test App</h1>
            <p><a href="/users">View Users</a></p>
            <p><a href="/products">View Products</a></p>
            <p><a href="/api/health">Health Check</a></p>
        </body>
        </html>
    ''')


@app.route('/users')
def list_users():
    """List all users."""
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Users</title>
        </head>
        <body>
            <h1>Users</h1>
            <ul>
            {% for user in users %}
                <li>{{ user.name }} - {{ user.email }}</li>
            {% endfor %}
            </ul>
            <p><a href="/">Back to Home</a></p>
        </body>
        </html>
    ''', users=users)


@app.route('/products')
def list_products():
    """List all products."""
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Products</title>
        </head>
        <body>
            <h1>Products</h1>
            <ul>
            {% for product in products %}
                <li>{{ product.name }} - ${{ product.price }} ({{ product.category }})</li>
            {% endfor %}
            </ul>
            <p><a href="/">Back to Home</a></p>
        </body>
        </html>
    ''', products=products)


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "Flask application is running",
        "timestamp": "2023-12-01T00:00:00Z"
    })


@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users."""
    return jsonify({
        "users": users,
        "count": len(users)
    })


@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id: int):
    """Get user by ID."""
    user = next((u for u in users if u["id"] == user_id), None)
    if user is None:
        return jsonify({"error": "User not found"}), 404

    return jsonify(user)


@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user."""
    data = request.get_json()

    if not data or 'name' not in data or 'email' not in data:
        return jsonify({"error": "Name and email are required"}), 400

    # Check if email already exists
    if any(u["email"] == data["email"] for u in users):
        return jsonify({"error": "Email already exists"}), 409

    # Create new user
    new_user = {
        "id": max(u["id"] for u in users) + 1,
        "name": data["name"],
        "email": data["email"]
    }

    users.append(new_user)

    return jsonify(new_user), 201


@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id: int):
    """Update user by ID."""
    user = next((u for u in users if u["id"] == user_id), None)
    if user is None:
        return jsonify({"error": "User not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Update user fields
    if 'name' in data:
        user["name"] = data["name"]
    if 'email' in data:
        # Check if new email conflicts with existing users
        if any(u["email"] == data["email"] and u["id"] != user_id for u in users):
            return jsonify({"error": "Email already exists"}), 409
        user["email"] = data["email"]

    return jsonify(user)


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id: int):
    """Delete user by ID."""
    global users
    user = next((u for u in users if u["id"] == user_id), None)
    if user is None:
        return jsonify({"error": "User not found"}), 404

    users = [u for u in users if u["id"] != user_id]

    return jsonify({"message": "User deleted successfully"})


@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products."""
    category = request.args.get('category')
    if category:
        filtered_products = [p for p in products if p["category"] == category]
    else:
        filtered_products = products

    return jsonify({
        "products": filtered_products,
        "count": len(filtered_products)
    })


@app.route('/api/products/<int:product_id>', methods=['GET'])
def get_product(product_id: int):
    """Get product by ID."""
    product = next((p for p in products if p["id"] == product_id), None)
    if product is None:
        return jsonify({"error": "Product not found"}), 404

    return jsonify(product)


@app.route('/api/search')
def search():
    """Search users and products."""
    query = request.args.get('q', '').lower()

    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    # Search in users
    matching_users = [
        u for u in users
        if query in u["name"].lower() or query in u["email"].lower()
    ]

    # Search in products
    matching_products = [
        p for p in products
        if query in p["name"].lower() or query in p["category"].lower()
    ]

    return jsonify({
        "query": query,
        "users": matching_users,
        "products": matching_products,
        "user_count": len(matching_users),
        "product_count": len(matching_products)
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


def create_app() -> Flask:
    """Create and configure the Flask app."""
    return app


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)