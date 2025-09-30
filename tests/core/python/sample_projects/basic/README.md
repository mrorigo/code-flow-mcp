# Basic Python Project

A basic Python project for testing purposes.

## Features

- Simple utility functions
- Basic class definitions
- Type annotations
- Exception handling
- File operations

## Installation

```bash
pip install -e .
```

## Usage

```python
from basic_python_project.utils import greet_user
from basic_python_project.models import User

# Use functions
message = greet_user("Alice")
print(message)

# Use classes
user = User("Alice", "alice@example.com")
print(user.get_display_name())