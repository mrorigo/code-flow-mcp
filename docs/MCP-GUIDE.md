# Detailed Guide to Implementing a Model Context Protocol Server Using FastMCP

The Model Context Protocol (MCP) is an open protocol that enables seamless integration between LLM applications and external data sources and tools. This guide provides comprehensive implementation details for creating MCP servers using the official Python SDK's FastMCP framework.[1]

## What is MCP?

The Model Context Protocol (MCP) lets you build servers that expose data and functionality to LLM applications in a secure, standardized way. MCP servers can:[1]

- **Expose data through Resources** (similar to GET endpoints; used to load information into the LLM's context)
- **Provide functionality through Tools** (similar to POST endpoints; used to execute code or produce side effects)
- **Define interaction patterns through Prompts** (reusable templates for LLM interactions)

## Installation and Setup

Install the official MCP Python SDK using `uv` (recommended) or `pip`:[1]

```bash
# Using uv (recommended)
uv init mcp-server-demo
cd mcp-server-demo
uv add "mcp[cli]"

# Using pip
pip install "mcp[cli]"
```

Verify installation:
```bash
uv run mcp
```

## Core FastMCP Implementation

### 1. Basic Server Structure

Every FastMCP application starts with a `FastMCP` class instance:[1]

```python
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo Server")

# Add basic tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Add dynamic resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# Add prompt template
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }
    return f"{styles.get(style, styles['friendly'])} for someone named {name}."
```

### 2. Tools - Function Calling for AI

Tools allow LLMs to execute Python functions and perform actions:[1]

```python
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

mcp = FastMCP(name="Calculator Server")

@mcp.tool()
def calculate_statistics(numbers: list[float], operation: str) -> dict:
    """Perform statistical calculations on a list of numbers."""
    if not numbers:
        return {"error": "Empty list provided"}
    
    result = {
        "input": numbers,
        "operation": operation,
        "count": len(numbers)
    }
    
    if operation == "sum":
        result["result"] = sum(numbers)
    elif operation == "average":
        result["result"] = sum(numbers) / len(numbers)
    elif operation == "max":
        result["result"] = max(numbers)
    elif operation == "min":
        result["result"] = min(numbers)
    else:
        result["error"] = f"Unknown operation: {operation}"
    
    return result

@mcp.tool()
async def long_running_task(
    task_name: str, 
    ctx: Context[ServerSession, None], 
    steps: int = 5
) -> str:
    """Execute a task with progress updates."""
    await ctx.info(f"Starting: {task_name}")
    
    for i in range(steps):
        progress = (i + 1) / steps
        await ctx.report_progress(
            progress=progress,
            total=1.0,
            message=f"Step {i + 1}/{steps}",
        )
        await ctx.debug(f"Completed step {i + 1}")
    
    return f"Task '{task_name}' completed"
```

### 3. Resources - Read-Only Data Access

Resources provide read-only data access to LLMs:[1]

```python
@mcp.resource("file://documents/{name}")
def read_document(name: str) -> str:
    """Read a document by name."""
    # In production, this would read from an actual file system
    documents = {
        "readme": "# Welcome to our application\n\nThis is a sample document.",
        "config": "debug=false\nport=8080\nhost=localhost",
        "changelog": "v1.0.0 - Initial release\nv1.1.0 - Added new features"
    }
    return documents.get(name, f"Document '{name}' not found")

@mcp.resource("api://users/{user_id}/profile")
def get_user_profile(user_id: str) -> dict:
    """Get user profile by ID."""
    # Mock user data - in production, this would query a database
    return {
        "user_id": user_id,
        "name": f"User {user_id}",
        "status": "active",
        "last_login": "2024-09-18T10:00:00Z",
        "permissions": ["read", "write"]
    }

@mcp.resource("config://settings")
def get_settings() -> str:
    """Get application settings."""
    return """{
  "theme": "dark",
  "language": "en",
  "debug": false,
  "api_version": "v1"
}"""
```

### 4. Structured Output Support

FastMCP supports various return types for structured data:[1]

```python
from typing import TypedDict
from pydantic import BaseModel, Field

# Using Pydantic models for rich structured data
class WeatherData(BaseModel):
    """Weather information structure."""
    temperature: float = Field(description="Temperature in Celsius")
    humidity: float = Field(description="Humidity percentage")
    condition: str
    wind_speed: float
    location: str

@mcp.tool()
def get_detailed_weather(city: str) -> WeatherData:
    """Get detailed weather data for a city."""
    return WeatherData(
        temperature=22.5,
        humidity=45.0,
        condition="sunny",
        wind_speed=5.2,
        location=city
    )

# Using TypedDict for simpler structures
class LocationInfo(TypedDict):
    latitude: float
    longitude: float
    name: str
    country: str

@mcp.tool()
def get_location(address: str) -> LocationInfo:
    """Get location coordinates."""
    return LocationInfo(
        latitude=51.5074, 
        longitude=-0.1278, 
        name="London", 
        country="UK"
    )

# Using regular classes with type hints
class DatabaseQuery:
    query: str
    execution_time: float
    rows_affected: int
    success: bool

    def __init__(self, query: str, execution_time: float, rows_affected: int, success: bool = True):
        self.query = query
        self.execution_time = execution_time
        self.rows_affected = rows_affected
        self.success = success

@mcp.tool()
def execute_query(sql: str) -> DatabaseQuery:
    """Execute a database query."""
    # Simulate query execution
    import time
    start_time = time.time()
    # Simulate processing
    execution_time = 0.045
    
    return DatabaseQuery(
        query=sql,
        execution_time=execution_time,
        rows_affected=10,
        success=True
    )
```

### 5. Context Access and Advanced Features

Tools can access context for logging, progress reporting, and advanced capabilities:[1]

```python
import asyncio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

mcp = FastMCP(name="Advanced Server")

@mcp.tool()
async def process_large_dataset(
    dataset_name: str, 
    ctx: Context[ServerSession, None],
    chunk_size: int = 1000
) -> str:
    """Process a large dataset with progress reporting."""
    
    await ctx.info(f"Starting processing of {dataset_name}")
    
    # Simulate processing in chunks
    total_items = 10000
    for i in range(0, total_items, chunk_size):
        current_chunk = min(chunk_size, total_items - i)
        
        # Process chunk (simulate work)
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Report progress
        progress = (i + current_chunk) / total_items
        await ctx.report_progress(
            progress=progress,
            total=1.0,
            message=f"Processed {i + current_chunk}/{total_items} items"
        )
        
        await ctx.debug(f"Completed chunk {i//chunk_size + 1}")
    
    await ctx.info(f"Finished processing {dataset_name}")
    return f"Successfully processed {total_items} items from {dataset_name}"

@mcp.tool()
async def read_and_analyze_resource(resource_uri: str, ctx: Context) -> dict:
    """Read a resource and perform analysis."""
    
    await ctx.info(f"Reading resource: {resource_uri}")
    
    try:
        # Read resource using context
        resource_content = await ctx.read_resource(resource_uri)
        content_text = resource_content[0].text
        
        # Perform analysis
        analysis = {
            "resource_uri": resource_uri,
            "content_length": len(content_text),
            "word_count": len(content_text.split()),
            "line_count": len(content_text.splitlines()),
            "request_id": ctx.request_id,
            "client_id": ctx.client_id
        }
        
        await ctx.info("Analysis completed successfully")
        return analysis
        
    except Exception as e:
        await ctx.error(f"Failed to read resource: {str(e)}")
        raise
```

### 6. Application Lifecycle Management

FastMCP supports sophisticated lifecycle management with dependency injection:[1]

```python
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import aiohttp

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

# Application dependencies
class DatabaseConnection:
    """Mock database connection."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.is_connected = False
    
    async def connect(self):
        """Establish database connection."""
        await asyncio.sleep(0.1)  # Simulate connection time
        self.is_connected = True
        print(f"Connected to database: {self.connection_string}")
    
    async def disconnect(self):
        """Close database connection."""
        self.is_connected = False
        print("Database connection closed")
    
    async def query(self, sql: str) -> list:
        """Execute database query."""
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        
        # Simulate query execution
        await asyncio.sleep(0.05)
        return [{"id": 1, "name": "Sample Data", "value": "test"}]

@dataclass
class AppContext:
    """Application context with typed dependencies."""
    db: DatabaseConnection
    http_session: aiohttp.ClientSession
    config: dict

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with dependencies."""
    
    print("Initializing application resources...")
    
    # Initialize resources on startup
    db = DatabaseConnection("postgresql://localhost/mydb")
    await db.connect()
    
    http_session = aiohttp.ClientSession()
    
    config = {
        "api_version": "v1",
        "max_connections": 100,
        "timeout": 30,
        "debug": True
    }
    
    try:
        yield AppContext(db=db, http_session=http_session, config=config)
    finally:
        # Cleanup on shutdown
        print("Cleaning up application resources...")
        await http_session.close()
        await db.disconnect()

# Create server with lifecycle management
mcp = FastMCP("Production Server", lifespan=app_lifespan)

@mcp.tool()
async def fetch_user_data(
    user_id: int, 
    ctx: Context[ServerSession, AppContext]
) -> dict:
    """Fetch user data using managed resources."""
    
    # Access typed lifespan context
    app_ctx = ctx.request_context.lifespan_context
    
    # Use database connection
    db_results = await app_ctx.db.query(f"SELECT * FROM users WHERE id = {user_id}")
    
    # Use HTTP session for external API call
    try:
        async with app_ctx.http_session.get(
            f"https://api.example.com/users/{user_id}",
            timeout=app_ctx.config["timeout"]
        ) as response:
            external_data = await response.json() if response.status == 200 else {}
    except Exception as e:
        external_data = {"error": str(e)}
    
    return {
        "user_id": user_id,
        "database_data": db_results,
        "external_data": external_data,
        "api_version": app_ctx.config["api_version"]
    }
```

### 7. Authentication and Security

For production deployments, FastMCP supports OAuth 2.1 authentication:[1]

```python
from pydantic import AnyHttpUrl
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

class ProductionTokenVerifier(TokenVerifier):
    """Production token verifier implementation."""
    
    def __init__(self, jwks_url: str, required_audience: str):
        self.jwks_url = jwks_url
        self.required_audience = required_audience
    
    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify JWT token against JWKS endpoint."""
        try:
            # In production, implement actual JWT verification
            # using libraries like python-jose or pyjwt
            
            # Mock verification for example
            if token.startswith("valid_"):
                return AccessToken(
                    token=token,
                    scopes=["read", "write"],
                    expires_at=1234567890,
                    subject="user123"
                )
            return None
            
        except Exception as e:
            print(f"Token verification failed: {e}")
            return None

# Create authenticated server
mcp = FastMCP(
    "Secure API Server",
    token_verifier=ProductionTokenVerifier(
        jwks_url="https://auth.example.com/.well-known/jwks.json",
        required_audience="mcp-api"
    ),
    auth=AuthSettings(
        issuer_url=AnyHttpUrl("https://auth.example.com"),
        resource_server_url=AnyHttpUrl("https://api.example.com"),
        required_scopes=["user", "api"],
    ),
)

@mcp.tool()
async def secure_operation(data: str) -> dict:
    """A secure operation that requires authentication."""
    return {
        "message": "Secure operation completed",
        "data": data,
        "timestamp": "2024-09-18T13:15:00Z"
    }
```

## Running and Deploying Your Server

### 1. Development Mode

For development and testing, use the MCP Inspector:[1]

```bash
# Run with inspector for testing
uv run mcp dev server.py

# Add dependencies for development
uv run mcp dev server.py --with pandas --with numpy

# Mount local code for development
uv run mcp dev server.py --with-editable .
```

### 2. Claude Desktop Integration

Install your server in Claude Desktop:[1]

```bash
# Basic installation
uv run mcp install server.py

# Custom name
uv run mcp install server.py --name "My Analytics Server"

# Environment variables
uv run mcp install server.py -v API_KEY=abc123 -v DB_URL=postgres://...
uv run mcp install server.py -f .env
```

### 3. Production Deployment Options

#### STDIO Transport (Default)
```python
if __name__ == "__main__":
    # Default STDIO transport for local clients
    mcp.run()
```

#### Streamable HTTP Transport for Remote Access
```python
if __name__ == "__main__":
    # HTTP transport for remote clients
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
```

#### Advanced ASGI Integration
```python
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware.cors import CORSMiddleware

# Create multiple MCP servers
api_server = FastMCP("API Server", stateless_http=True)
analytics_server = FastMCP("Analytics Server", stateless_http=True)

@api_server.tool()
def api_endpoint(data: str) -> str:
    return f"API processed: {data}"

@analytics_server.tool()
def analyze_data(dataset: list[float]) -> dict:
    return {
        "count": len(dataset),
        "sum": sum(dataset),
        "average": sum(dataset) / len(dataset) if dataset else 0,
        "max": max(dataset) if dataset else None,
        "min": min(dataset) if dataset else None
    }

# Create ASGI application with CORS support
app = Starlette(
    routes=[
        Mount("/api", api_server.streamable_http_app()),
        Mount("/analytics", analytics_server.streamable_http_app()),
    ]
)

# Add CORS middleware for browser clients
app = CORSMiddleware(
    app,
    allow_origins=["*"],  # Configure appropriately for production
    allow_methods=["GET", "POST", "DELETE"],
    expose_headers=["Mcp-Session-Id"],  # Required for MCP clients
)
```

### 4. Production Configuration Example

```python
import os
import logging
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("MCP_LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Production server configuration
mcp = FastMCP(
    name="Production MCP Server",
    instructions="""
    This server provides secure access to business data and operations.
    Available tools include data analysis, report generation, and system monitoring.
    All operations are logged and require appropriate authentication.
    """,
    # Production settings
    debug=os.getenv("MCP_DEBUG", "false").lower() == "true",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8080")),
)

# Production-ready tools with comprehensive error handling
@mcp.tool()
async def generate_report(
    report_type: str, 
    date_range: str,
    ctx: Context
) -> dict:
    """Generate business reports with comprehensive error handling."""
    
    try:
        await ctx.info(f"Generating {report_type} report for {date_range}")
        
        # Validate inputs
        valid_types = ["sales", "financial", "operational"]
        if report_type not in valid_types:
            raise ValueError(f"Invalid report type. Must be one of: {valid_types}")
        
        # Generate report (mock implementation)
        report_data = {
            "report_type": report_type,
            "date_range": date_range,
            "generated_at": "2024-09-18T13:15:00Z",
            "status": "completed",
            "data": {"placeholder": "Report data would go here"}
        }
        
        await ctx.info("Report generation completed successfully")
        return report_data
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        await ctx.error(f"Unexpected error generating report: {str(e)}")
        raise RuntimeError(f"Report generation failed: {str(e)}")

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    
    if transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=os.getenv("MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("MCP_PORT", "8080"))
        )
    else:
        mcp.run()  # Default STDIO transport
```

## Advanced Patterns and Best Practices

### 1. Error Handling and Validation

```python
from enum import Enum
from pydantic import BaseModel, Field, validator

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingRequest(BaseModel):
    data: str = Field(..., min_length=1, max_length=10000)
    priority: int = Field(1, ge=1, le=5)
    callback_url: str | None = Field(None, regex=r'^https?://')

@mcp.tool()
async def process_data_with_validation(
    request: ProcessingRequest,
    ctx: Context
) -> dict:
    """Process data with comprehensive validation."""
    
    try:
        await ctx.info(f"Processing data with priority {request.priority}")
        
        # Simulate processing
        result = {
            "request_id": ctx.request_id,
            "status": ProcessingStatus.COMPLETED,
            "processed_data": f"Processed: {request.data[:50]}...",
            "priority": request.priority,
        }
        
        if request.callback_url:
            # In production, make actual HTTP callback
            result["callback_sent"] = True
            
        return result
        
    except Exception as e:
        await ctx.error(f"Processing failed: {str(e)}")
        return {
            "request_id": ctx.request_id,
            "status": ProcessingStatus.FAILED,
            "error": str(e)
        }
```

### 2. Testing Your MCP Server

The official MCP Python SDK includes testing utilities:[2]

```python
import pytest
from mcp.types import TextContent
from your_server import mcp  # Your FastMCP instance

@pytest.mark.asyncio
async def test_add_numbers():
    """Test the add_numbers tool."""
    from mcp.server.fastmcp import Client
    
    async with Client(mcp) as client:
        result = await client.call_tool(
            "add_numbers",
            arguments={"a": 1, "b": 2}
        )
        
        # FastMCP client provides structured results
        assert result.data == 3
        assert result.structured_content == {"result": 3}
        
        # Also check content blocks
        assert result.content and isinstance(result.content[0], TextContent)
        assert result.content[0].text == "3"

@pytest.mark.asyncio  
async def test_weather_tool():
    """Test structured output tool."""
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_detailed_weather",
            arguments={"city": "London"}
        )
        
        # Verify structured data
        assert result.data["location"] == "London"
        assert "temperature" in result.data
        assert "humidity" in result.data
```

This comprehensive guide provides Python developers with everything needed to implement production-ready MCP servers using the official FastMCP framework from the Model Context Protocol Python SDK. The framework handles all protocol complexities while enabling developers to focus on writing clean, typed Python functions that integrate seamlessly with AI applications.[1]

[1](https://github.com/modelcontextprotocol/python-sdk/releases)
[2](https://github.com/modelcontextprotocol/python-sdk/issues/1252)
[3](https://github.com/modelcontextprotocol)
[4](https://github.com/modelcontextprotocol/python-sdk/issues/141)
[5](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/711)
[6](https://github.com/modelcontextprotocol/python-sdk/issues/923)
[7](https://github.com/modelcontextprotocol/swift-sdk)
[8](https://github.com/modelcontextprotocol/python-sdk/issues/1219)
[9](https://github.com/modelcontextprotocol/python-sdk/issues/1063)
[10](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1309)
[11](https://github.com/modelcontextprotocol/python-sdk/milestone/10)
[12](https://github.com/modelcontextprotocol/python-sdk/pull/1244)
[13](https://github.com/modelcontextprotocol/python-sdk)
[14](https://github.com/modelcontextprotocol/python-sdk/issues/1068)
[15](https://github.com/modelcontextprotocol/python-sdk/issues/951)
[16](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/600)
[17](https://github.com/modelcontextprotocol/python-sdk/security)
[18](https://github.com/modelcontextprotocol/python-sdk/issues/750)
[19](https://github.com/modelcontextprotocol/inspector/issues/763)
[20](https://github.com/modelcontextprotocol/python-sdk/labels/question)