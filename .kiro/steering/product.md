---
inclusion: always
---

# CodeFlow Product Overview

CodeFlow is a cognitive load optimized code analysis tool that generates detailed call graphs, identifies critical code elements, and provides semantic search capabilities for complex codebases. It's designed to help developers and autonomous agents understand code structure with minimal cognitive overhead.

## Core Value Proposition

- **Deep AST Analysis**: Extracts comprehensive metadata from Python and TypeScript codebases including complexity metrics, dependencies, decorators, and exception handling
- **Semantic Search**: Uses ChromaDB vector store with sentence transformers for intelligent code querying
- **Call Graph Generation**: Builds function-to-function call relationships with entry point detection
- **Multiple Interfaces**: CLI tool, MCP server for AI assistants, and unified programmatic API
- **Cognitive Load Optimization**: Designed with principles that prioritize human comprehension and clear mental models

## Key Features

- Unified interface supporting both Python and TypeScript with automatic language detection
- Persistent vector store with automatic cleanup of stale references
- Mermaid diagram generation with LLM-optimized output modes
- Real-time file watching and incremental updates via MCP server
- Rich metadata extraction including cyclomatic complexity, NLOC, decorators, and local variables
- Background maintenance processes for index accuracy

## Target Users

- Developers analyzing complex codebases
- AI assistants and autonomous agents needing code understanding
- Development teams requiring code structure visualization
- Code review and refactoring workflows