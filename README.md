# CodeFlowGraph: Cognitive Load Optimized Code Analysis Tool

## Overview

CodeFlowGraph is a Python-based code analysis tool that builds call graphs and identifies entry points in codebases, with a strong focus on minimizing cognitive load for developers. This tool is designed to help understand complex code structures by providing a clear, structured view of function and class relationships.

The implementation follows the Cognitive Load Optimization principles, ensuring that the code is as easy to understand and maintain as possible while still providing powerful analysis capabilities.

## Features

-   **Call Graph Generation** - Visualizes function and class call relationships
-   **Entry Point Detection** - Identifies potential entry points using multiple strategies
-   **Vector Store Integration** - Uses ChromaDB with semantic search capabilities, persisting analysis results for later queries.
-   **Language Support** - Currently supports Python and TypeScript (with Python as the default)
-   **Modular Architecture** - Separates concerns between AST extraction, graph building, and vector storage

## Requirements

Before running CodeFlowGraph, make sure you have the following dependencies installed:

```txt
chromadb==1.1.0
sentence-transformers==5.1.0
```

You can install all required dependencies using:

```bash
pip install -r requirements.txt
```

## Getting Started

### Installation

To install CodeFlowGraph, clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/codeflowgraph.git
cd codeflowgraph
pip install -r requirements.txt
```

### Basic Usage

#### 1. Analyze a Codebase and Generate a Report

Run the following command in your codebase directory or specify a target directory:

```bash
python -m code_flow_graph.cli.code_flow_graph [YOUR_CODE_DIRECTORY] --language python --output my_analysis_report.json
```

-   Replace `[YOUR_CODE_DIRECTORY]` with the path to your project. If omitted, the current directory (`.`) will be used.
-   This will create a `my_analysis_report.json` file and, importantly, a `code_vectors_chroma/` directory within `[YOUR_CODE_DIRECTORY]` (or current directory) containing the persistent vector store.

#### 2. Querying the Codebase (Analysis + Query)

If you want to analyze a directory and immediately run a semantic query:

```bash
python -m code_flow_graph.cli.code_flow_graph [YOUR_CODE_DIRECTORY] --language python --query "functions that handle user authentication"
```

This will perform the full analysis, populate/update the vector store, and then execute the query, displaying the top semantic search results.

#### 3. Querying an Existing Analysis (Query Only)

Once you have analyzed a codebase and its vector store has been populated (i.e., the `code_vectors_chroma/` directory exists in the target directory), you can query it directly without re-running the full analysis pipeline. This is much faster for subsequent queries.

```bash
python -m code_flow_graph.cli.code_flow_graph [YOUR_CODE_DIRECTORY] --no-analyze --query "functions related to data serialization"
```

-   **`[YOUR_CODE_DIRECTORY]`**: This *must* be the same directory that was previously analyzed, as it tells the tool where to find the `code_vectors_chroma/` persistence directory.
-   **`--no-analyze`**: This flag instructs the tool to skip the AST extraction and call graph building steps, and instead load the existing vector store for querying. It must be used with `--query`.

### Command Line Arguments

The tool accepts the following command line arguments:

-   `<directory>`: (Positional, optional) The path to the codebase directory (default: current directory `.` ). This directory is also used as the base for the persistent ChromaDB vector store (`<directory>/code_vectors_chroma/`).
-   `--language`: The programming language to analyze (choices: `python` or `typescript`, default: `python`)
-   `--output`: The file path for the generated analysis report (default: `code_analysis_report.json`). *Only used when performing a full analysis (i.e., without `--no-analyze` and `--query` flags).*
-   `--query <QUESTION>`: Run a semantic query against the analyzed codebase or a previously stored vector store.
-   `--no-analyze`: (Flag) Do not perform analysis. Assume the vector store in the specified `directory` is already populated from a previous run. This flag **must** be used in conjunction with `--query`.

### Example Report Output

The generated `code_analysis_report.json` report will contain:

1.  A `summary` of the codebase (total functions, edges, modules, etc.)
2.  A list of identified `entry_points` with detailed metadata.
3.  A `classes_summary`.
4.  A `call_graph` object with:
    -   `functions`: A dictionary where keys are fully qualified names (FQN) and values are detailed metadata for each function (name, file path, line numbers, parameters, docstring, connectivity, etc.).
    -   `edges`: A list of all identified function calls, specifying caller, callee, file, line number, and confidence.

This JSON format can be used for further analysis or custom visualization.

## Architecture

The tool is structured into three main components:

1.  **AST Extractor** (`core/ast_extractor.py`)
    -   Parses source code into abstract syntax trees (ASTs).
    -   Extracts functions and classes with metadata (e.g., parameters, return types, docstrings).
    -   Handles Python and provides a placeholder for TypeScript.
2.  **Call Graph Builder** (`core/call_graph_builder.py`)
    -   Constructs a graph representation of function calls based on AST analysis.
    -   Identifies potential entry points in the codebase using various heuristics.
    -   Provides structured data (`FunctionNode`, `CallEdge`) for graph representation.
3.  **Vector Store** (`core/vector_store.py`)
    -   Integrates with [ChromaDB](https://www.trychroma.com/) to store function and edge metadata as embeddings.
    -   Enables semantic search capabilities over the codebase's functions.
    -   Persists data to disk (`<directory>/code_vectors_chroma/`) for efficient re-loading and querying.

## Cognitive Load Optimization

This tool was specifically designed with the following principles in mind:

### 1. Mental Model Simplicity
-   Uses predictable patterns and clear interfaces.
-   Avoids complex or obscure language features.
-   Prioritizes explicit code over implicit behavior.

### 2. Conditional Logic Optimization
-   Uses early returns and avoids deep nesting.
-   Makes execution paths easy to follow.

### 3. Information Hiding and Locality
-   Groups related code together.
-   Uses deep modules with simple interfaces.
-   Applies DRY principles where they reduce cognitive load.

### 4. Minimizing Required Background Knowledge
-   Uses self-describing data (e.g., strings instead of numeric codes).
-   Follows common patterns and conventions.
-   Avoids arbitrary mappings.

### 5. Strategic Abstraction
-   Introduces abstractions only when they reduce complexity.
-   Optimizes for API usability over implementation simplicity.

### 6. Linear Reading Support
-   Structures code for top-to-bottom reading.
-   Uses guard clauses and fail-fast approaches.
-   Keeps main logic paths clear.

## Example Use Cases

### 1. Codebase Understanding
-   Quickly identify all entry points in a large Python codebase.
-   Visualize how functions interact with each other (external visualization tools would be needed for the JSON output).

### 2. Documentation Generation
-   Use the extracted function metadata to generate documentation.
-   Identify functions that lack docstrings.

### 3. Code Navigation
-   Semantically query the vector store to find functions related to a specific task (e.g., "functions that handle user authentication" or "code for database transactions").
-   Jump to relevant code locations with rich metadata.

### 4. Code Quality Analysis
-   Identify functions with high incoming/outgoing connectivity (potential high coupling or complexity).
-   Find functions that might be good candidates for refactoring due to being entry points or having many connections.

## Contributing

We welcome contributions to improve and expand CodeFlowGraph. Please follow these steps:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Ensure all tests pass (if any are added/present).
5.  Submit a pull request with a clear description of your changes.

### Development

To install the project in editable mode for development:

```bash
pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

-   Add more robust support for TypeScript parsing (currently a placeholder).
-   Implement visualization tools for the call graph (e.g., D3.js, Graphviz export).
-   Enhance entry point detection with more sophisticated heuristics (e.g., framework-specific patterns).
-   Add capabilities to infer more details (e.g., external library calls, data flow).
-   Integrate with popular IDEs for seamless code navigation.

## Acknowledgments

This project was inspired by the Cognitive Load Optimization principles and the importance of human-readable, maintainable code. Special thanks to the developers of:
-   [ChromaDB](https://github.com/chroma-core/chroma)
-   [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
-   [Python AST module](https://docs.python.org/3/library/ast.html)

## Contact

For questions, suggestions, or to report issues, please open an issue on the GitHub repository.
