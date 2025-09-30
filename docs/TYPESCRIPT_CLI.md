# TypeScript CLI Documentation

This document provides comprehensive guidance for using CodeFlow's CLI tool with TypeScript projects, including command-line options, configuration examples, and best practices.

## Quick Start

### Basic TypeScript Analysis

```bash
# Analyze a TypeScript project
python -m code_flow_graph.cli.code_flow_graph /path/to/typescript/project --language typescript --output analysis.json

# Query without re-analysis (if vector store exists)
python -m code_flow_graph.cli.code_flow_graph /path/to/typescript/project --language typescript --no-analyze --query "user authentication"
```

### Prerequisites Check

Before running TypeScript analysis, verify your environment:

```bash
# Check Node.js installation
node --version

# Check TypeScript installation
tsc --version

# Verify project structure
ls /path/to/typescript/project/
# Should show: src/, tsconfig.json, package.json
```

## Command Line Options

### Language Selection

```bash
# Specify TypeScript language (required for .ts/.tsx files)
--language typescript
```

**Note**: The `--language` flag is required when analyzing TypeScript projects to ensure proper AST parsing and metadata extraction.

### Output Options

```bash
# Generate JSON analysis report
--output typescript_analysis.json

# Generate Mermaid diagram for query results
--query "component dependencies" --mermaid

# Generate LLM-optimized Mermaid (minimal styling, token-efficient)
--query "service interactions" --llm-optimized
```

### Analysis Control

```bash
# Skip analysis and query existing vector store only
--no-analyze --query "function search query"

# Force complete re-analysis even if vector store exists
# (default behavior - combines analysis + query)
python -m code_flow_graph.cli.code_flow_graph /path/to/project --language typescript --query "search"
```

### Embedding Configuration

```bash
# Use different embedding model for better TypeScript understanding
--embedding-model all-MiniLM-L6-v2

# Adjust token limits for large TypeScript files
--max-tokens 512
```

## TypeScript Project Structure Support

### Automatic Configuration Detection

CodeFlow automatically detects and respects TypeScript project configuration:

#### tsconfig.json Support
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*", "tests/**/*"],
  "exclude": ["node_modules", "dist", "**/*.spec.ts"]
}
```

**Automatic Features:**
- **Path Mapping**: Resolves TypeScript path mappings (`@app/*`, `@shared/*`, etc.)
- **Project References**: Handles TypeScript project references
- **Compiler Options**: Respects `strict` mode, target ES version, module system
- **Include/Exclude Patterns**: Uses TypeScript's file inclusion rules

### Supported File Types

| Extension | Description | Analysis Support |
|-----------|-------------|------------------|
| `.ts` | TypeScript files | Full analysis with type information |
| `.tsx` | TypeScript JSX | Component and type analysis |
| `.d.ts` | TypeScript declarations | Interface and type extraction |
| `.js` | JavaScript files | Basic analysis (when mixed in TS project) |

## Framework-Specific CLI Usage

### Angular Projects

```bash
# Analyze Angular application structure
python -m code_flow_graph.cli.code_flow_graph /path/to/angular-app --language typescript --query "component tree"

# Find Angular services and their dependencies
python -m code_flow_graph.cli.code_flow_graph /path/to/angular-app --language typescript --query "injectable services"

# Generate component interaction diagram
python -m code_flow_graph.cli.code_flow_graph /path/to/angular-app --language typescript --query "user management components" --mermaid
```

**Angular-Specific Queries:**
- `"@Component decorators"` - Find all Angular components
- `"@Injectable services"` - Locate service classes
- `"lifecycle hooks"` - Identify ngOnInit, ngOnDestroy usage
- `"template-driven forms"` - Find form components
- `"routing configuration"` - Locate route definitions

### NestJS Projects

```bash
# Analyze NestJS backend structure
python -m code_flow_graph.cli.code_flow_graph /path/to/nestjs-app --language typescript --query "controller endpoints"

# Find service layer dependencies
python -m code_flow_graph.cli.code_flow_graph /path/to/nestjs-app --language typescript --query "service dependencies"

# Generate API endpoint diagram
python -m code_flow_graph.cli.code_flow_graph /path/to/nestjs-app --language typescript --query "REST endpoints" --mermaid
```

**NestJS-Specific Queries:**
- `"@Controller classes"` - Find all controllers
- `"@Injectable services"` - Locate injectable services
- `"middleware functions"` - Find Express middleware
- `"guard classes"` - Identify route guards
- `"interceptor methods"` - Locate interceptor implementations

### React TypeScript Projects

```bash
# Analyze React component structure
python -m code_flow_graph.cli.code_flow_graph /path/to/react-ts-app --language typescript --query "component hierarchy"

# Find custom hooks
python -m code_flow_graph.cli.code_flow_graph /path/to/react-ts-app --language typescript --query "custom hooks"

# Generate component dependency graph
python -m code_flow_graph.cli.code_flow_graph /path/to/react-ts-app --language typescript --query "state management" --mermaid
```

**React-Specific Queries:**
- `"functional components"` - Find function-based components
- `"class components"` - Locate class-based components
- `"custom hooks"` - Identify custom hook definitions
- `"context providers"` - Find React context usage
- `"prop interfaces"` - Locate TypeScript interfaces for props

### Express TypeScript Projects

```bash
# Analyze Express application structure
python -m code_flow_graph.cli.code_flow_graph /path/to/express-ts-app --language typescript --query "route handlers"

# Find middleware functions
python -m code_flow_graph.cli.code_flow_graph /path/to/express-ts-app --language typescript --query "middleware chain"

# Generate API structure diagram
python -m code_flow_graph.cli.code_flow_graph /path/to/express-ts-app --language typescript --query "API endpoints" --mermaid
```

**Express-Specific Queries:**
- `"route definitions"` - Find Express route handlers
- `"middleware functions"` - Locate middleware implementations
- `"error handlers"` - Identify error handling middleware
- `"request validation"` - Find input validation logic
- `"authentication middleware"` - Locate auth-related middleware

## Configuration Examples

### Basic TypeScript Project

**Directory Structure:**
```
my-ts-project/
├── src/
│   ├── controllers/
│   │   └── user.controller.ts
│   ├── services/
│   │   └── user.service.ts
│   ├── models/
│   │   └── user.model.ts
│   └── index.ts
├── tsconfig.json
└── package.json
```

**Analysis Commands:**
```bash
# Full project analysis
python -m code_flow_graph.cli.code_flow_graph /path/to/my-ts-project --language typescript --output project-analysis.json

# Query specific functionality
python -m code_flow_graph.cli.code_flow_graph /path/to/my-ts-project --language typescript --query "user management functions"

# Generate architecture diagram
python -m code_flow_graph.cli.code_flow_graph /path/to/my-ts-project --language typescript --query "service layer" --mermaid
```

### Monorepo with Multiple TypeScript Projects

**Directory Structure:**
```
monorepo/
├── packages/
│   ├── api/
│   │   ├── src/
│   │   └── tsconfig.json
│   ├── admin/
│   │   ├── src/
│   │   └── tsconfig.json
│   └── shared/
│       ├── src/
│       └── tsconfig.json
└── tsconfig.base.json
```

**Analysis Commands:**
```bash
# Analyze entire monorepo
python -m code_flow_graph.cli.code_flow_graph /path/to/monorepo --language typescript --query "cross-package dependencies"

# Analyze specific package
python -m code_flow_graph.cli.code_flow_graph /path/to/monorepo/packages/api --language typescript --query "API controllers"
```

### Custom Configuration File

Create a configuration file for reusable settings:

**analysis-config.json:**
```json
{
  "language": "typescript",
  "embedding_model": "all-MiniLM-L6-v2",
  "max_tokens": 512,
  "output_format": "json",
  "enable_mermaid": true
}
```

## Performance Optimization

### Large Codebases

For projects with many TypeScript files:

```bash
# Use higher token limit for better context
--max-tokens 512

# Use more efficient embedding model
--embedding-model paraphrase-MiniLM-L3-v2

# Query existing analysis to avoid re-processing
--no-analyze --query "search query"
```

### Memory Management

```bash
# Reduce batch size for large projects (adjust based on available memory)
# This is handled automatically but can be influenced by:
--max-tokens 256  # Smaller chunks use less memory
```

### Incremental Analysis

```bash
# Only query existing analysis (no re-processing)
python -m code_flow_graph.cli.code_flow_graph /path/to/project --language typescript --no-analyze --query "function search"

# Force complete re-analysis (for updated codebases)
python -m code_flow_graph.cli.code_flow_graph /path/to/project --language typescript --query "updated search"
```

## Best Practices

### Project Setup

1. **Organize Code Structure**: Use consistent directory patterns (`src/controllers/`, `src/services/`, etc.)
2. **Configure tsconfig.json**: Set appropriate `include`/`exclude` patterns and compiler options
3. **Use Path Mapping**: Configure `@app/*`, `@shared/*` aliases for better dependency tracking
4. **Add JSDoc Comments**: Include `@param`, `@returns`, and `@description` for better analysis

### Analysis Workflow

1. **Initial Analysis**: Run full analysis to build vector store
2. **Iterative Querying**: Use `--no-analyze` flag for subsequent queries
3. **Regular Updates**: Re-run analysis when codebase structure changes significantly
4. **Documentation**: Keep analysis results for project documentation

### Query Strategies

1. **Start Specific**: Begin with targeted queries, then broaden scope
2. **Use Framework Terms**: Include Angular/NestJS/React terminology in queries
3. **Combine with Filters**: Use multiple queries to narrow down results
4. **Generate Visualizations**: Use `--mermaid` for complex dependency understanding

## Troubleshooting

See the [TypeScript Troubleshooting Guide](TYPESCRIPT_TROUBLESHOOTING.md) for detailed solutions to common issues.

## Examples Summary

| Framework | Analysis Command | Key Queries |
|-----------|------------------|-------------|
| Angular | `--language typescript --query "components"` | `"@Component"`, `"services"`, `"routing"` |
| NestJS | `--language typescript --query "controllers"` | `"@Controller"`, `"endpoints"`, `"middleware"` |
| React TS | `--language typescript --query "hooks"` | `"components"`, `"state management"`, `"props"` |
| Express TS | `--language typescript --query "routes"` | `"handlers"`, `"middleware"`, `"validation"` |

This CLI documentation provides everything needed to effectively analyze TypeScript projects with CodeFlow, from basic setup to advanced framework-specific analysis techniques.