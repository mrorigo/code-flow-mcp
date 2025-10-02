# TypeScript Troubleshooting Guide

This guide provides solutions to common issues encountered when using CodeFlow with TypeScript projects, including configuration issues, framework detection, and performance optimization.

## TypeScript Analysis

### No External Dependencies Required

CodeFlow uses sophisticated regex-based parsing for TypeScript analysis. No Node.js or TypeScript compiler installation is required.

**Benefits:**
- Works out-of-the-box with any TypeScript codebase
- No version compatibility issues  
- Consistent performance across all environments
- Comprehensive framework support (Angular, React, NestJS, Express)

**Supported Features:**
- Type annotations and generics
- Classes, interfaces, and enums
- Decorators and access modifiers
- Import/export analysis
- Framework-specific patterns

### Analysis Performance

**Optimized Regex Parsing:**
CodeFlow uses pre-compiled regex patterns for maximum performance:

- **Pattern Compilation**: Regex patterns are compiled once at startup
- **Parallel Processing**: Multiple files processed simultaneously  
- **Framework Detection**: Built-in patterns for Angular, React, NestJS, Express
- **Type Analysis**: Comprehensive support for TypeScript type system

**Performance Tips:**
1. Use `.gitignore` to exclude unnecessary files
2. Process large projects in batches
3. Use incremental analysis (`--no-analyze`) for repeated queries

## Configuration Issues

### tsconfig.json Not Found

**Error Message:**
```
Info: Loaded TypeScript configuration from /path/to/project/tsconfig.json
# (This message is missing, indicating no tsconfig.json found)
```

**Symptoms:**
- TypeScript project integration features unavailable
- Path mappings not resolved
- Compiler options not applied

**Solutions:**

1. **Create basic tsconfig.json:**
   ```json
   {
     "compilerOptions": {
       "target": "ES2020",
       "module": "commonjs",
       "strict": true,
       "esModuleInterop": true,
       "skipLibCheck": true,
       "forceConsistentCasingInFileNames": true,
       "declaration": true,
       "outDir": "./dist",
       "rootDir": "./src"
     },
     "include": ["src/**/*"],
     "exclude": ["node_modules", "dist", "**/*.spec.ts", "**/*.test.ts"]
   }
   ```

2. **Check file location:**
   - tsconfig.json should be in project root or parent directories
   - CodeFlow searches from the analyzed directory upward

3. **Validate JSON syntax:**
   ```bash
   # Test JSON validity
   cat tsconfig.json | python -m json.tool > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
   ```

### Invalid tsconfig.json

**Error Message:**
```
Warning: Invalid JSON in tsconfig.json /path/to/project/tsconfig.json: Line X, column Y: Invalid syntax
```

**Solutions:**

1. **Fix JSON syntax errors:**
   - Missing commas after properties
   - Unclosed braces or brackets
   - Invalid escape characters
   - Trailing commas (invalid in JSON)

2. **Use JSON validator:**
   ```bash
   # Online validators:
   # - https://jsonlint.com/
   # - https://jsonformatter.curiousconcept.com/
   ```

3. **Simplify configuration:**
   ```bash
   # Start with minimal config and add complexity gradually
   cp /dev/null tsconfig.json
   echo '{"compilerOptions": {"target": "ES2020"}}' > tsconfig.json
   ```

### Path Mapping Issues

**Error Message:**
```
Warning: Could not resolve import path: @app/services/user
```

**Symptoms:**
- Import statements not properly resolved
- Incorrect dependency relationships
- Missing function connections in call graphs

**Solutions:**

1. **Check baseUrl configuration:**
   ```json
   {
     "compilerOptions": {
       "baseUrl": "./src",
       "paths": {
         "@app/*": ["./*"],
         "@shared/*": ["../shared/*"]
       }
     }
   }
   ```

2. **Verify path mappings:**
   ```typescript
   // If you have: "@app/services/user"
   // Ensure baseUrl points to the correct directory
   // And the actual file exists at: src/services/user.ts
   ```

3. **Test path resolution:**
   ```bash
   # Check tsconfig.json path mappings
   cat tsconfig.json | grep -A 10 "paths"
   ```

## Framework Detection Issues

### Angular Decorators Not Detected

**Error Message:**
```
Framework detection: angular not found in /path/to/angular-app
```

**Symptoms:**
- Components not identified as Angular components
- Missing Angular-specific metadata
- Incorrect framework classification

**Solutions:**

1. **Check Angular imports:**
   ```typescript
   // Ensure proper Angular imports
   import { Component, OnInit } from '@angular/core';

   @Component({
     selector: 'app-user',
     template: '...'
   })
   export class UserComponent implements OnInit {
     // Component implementation
   }
   ```

2. **Verify decorator syntax:**
   ```typescript
   // Correct syntax
   @Component({
     selector: 'app-example',
     templateUrl: './example.component.html'
   })

   // Incorrect syntax (missing decorator arguments)
   @Component
   export class ExampleComponent {}
   ```

3. **Check for Angular CLI setup:**
   ```bash
   # Verify Angular CLI installation
   ng --version

   # Check if project was created with Angular CLI
   ls -la | grep angular.json
   ```

### NestJS Controllers Not Detected

**Error Message:**
```
Framework detection: nestjs not found in /path/to/nestjs-app
```

**Symptoms:**
- Controllers not identified properly
- Missing HTTP method information
- Incorrect service dependencies

**Solutions:**

1. **Check NestJS imports and decorators:**
   ```typescript
   // Ensure proper NestJS imports
   import { Controller, Get, Post } from '@nestjs/common';

   @Controller('users')
   export class UserController {
     @Get()
     findAll() {
       return 'This action returns all users';
     }
   }
   ```

2. **Verify NestJS module setup:**
   ```typescript
   // Check main.ts
   import { NestFactory } from '@nestjs/core';
   import { AppModule } from './app.module';

   async function bootstrap() {
     const app = await NestFactory.create(AppModule);
     await app.listen(3000);
   }
   bootstrap();
   ```

3. **Check package.json dependencies:**
   ```json
   {
     "dependencies": {
       "@nestjs/common": "^10.0.0",
       "@nestjs/core": "^10.0.0",
       "@nestjs/platform-express": "^10.0.0"
     }
   }
   ```

### React TypeScript Components Not Detected

**Error Message:**
```
Framework detection: react not found in /path/to/react-app
```

**Symptoms:**
- Components not identified as React components
- Missing hook usage information
- Incorrect JSX handling

**Solutions:**

1. **Check React imports:**
   ```typescript
   // Ensure React is imported
   import React, { useState, useEffect } from 'react';

   const UserComponent: React.FC = () => {
     const [users, setUsers] = useState([]);
     // Component implementation
   };
   ```

2. **Verify TypeScript React setup:**
   ```bash
   # Install React TypeScript definitions
   npm install --save-dev @types/react @types/react-dom

   # Ensure .tsx files are included in tsconfig.json
   # "include": ["src/**/*"]
   ```

3. **Check for React-specific patterns:**
   ```typescript
   // Look for these patterns:
   - React.FC, React.Component
   - useState, useEffect, useContext
   - JSX elements (<div>, <Component />)
   - TypeScript interfaces for props
   ```

## Analysis Issues

### Parsing Errors

**Error Message:**
```
Warning: Error processing /path/to/file.ts: SyntaxError: invalid syntax
```

**Symptoms:**
- Individual files fail to parse
- Missing elements from specific files
- Inconsistent analysis results

**Solutions:**

1. **Check TypeScript syntax:**
   ```bash
   # Verify file syntax is valid TypeScript
   # CodeFlow will report parsing errors for invalid syntax

   # Or use an online TypeScript validator
   ```

2. **Common syntax issues:**
   - Missing semicolons in strict mode
   - Incorrect type annotations
   - Malformed generic types
   - Invalid decorator syntax

3. **Simplify complex files:**
   ```typescript
   // If a file has complex syntax, try simplifying:
   // Instead of: const x: Map<string, Array<{id: number}>> = new Map()
   // Use: const x = new Map() // Let TypeScript infer types
   ```

### Memory Issues

### Large File Processing

**Optimized for Performance:**
CodeFlow's regex-based parsing handles large files efficiently:

**Performance Features:**
- **Parallel Processing**: Multiple files processed simultaneously
- **Memory Efficient**: Streaming processing for large files  
- **Batch Operations**: Optimal batching for maximum throughput

**Best Practices:**
1. **File Organization:**
   ```typescript
   // Split large components into smaller, focused files
   // Move complex logic to separate service files
   // Extract interfaces to separate .d.ts files
   ```

2. **Use Incremental Analysis:**
   ```bash
   # For repeated queries on large projects
   python -m code_flow_graph.cli.code_flow_graph . --no-analyze --query "your query"
   ```

### Performance Issues

**Error Message:**
```
Processing /path/to/project: Slow analysis performance
```

**Symptoms:**
- Analysis takes very long to complete
- High CPU/memory usage
- Timeout errors

**Solutions:**

1. **Use incremental analysis:**
   ```bash
   # Only analyze when needed
   python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "search"

   # Force re-analysis only when code changes
   python -m code_flow_graph.cli.code_flow_graph . --language typescript --query "updated search"
   ```

2. **Optimize file patterns:**
   ```json
   {
     "exclude": [
       "node_modules",
       "dist",
       "**/*.spec.ts",
       "**/*.test.ts",
       "**/*.d.ts"
     ]
   }
   ```

3. **Use smaller batch sizes:**
   ```python
   # Process files in smaller batches
   batch_size = 10
   for i in range(0, len(files), batch_size):
       batch = files[i:i+batch_size]
       # Process batch
   ```

## Framework-Specific Issues

### Angular Module Resolution

**Error Message:**
```
Warning: Cannot resolve Angular module imports in /path/to/app.module.ts
```

**Solutions:**

1. **Check Angular module structure:**
   ```typescript
   // app.module.ts
   import { NgModule } from '@angular/core';
   import { BrowserModule } from '@angular/platform-browser';

   @NgModule({
     imports: [BrowserModule],
     declarations: [AppComponent],
     bootstrap: [AppComponent]
   })
   export class AppModule {}
   ```

2. **Verify Angular CLI project:**
   ```bash
   # Check for Angular project markers
   ls -la angular.json
   ls -la src/main.ts
   ```

### NestJS Dependency Injection

**Error Message:**
```
Warning: Cannot detect NestJS service dependencies
```

**Solutions:**

1. **Check service registration:**
   ```typescript
   // app.module.ts
   import { Module } from '@nestjs/common';
   import { UserService } from './user.service';
   import { UserController } from './user.controller';

   @Module({
     controllers: [UserController],
     providers: [UserService], // Ensure services are in providers
     exports: [UserService]
   })
   export class AppModule {}
   ```

2. **Verify constructor injection:**
   ```typescript
   // user.controller.ts
   import { Controller, Get } from '@nestjs/common';
   import { UserService } from './user.service';

   @Controller('users')
   export class UserController {
     constructor(private readonly userService: UserService) {} // Proper injection

     @Get()
     findAll() {
       return this.userService.findAll();
     }
   }
   ```

### React Hook Dependencies

**Error Message:**
```
Warning: Cannot extract React hook dependencies
```

**Solutions:**

1. **Check hook usage patterns:**
   ```typescript
   import { useState, useEffect } from 'react';

   const UserComponent = () => {
     const [users, setUsers] = useState([]);
     const [loading, setLoading] = useState(true);

     useEffect(() => {
       fetchUsers().then(users => {
         setUsers(users);
         setLoading(false);
       });
     }, []); // Empty dependency array
   };
   ```

2. **Verify TypeScript React setup:**
   ```json
   {
     "compilerOptions": {
       "jsx": "react-jsx", // or "react"
       "types": ["react", "react-dom"]
     }
   }
   ```

## Environment-Specific Issues

### Windows Path Issues

**Error Message:**
```
Warning: Cannot find TypeScript files due to path separator issues
```

**Solutions:**

1. **Use forward slashes in paths:**
   ```python
   # In Python scripts
   file_path = "src/components/UserComponent.tsx"
   # Instead of: "src\components\UserComponent.tsx"
   ```

2. **Check file permissions:**
   ```cmd
   # Ensure read permissions on TypeScript files
   dir /q src\*.ts
   ```

### Docker Environment

**CodeFlow in Docker:**
CodeFlow works seamlessly in Docker containers with no additional setup required.

**Simple Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Install CodeFlow
COPY . /app
WORKDIR /app
RUN pip install -e .

# No Node.js or TypeScript installation needed
# CodeFlow uses regex-based parsing
```

**Multi-stage builds:**
```dockerfile
# Analysis stage
FROM python:3.9-slim as analyzer
COPY . /app
WORKDIR /app
RUN pip install -e .
RUN python -m code_flow_graph.cli.code_flow_graph . --output analysis.json

   # Runtime stage
   FROM node:18-alpine
   COPY --from=builder /app/dist ./dist
   ```

## Verification Commands

### Quick Diagnostics

```bash
# 1. Validate project structure
find . -name "*.ts" -o -name "*.tsx" | head -10

# 2. Check tsconfig.json (if present)
cat tsconfig.json | python -m json.tool

# 3. Test CodeFlow analysis
python -m code_flow_graph.cli.code_flow_graph . --language typescript --query "test"
```

### Framework Verification

**Angular:**
```bash
ng --version
ls -la angular.json
grep -r "@Component" src/ | wc -l
```

**NestJS:**
```bash
grep -r "@Controller" src/ | wc -l
grep -r "@Injectable" src/ | wc -l
```

**React:**
```bash
grep -r "useState\|useEffect" src/ | wc -l
grep -r "React\.FC\|React\.Component" src/ | wc -l
```

## Getting Help

### Debug Mode

Enable verbose logging for detailed diagnostics:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with detailed output
python -m code_flow_graph.cli.code_flow_graph . --language typescript --query "debug"
```

### Log Files

Check for detailed error information:

```bash
# Look for warning messages in output
python -m code_flow_graph.cli.code_flow_graph . --language typescript 2>&1 | grep -i "warning\|error"

# Check system logs for Node.js issues
# On Linux: /var/log/syslog
# On macOS: Console.app
```

### Community Resources

1. **TypeScript Official Documentation:**
   - https://www.typescriptlang.org/docs/

2. **Framework Documentation:**
   - Angular: https://angular.io/docs
   - NestJS: https://docs.nestjs.com/
   - React: https://react.dev/learn

3. **Node.js Issues:**
   - https://nodejs.org/en/docs/
   - https://github.com/nodejs/help

## Best Practices

### Project Setup

1. **Consistent Configuration:**
   ```json
   // Use consistent tsconfig.json across projects
   {
     "extends": "@tsconfig/node18/tsconfig.json",
     "compilerOptions": {
       "outDir": "./dist",
       "rootDir": "./src"
     }
   }
   ```

2. **Organized File Structure:**
   ```
   src/
   ├── components/     # Angular/React components
   ├── services/       # Business logic
   ├── models/         # TypeScript interfaces/types
   ├── controllers/    # NestJS controllers
   └── index.ts        # Main entry point
   ```

3. **Regular Validation:**
   ```bash
   # Add to CI/CD pipeline
   python -m code_flow_graph.cli.code_flow_graph . --language typescript --output analysis.json
   npm run lint
   npm test
   ```

### Performance Optimization

1. **Use appropriate compiler options:**
   ```json
   {
     "compilerOptions": {
       "skipLibCheck": true,
       "incremental": true,
       "tsBuildInfoFile": "./.tsbuildinfo"
     }
   }
   ```

2. **Exclude unnecessary files:**
   ```json
   {
     "exclude": [
       "node_modules",
       "dist",
       "**/*.spec.ts",
       "**/*.test.ts"
     ]
   }
   ```

3. **Use project references for monorepos:**
   ```json
   {
     "references": [
       { "path": "./packages/api" },
       { "path": "./packages/admin" }
     ]
   }
   ```

This troubleshooting guide covers the most common issues encountered with TypeScript support in CodeFlow. If you encounter problems not covered here, please check the official documentation or community resources for the latest solutions.