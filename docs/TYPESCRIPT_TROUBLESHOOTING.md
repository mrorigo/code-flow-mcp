# TypeScript Troubleshooting Guide

This guide provides solutions to common issues encountered when using CodeFlow with TypeScript projects, including installation problems, configuration issues, framework detection, and performance optimization.

## Installation Issues

### Node.js Not Found

**Error Message:**
```
Warning: TypeScript compiler not found for /path/to/file.ts
```

**Symptoms:**
- TypeScript analysis falls back to regex-based parsing
- Reduced accuracy in type information extraction
- Missing framework-specific metadata

**Solutions:**

1. **Install Node.js (Linux/Debian/Ubuntu):**
   ```bash
   # Using NodeSource repository (recommended)
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs

   # Verify installation
   node --version  # Should show v18.x.x or higher
   ```

2. **Install Node.js (macOS):**
   ```bash
   # Using Homebrew
   brew install node

   # Or using official installer from nodejs.org
   # Download from https://nodejs.org/

   # Verify installation
   node --version  # Should show v18.x.x or higher
   ```

3. **Install Node.js (Windows):**
   - Download the latest LTS version from [nodejs.org](https://nodejs.org/)
   - Run the installer and follow the setup wizard
   - Verify installation via Command Prompt:
     ```cmd
     node --version
     npm --version
     ```

### TypeScript Compiler Not Available

**Error Message:**
```
Warning: TypeScript compiler error for /path/to/file.ts: npx: not found
```

**Symptoms:**
- Cannot run TypeScript compiler integration
- Falls back to regex-based parsing only
- Missing advanced type system features

**Solutions:**

1. **Install TypeScript globally:**
   ```bash
   npm install -g typescript

   # Verify installation
   tsc --version  # Should show TypeScript version
   ```

2. **Check npm permissions:**
   ```bash
   # If you get permission errors, try:
   npm install -g typescript --force

   # Or fix npm permissions
   sudo chown -R $(whoami) ~/.npm
   ```

3. **Use alternative package manager:**
   ```bash
   # Using yarn
   yarn global add typescript

   # Using pnpm
   pnpm add -g typescript
   ```

### Environment Path Issues

**Error Message:**
```
Warning: Could not start CodeVectorStore... Error: Node.js not found in PATH
```

**Solutions:**

1. **Check PATH configuration:**
   ```bash
   # Check if node is in PATH
   which node
   echo $PATH

   # Add to PATH if missing
   export PATH="$PATH:/usr/local/bin:/usr/bin"
   ```

2. **Restart terminal/shell after installation:**
   - Close and reopen your terminal
   - Or source your shell configuration:
     ```bash
     source ~/.bashrc  # Linux
     source ~/.zshrc   # macOS
     ```

3. **Verify complete installation:**
   ```bash
   node --version
   npm --version
   npx --version
   ```

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
   # Use TypeScript compiler to verify paths
   npx tsc --noEmit --listFiles
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
   # Use TypeScript compiler to validate
   npx tsc --noEmit file.ts

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

**Error Message:**
```
Warning: TypeScript compiler timeout for /path/to/large-file.ts
```

**Symptoms:**
- Large files fail to process
- Timeout errors during compilation
- High memory usage

**Solutions:**

1. **Break down large files:**
   ```typescript
   // Split large components into smaller ones
   // Move complex logic to separate service files
   // Extract interfaces to separate .d.ts files
   ```

2. **Optimize TypeScript configuration:**
   ```json
   {
     "compilerOptions": {
       "skipLibCheck": true,
       "incremental": true,
       "tsBuildInfoFile": "./.tsbuildinfo"
     }
   }
   ```

3. **Process files individually:**
   ```python
   # In your analysis script:
   for file_path in large_ts_files:
       try:
           elements = extractor.extract_from_file(file_path)
           # Process immediately and free memory
           gc.collect()
       except Exception as e:
           print(f"Skipping {file_path}: {e}")
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

### macOS Security Issues

**Error Message:**
```
Warning: Cannot execute TypeScript compiler due to security restrictions
```

**Solutions:**

1. **Allow downloaded applications:**
   - System Preferences → Security & Privacy → General
   - Allow apps downloaded from "App Store and identified developers"

2. **Check Gatekeeper settings:**
   ```bash
   # Check if Node.js is blocked
   spctl --status

   # Allow specific applications
   spctl --add /usr/local/bin/node
   ```

### Docker Environment Issues

**Error Message:**
```
Warning: TypeScript compiler not available in Docker container
```

**Solutions:**

1. **Install Node.js in Dockerfile:**
   ```dockerfile
   FROM node:18-alpine

   # Install TypeScript
   RUN npm install -g typescript

   # Copy your application
   COPY . /app
   WORKDIR /app
   ```

2. **Use multi-stage builds:**
   ```dockerfile
   # Build stage
   FROM node:18-alpine as builder
   # ... build commands

   # Runtime stage
   FROM node:18-alpine
   COPY --from=builder /app/dist ./dist
   ```

## Verification Commands

### Quick Diagnostics

```bash
# 1. Check Node.js installation
node --version && npm --version

# 2. Check TypeScript installation
npx tsc --version

# 3. Validate project structure
find . -name "*.ts" -o -name "*.tsx" | head -10

# 4. Check tsconfig.json
cat tsconfig.json | python -m json.tool

# 5. Test TypeScript compilation
npx tsc --noEmit

# 6. Test CodeFlow analysis
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
   npx tsc --noEmit
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