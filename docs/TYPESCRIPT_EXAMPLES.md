# TypeScript Usage Examples and Walkthroughs

This document provides comprehensive examples and step-by-step walkthroughs for using CodeFlow with various TypeScript project types, from basic applications to complex framework-based projects.

## Prerequisites

Before following these examples, ensure you have:

1. **CodeFlow installed** and working
2. **TypeScript project files** (no external TypeScript installation required)

## Example Projects Overview

The following examples demonstrate TypeScript analysis across different project types:

| Project Type | Framework | Complexity | Key Features |
|--------------|-----------|------------|--------------|
| Basic TypeScript | None | Simple | Core TypeScript features |
| Angular Application | Angular | Medium | Components, Services, Dependency Injection |
| React TypeScript | React | Medium | Hooks, Components, State Management |
| NestJS Backend | NestJS | Advanced | Controllers, Services, Middleware |
| Express API | Express.js | Medium | Route handlers, Middleware |

## 1. Basic TypeScript Project Analysis

### Project Structure

```
basic-typescript-project/
├── src/
│   ├── models/
│   │   ├── user.ts
│   │   └── product.ts
│   ├── services/
│   │   ├── user.service.ts
│   │   └── product.service.ts
│   ├── utils/
│   │   └── helpers.ts
│   └── index.ts
├── tests/
│   ├── user.service.test.ts
│   └── product.service.test.ts
├── tsconfig.json
└── package.json
```

### Step-by-Step Walkthrough

#### Step 1: Setup Project

```bash
# Create project directory
mkdir basic-typescript-project
cd basic-typescript-project

# Initialize npm project
npm init -y

# Create tsconfig.json (optional - CodeFlow will work without it)
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
EOF
```

#### Step 2: Create Sample Code

**src/models/user.ts:**
```typescript
export interface User {
    id: number;
    name: string;
    email: string;
    createdAt: Date;
    role: 'admin' | 'user' | 'guest';
}

export interface CreateUserRequest {
    name: string;
    email: string;
    role?: 'admin' | 'user' | 'guest';
}
```

**src/models/product.ts:**
```typescript
export interface Product {
    id: number;
    name: string;
    price: number;
    category: string;
    inStock: boolean;
}

export type ProductCategory = 'electronics' | 'clothing' | 'books' | 'food';
```

**src/services/user.service.ts:**
```typescript
import { User, CreateUserRequest } from '../models/user';

export class UserService {
    private users: User[] = [];

    public async createUser(request: CreateUserRequest): Promise<User> {
        // Validate request
        if (!request.name || !request.email) {
            throw new Error('Name and email are required');
        }

        // Check for duplicates
        const existingUser = this.users.find(u => u.email === request.email);
        if (existingUser) {
            throw new Error('User with this email already exists');
        }

        // Create new user
        const newUser: User = {
            id: Date.now(),
            name: request.name,
            email: request.email,
            createdAt: new Date(),
            role: request.role || 'user'
        };

        this.users.push(newUser);
        return newUser;
    }

    public async getUserById(id: number): Promise<User | null> {
        const user = this.users.find(u => u.id === id);
        return user || null;
    }

    public async getUsersByRole(role: User['role']): Promise<User[]> {
        return this.users.filter(u => u.role === role);
    }

    public async updateUser(id: number, updates: Partial<User>): Promise<User | null> {
        const userIndex = this.users.findIndex(u => u.id === id);
        if (userIndex === -1) {
            return null;
        }

        this.users[userIndex] = { ...this.users[userIndex], ...updates };
        return this.users[userIndex];
    }
}
```

**src/index.ts:**
```typescript
import { UserService } from './services/user.service';

async function main() {
    console.log('🚀 Starting TypeScript Application...');

    const userService = new UserService();

    try {
        // Create users
        const user1 = await userService.createUser({
            name: 'John Doe',
            email: 'john@example.com',
            role: 'admin'
        });

        const user2 = await userService.createUser({
            name: 'Jane Smith',
            email: 'jane@example.com',
            role: 'user'
        });

        console.log('✅ Users created successfully');

        // Get users by role
        const admins = await userService.getUsersByRole('admin');
        const users = await userService.getUsersByRole('user');

        console.log(`Found ${admins.length} admins and ${users.length} users`);

        // Update user
        const updatedUser = await userService.updateUser(user1.id, {
            name: 'John Doe Updated'
        });

        console.log('✅ User updated:', updatedUser?.name);

    } catch (error) {
        console.error('❌ Error:', error);
    }
}

// Run the application
main().catch(console.error);
```

#### Step 3: Analyze with CodeFlow

```bash
# 1. Perform complete analysis
python -m code_flow_graph.cli.code_flow_graph . --language typescript --output basic-analysis.json

# Expected output:
# 🚀 Starting analysis of typescript codebase at /path/to/basic-typescript-project
# 📖 Step 1: Extracting AST elements...
# Found X TypeScript files to analyze (after filtering .gitignore).
# 🔗 Step 2: Building call graph...
# 💾 Step 3: Populating vector store...
# 📊 Step 4: Generating analysis report...
# ✅ Analysis complete!
# 📄 Report exported to basic-analysis.json
```

#### Step 4: Query the Analysis

```bash
# 1. Find user-related functions
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "user management functions"

# 2. Find service classes
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "service classes"

# 3. Generate call graph for user operations
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "user creation and management" --mermaid

# 4. Find TypeScript interfaces
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "TypeScript interfaces and types"
```

#### Step 5: Interpret Results

**Expected Findings:**
- **Functions**: `createUser`, `getUserById`, `getUsersByRole`, `updateUser`
- **Classes**: `UserService`
- **Interfaces**: `User`, `CreateUserRequest`, `Product`
- **TypeScript Features**: Union types, optional properties, generic constraints
- **Complexity**: Low to medium cyclomatic complexity
- **Dependencies**: Internal module dependencies, no external packages

## 2. Angular Application Analysis

### Project Structure

```
angular-user-management/
├── src/
│   ├── app/
│   │   ├── models/
│   │   │   └── user.model.ts
│   │   ├── services/
│   │   │   └── user.service.ts
│   │   ├── components/
│   │   │   ├── user-list/
│   │   │   └── user-form/
│   │   └── user-management.component.ts
│   ├── index.html
│   └── main.ts
├── angular.json
├── package.json
└── tsconfig.json
```

### Step-by-Step Walkthrough

#### Step 1: Analyze Angular Project

```bash
# Navigate to Angular project
cd angular-user-management

# Perform complete analysis
python -m code_flow_graph.cli.code_flow_graph . --language typescript --output angular-analysis.json

# Expected output:
# 🚀 Starting analysis of typescript codebase at /path/to/angular-user-management
# 📖 Step 1: Extracting AST elements...
# Found X TypeScript files to analyze
# 🔗 Step 2: Building call graph...
# 💾 Step 3: Populating vector store...
# ✅ Analysis complete!
```

#### Step 2: Framework-Specific Queries

```bash
# 1. Find all Angular components
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "Angular components"

# 2. Find service dependencies
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "service injection dependencies"

# 3. Find lifecycle methods
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "ngOnInit ngOnDestroy lifecycle"

# 4. Generate component interaction diagram
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "user management component interactions" --mermaid

# 5. Find form handling
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "reactive forms template forms"
```

#### Step 3: Angular-Specific Analysis

**Expected Findings:**
- **Components**: Classes decorated with `@Component`
- **Services**: Classes decorated with `@Injectable`
- **Lifecycle Hooks**: `ngOnInit`, `ngOnDestroy`, `ngAfterViewInit`
- **Dependency Injection**: Constructor injection patterns
- **Template Integration**: Component/template relationships
- **RxJS Usage**: Observable patterns, subscription management

## 3. Advanced Framework Examples

### NestJS Backend Analysis

```bash
# Example NestJS project analysis
cd nestjs-api

# Analyze controllers and services
python -m code_flow_graph.cli.code_flow_graph . --language typescript --query "NestJS controllers and services"

# Find API endpoints
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "REST API endpoints"

# Analyze dependency injection
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "service dependencies injection"

# Generate API structure diagram
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "API architecture" --mermaid
```

**Expected NestJS Findings:**
- **Controllers**: `@Controller` decorated classes
- **Services**: `@Injectable` decorated classes
- **HTTP Methods**: `@Get`, `@Post`, `@Put`, `@Delete` decorators
- **Middleware**: Route-level and global middleware
- **Guards**: Authentication and authorization logic
- **Interceptors**: Request/response processing

### React TypeScript Analysis

```bash
# Example React TypeScript project analysis
cd react-typescript-app

# Find components and hooks
python -m code_flow_graph.cli.code_flow_graph . --language typescript --query "React components and custom hooks"

# Analyze state management
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "useState useEffect state management"

# Find prop interfaces
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "component props interfaces"

# Generate component hierarchy
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "component hierarchy tree" --mermaid
```

**Expected React Findings:**
- **Functional Components**: `React.FC`, arrow functions
- **Custom Hooks**: `use*` prefixed functions
- **State Management**: `useState`, `useReducer` patterns
- **Effects**: `useEffect`, `useLayoutEffect` usage
- **Context**: `useContext`, `createContext` patterns
- **TypeScript Integration**: Props interfaces, generic components

### Express TypeScript API

```bash
# Example Express TypeScript project analysis
cd express-api

# Find route handlers
python -m code_flow_graph.cli.code_flow_graph . --language typescript --query "Express route handlers middleware"

# Analyze API structure
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "API routes endpoints"

# Find error handling
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "error handling middleware"

# Generate API call graph
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "request flow" --mermaid
```

**Expected Express Findings:**
- **Route Handlers**: Express route functions
- **Middleware**: Request processing functions
- **Error Handling**: Error middleware patterns
- **Request Validation**: Input validation logic
- **Authentication**: Auth middleware and guards

## 4. Advanced Analysis Techniques

### Type System Analysis

```bash
# Find all TypeScript interfaces
python -m code_flow_graph.cli.code_flow_graph . --language typescript --query "interface definitions"

# Find type aliases
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "type aliases type definitions"

# Find generic types
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "generic types constraints"

# Analyze type complexity
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "complex type definitions"
```

### Dependency Analysis

```bash
# Find internal dependencies
python -m code_flow_graph.cli.code_flow_graph . --language typescript --query "internal module dependencies"

# Find external dependencies
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "external library dependencies"

# Analyze coupling between modules
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "module coupling relationships"
```

### Performance Analysis

```bash
# Find complex functions
python -m code_flow_graph.cli.code_flow_graph . --language typescript --query "high complexity functions"

# Find functions with many dependencies
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "functions with many dependencies"

# Analyze code metrics
python -m code_flow_graph.cli.code_flow_graph . --language typescript --no-analyze --query "code metrics NLOC complexity"
```

## 5. Integration with Development Workflow

### CI/CD Integration

```yaml
# .github/workflows/typescript-analysis.yml
name: TypeScript Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm install

      # No TypeScript installation needed - CodeFlow uses regex parsing

      - name: Install CodeFlow
        run: pip install codeflow

      - name: Analyze TypeScript code
        run: |
          python -m code_flow_graph.cli.code_flow_graph . --language typescript --output analysis.json

      - name: Upload analysis results
        uses: actions/upload-artifact@v3
        with:
          name: typescript-analysis
          path: analysis.json
```

### Pre-commit Hooks

**Setup with husky:**

```bash
# Install husky
npm install husky --save-dev

# Add hook
npx husky add .husky/pre-commit "python -m code_flow_graph.cli.code_flow_graph . --language typescript --query 'recent changes'"
```

### IDE Integration

**VS Code Integration:**

```json
// settings.json
{
  "codeflow.typescript.enabled": true,
  "codeflow.analysis.onSave": true,
  "codeflow.queries": [
    "functions in current file",
    "dependencies of current function"
  ]
}
```

## 6. Best Practices and Recommendations

### Project Organization

1. **Consistent Structure**: Use standard directory layouts for each framework
2. **Project Configuration**: Optional `tsconfig.json` for project structure information
3. **Import Organization**: Use clear import paths for better analysis
4. **Documentation**: Add JSDoc comments for better analysis results

### Analysis Optimization

1. **Regular Analysis**: Run analysis after major code changes
2. **Incremental Queries**: Use `--no-analyze` flag for faster subsequent queries
3. **Focused Queries**: Start with specific queries, then broaden scope
4. **Result Storage**: Save analysis results for historical comparison

### Framework-Specific Optimization

**Angular Projects:**
- Use Angular CLI for consistent project structure
- Implement OnPush change detection for better performance
- Use Angular services for shared business logic

**NestJS Projects:**
- Organize by modules for better separation of concerns
- Use proper dependency injection patterns
- Implement global pipes for request validation

**React Projects:**
- Use custom hooks for reusable logic
- Implement proper TypeScript interfaces for props
- Use context API for state that needs to be accessed by many components

## 7. Troubleshooting Common Issues

### Analysis Not Finding Components

**Problem**: Framework components not detected

**Solutions:**
1. Check decorator syntax:
   ```typescript
   // ✅ Correct
   @Component({ selector: 'app-user' })
   export class UserComponent {}

   // ❌ Missing decorator arguments
   @Component
   export class UserComponent {}
   ```

2. Verify imports:
   ```typescript
   // ✅ Correct imports
   import { Component } from '@angular/core';

   // ❌ Wrong import
   import { Component } from 'angular-core';
   ```

### Type Information Missing

**Problem**: Type annotations not extracted

**Solutions:**
1. Ensure TypeScript files have proper syntax
2. Check tsconfig.json configuration (if present)
3. Verify type annotations are properly formatted

### Performance Issues

**Problem**: Analysis taking too long

**Solutions:**
1. Use incremental analysis (`--no-analyze`)
2. Exclude unnecessary files in tsconfig.json (if present)
3. Process large projects in smaller batches

## Summary

This comprehensive guide demonstrates how to use CodeFlow effectively with TypeScript projects across different frameworks and complexity levels. The key takeaways are:

1. **Setup**: CodeFlow works out-of-the-box with TypeScript files - no external dependencies required
2. **Configuration**: Optional tsconfig.json provides project structure information
3. **Analysis**: Start with broad analysis, then use specific queries for detailed insights
4. **Framework Support**: Leverage built-in framework-specific detection for Angular, NestJS, React, and Express
5. **Integration**: Incorporate analysis into your development workflow for continuous insights

For more detailed troubleshooting, see the [TypeScript Troubleshooting Guide](TYPESCRIPT_TROUBLESHOOTING.md).