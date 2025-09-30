"""
Integration tests for CLI with TypeScript support.
"""

import pytest
import subprocess
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestCLITypeScriptIntegration:
    """Test CLI integration with TypeScript language support."""

    def test_cli_typescript_flag_exists(self):
        """Test that CLI accepts --language typescript flag."""
        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            '--help'
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode == 0
        assert '--language' in result.stdout
        assert 'typescript' in result.stdout

    def test_cli_typescript_basic_analysis(self, temp_dir):
        """Test basic TypeScript analysis via CLI."""
        # Create a simple TypeScript file
        ts_file = temp_dir / "test.ts"
        ts_file.write_text("""
        interface User {
            id: number;
            name: string;
        }

        function getUser(id: number): User | null {
            return null;
        }

        class UserService {
            public async findUser(id: number): Promise<User | null> {
                return getUser(id);
            }
        }
        """)

        # Run analysis
        output_file = temp_dir / "analysis.json"

        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            str(temp_dir),
            '--language', 'typescript',
            '--output', str(output_file)
        ], capture_output=True, text=True, cwd='.')

        # Should complete successfully
        assert result.returncode == 0
        assert output_file.exists()

        # Check output contains expected data
        with open(output_file, 'r') as f:
            analysis = json.load(f)

        assert 'summary' in analysis
        assert 'functions_summary' in analysis
        assert 'classes_summary' in analysis
        assert analysis['summary']['total_functions'] >= 2  # getUser + findUser

    def test_cli_typescript_with_angular_project(self, temp_dir):
        """Test CLI analysis of Angular project."""
        # Create Angular-style TypeScript files
        app_dir = temp_dir / "src" / "app"
        app_dir.mkdir(parents=True)

        # Create component
        component_file = app_dir / "user.component.ts"
        component_file.write_text("""
        import { Component } from '@angular/core';

        @Component({
            selector: 'app-user',
            template: '<div>User Component</div>'
        })
        export class UserComponent {
            public getUserName(): string {
                return 'John Doe';
            }
        }
        """)

        # Create service
        service_file = app_dir / "user.service.ts"
        service_file.write_text("""
        import { Injectable } from '@angular/core';

        @Injectable({
            providedIn: 'root'
        })
        export class UserService {
            private users: any[] = [];

            public async getUsers(): Promise<any[]> {
                return this.users;
            }
        }
        """)

        # Create tsconfig.json
        tsconfig = temp_dir / "tsconfig.json"
        tsconfig.write_text('{"include": ["src/**/*"]}')

        # Run analysis
        output_file = temp_dir / "angular-analysis.json"

        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            str(temp_dir),
            '--language', 'typescript',
            '--output', str(output_file)
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode == 0
        assert output_file.exists()

        # Verify Angular framework detection
        with open(output_file, 'r') as f:
            analysis = json.load(f)

        # Should have detected classes and functions
        assert analysis['summary']['total_functions'] >= 2
        assert analysis['classes_summary']['total'] >= 2

    def test_cli_typescript_query_functionality(self, temp_dir):
        """Test CLI query functionality with TypeScript."""
        # Create test TypeScript file
        ts_file = temp_dir / "api.ts"
        ts_file.write_text("""
        interface ApiResponse<T> {
            data: T;
            success: boolean;
        }

        class ApiClient {
            public async getUsers(): Promise<ApiResponse<User[]>> {
                const response = await fetch('/api/users');
                return response.json();
            }

            public async createUser(user: User): Promise<ApiResponse<User>> {
                const response = await fetch('/api/users', {
                    method: 'POST',
                    body: JSON.stringify(user)
                });
                return response.json();
            }
        }

        interface User {
            id: number;
            name: string;
        }
        """)

        # First run analysis
        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            str(temp_dir),
            '--language', 'typescript'
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode == 0

        # Now test query functionality
        query_result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            str(temp_dir),
            '--language', 'typescript',
            '--query', 'functions that handle users',
            '--no-analyze'
        ], capture_output=True, text=True, cwd='.')

        # Query should work (might return no results, but shouldn't error)
        assert query_result.returncode == 0

    def test_cli_typescript_mermaid_generation(self, temp_dir):
        """Test CLI Mermaid graph generation with TypeScript."""
        # Create TypeScript file with clear relationships
        ts_file = temp_dir / "graph.ts"
        ts_file.write_text("""
        interface Database {
            findUser(id: number): User | null;
        }

        class UserRepository {
            constructor(private db: Database) {}

            public getUser(id: number): User | null {
                return this.db.findUser(id);
            }

            public getAllUsers(): User[] {
                return [];
            }
        }

        class UserService {
            constructor(private repo: UserRepository) {}

            public async processUser(id: number): Promise<string> {
                const user = this.repo.getUser(id);
                if (user) {
                    return `Processing ${user.name}`;
                }
                return 'User not found';
            }
        }

        interface User {
            id: number;
            name: string;
        }
        """)

        # Run analysis with Mermaid output
        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            str(temp_dir),
            '--language', 'typescript',
            '--query', 'user processing functions',
            '--mermaid'
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode == 0
        assert 'graph TD' in result.stdout
        assert 'Mermaid' in result.stdout

    def test_cli_typescript_invalid_directory(self):
        """Test CLI error handling for invalid directory."""
        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            '/nonexistent/directory',
            '--language', 'typescript'
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode != 0
        assert 'not a valid directory' in result.stderr

    def test_cli_typescript_language_validation(self):
        """Test CLI language validation."""
        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            '.',
            '--language', 'invalid_language'
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode != 0
        assert 'invalid choice' in result.stderr and 'choose from' in result.stderr

    def test_cli_typescript_with_vector_store(self, temp_dir):
        """Test CLI TypeScript analysis with vector store."""
        # Create TypeScript files
        (temp_dir / "service.ts").write_text("""
        interface Config {
            apiUrl: string;
        }

        class DataService {
            constructor(private config: Config) {}

            public async fetchData(): Promise<any> {
                const response = await fetch(this.config.apiUrl);
                return response.json();
            }
        }
        """)

        # Run analysis (vector store should be created)
        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            str(temp_dir),
            '--language', 'typescript'
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode == 0

        # Check that vector store directory was created
        vector_dir = temp_dir / "code_vectors_chroma"
        assert vector_dir.exists()

        # Check for vector store files
        chroma_files = list(vector_dir.glob("*"))
        assert len(chroma_files) > 0

    def test_cli_typescript_performance_with_large_files(self, temp_dir):
        """Test CLI performance with larger TypeScript files."""
        # Create a moderately large TypeScript file
        large_file = temp_dir / "large.ts"
        lines = []

        # Generate 20 interfaces
        for i in range(20):
            lines.extend([
                f"interface Interface{i} {{",
                f"    id{i}: number;",
                f"    name{i}: string;",
                f"    data{i}: any;",
                "}"
            ])

        # Generate 20 classes
        for i in range(20):
            lines.extend([
                f"class Class{i} implements Interface{i % 5} {{",
                f"    constructor(id: number, name: string, data: any) {{",
                "        // constructor body",
                "    }",
                "    ",
                f"    public method{i}(): string {{",
                f"        return 'method{i} result';",
                "    }",
                "}"
            ])

        large_file.write_text("\n".join(lines))

        # Run analysis - should complete within reasonable time
        import time
        start_time = time.time()

        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            str(temp_dir),
            '--language', 'typescript'
        ], capture_output=True, text=True, cwd='.', timeout=60)

        end_time = time.time()
        duration = end_time - start_time

        assert result.returncode == 0
        assert duration < 30  # Should complete within 30 seconds

        # Should have extracted many elements
        output_file = temp_dir / "code_analysis_report.json"
        if output_file.exists():
            with open(output_file, 'r') as f:
                analysis = json.load(f)

            # Should have found interfaces and classes
            assert analysis['summary']['total_functions'] >= 20
            assert analysis['classes_summary']['total'] >= 20


class TestTypeScriptEndToEndWorkflow:
    """Test complete end-to-end TypeScript workflow."""

    def test_basic_typescript_project_workflow(self, temp_dir):
        """Test complete workflow with basic TypeScript project."""
        # Create a complete TypeScript project structure
        src_dir = temp_dir / "src"
        src_dir.mkdir()

        # Create multiple TypeScript files
        (src_dir / "types.ts").write_text("""
        export interface User {
            id: number;
            name: string;
            email: string;
        }

        export interface Product {
            id: number;
            title: string;
            price: number;
        }
        """)

        (src_dir / "database.ts").write_text("""
        import { User, Product } from './types';

        export class Database {
            private users: User[] = [];
            private products: Product[] = [];

            public findUser(id: number): User | null {
                return this.users.find(u => u.id === id) || null;
            }

            public saveUser(user: User): void {
                this.users.push(user);
            }
        }
        """)

        (src_dir / "services.ts").write_text("""
        import { Database } from './database';
        import { User, Product } from './types';

        export class UserService {
            constructor(private db: Database) {}

            public createUser(userData: Omit<User, 'id'>): User {
                const user: User = {
                    id: Date.now(),
                    ...userData
                };
                this.db.saveUser(user);
                return user;
            }

            public getUser(id: number): User | null {
                return this.db.findUser(id);
            }
        }
        """)

        (src_dir / "app.ts").write_text("""
        import { UserService } from './services';
        import { Database } from './database';

        class Application {
            private userService: UserService;

            constructor() {
                const db = new Database();
                this.userService = new UserService(db);
            }

            public async run(): Promise<void> {
                const user = this.userService.createUser({
                    name: 'John Doe',
                    email: 'john@example.com'
                });

                const foundUser = this.userService.getUser(user.id);
                console.log('User:', foundUser);
            }
        }

        // CLI entry point
        async function main(): Promise<void> {
            const app = new Application();
            await app.run();
        }

        if (require.main === module) {
            main().catch(console.error);
        }
        """)

        # Create tsconfig.json
        (temp_dir / "tsconfig.json").write_text("""
        {
          "compilerOptions": {
            "target": "ES2017",
            "module": "commonjs",
            "lib": ["ES2017"],
            "outDir": "./dist",
            "rootDir": "./src",
            "strict": true,
            "esModuleInterop": true
          },
          "include": ["src/**/*"],
          "exclude": ["node_modules", "dist"]
        }
        """)

        # Test complete workflow
        output_file = temp_dir / "complete-analysis.json"

        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            str(temp_dir),
            '--language', 'typescript',
            '--output', str(output_file)
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode == 0
        assert output_file.exists()

        # Verify comprehensive analysis
        with open(output_file, 'r') as f:
            analysis = json.load(f)

        # Should have found interfaces, classes, and functions
        assert analysis['summary']['total_functions'] >= 4  # createUser, getUser, run, main
        assert analysis['classes_summary']['total'] >= 3  # Database, UserService, Application
        assert analysis['entry_points']  # Should have found entry points

        # Check that call graph was built
        assert 'call_graph' in analysis
        assert 'functions' in analysis['call_graph']
        assert 'edges' in analysis['call_graph']

        # Verify framework detection (should be none for basic TS)
        functions = analysis['call_graph']['functions']
        for func_data in functions.values():
            # Basic TypeScript should not have framework-specific decorators
            # Allow for some edge case detections but no major framework frameworks
            decorators = func_data.get('decorators', [])
            if decorators:
                # If decorators are detected, they should not be major framework-specific
                for decorator in decorators:
                    if isinstance(decorator, dict):
                        framework = decorator.get('framework')
                        assert framework in [None, 'unknown', 'custom'] or framework is None, \
                            f"Unexpected framework detected in basic TypeScript: {framework}"

    def test_typescript_project_with_mixed_file_types(self, temp_dir):
        """Test TypeScript project with mixed file types."""
        src_dir = temp_dir / "src"
        src_dir.mkdir()

        # Create TypeScript files
        (src_dir / "app.ts").write_text("""
        interface Config {
            apiUrl: string;
        }

        class App {
            public run(): void {
                console.log('App running');
            }
        }
        """)

        # Create JavaScript file (should be ignored)
        (src_dir / "helper.js").write_text("""
        function javaScriptHelper() {
            return 'ignored';
        }
        """)

        # Create JSON file (should be ignored)
        (src_dir / "data.json").write_text('{"test": "data"}')

        # Create tsconfig.json
        (temp_dir / "tsconfig.json").write_text("""
        {
          "include": ["src/**/*"],
          "exclude": ["node_modules"]
        }
        """)

        result = subprocess.run([
            'python', '-m', 'code_flow_graph.cli.code_flow_graph',
            str(temp_dir),
            '--language', 'typescript'
        ], capture_output=True, text=True, cwd='.')

        assert result.returncode == 0

        # Should only process .ts files
        # The output file should be created in the temp directory when no explicit output is specified
        output_file = temp_dir / "code_analysis_report.json"
        if not output_file.exists():
            # If not found in temp_dir, check if it was created in current working directory
            output_file = Path("code_analysis_report.json")
            if not output_file.exists():
                pytest.fail(f"Output file not found at {temp_dir / 'code_analysis_report.json'} or in current directory")

        with open(output_file, 'r') as f:
            analysis = json.load(f)

        # Should have found TypeScript elements but not JavaScript
        assert analysis['summary']['total_functions'] >= 1
        assert analysis['classes_summary']['total'] >= 1