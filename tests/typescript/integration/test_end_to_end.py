"""
End-to-end integration tests for TypeScript implementation.
"""

import pytest
import subprocess
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestTypeScriptEndToEndWorkflow:
    """Test complete TypeScript workflow from extraction to analysis."""

    def test_typescript_basic_workflow(self):
        """Test basic TypeScript workflow with CLI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a comprehensive TypeScript project
            (temp_path / "tsconfig.json").write_text("""
            {
              "compilerOptions": {
                "target": "ES2017",
                "module": "commonjs",
                "strict": true
              },
              "include": ["src/**/*"],
              "exclude": ["node_modules"]
            }
            """)

            src_dir = temp_path / "src"
            src_dir.mkdir()

            # Create TypeScript files
            (src_dir / "types.ts").write_text("""
            export interface User {
                id: number;
                name: string;
                email: string;
            }

            export interface Product {
                id: number;
                name: string;
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

            # Run complete analysis
            output_file = temp_path / "analysis.json"

            result = subprocess.run([
                'python', '-m', 'code_flow_graph.cli.code_flow_graph',
                str(temp_path),
                '--language', 'typescript',
                '--output', str(output_file)
            ], capture_output=True, text=True, cwd='.')

            assert result.returncode == 0
            assert output_file.exists()

            # Verify analysis results
            with open(output_file, 'r') as f:
                analysis = json.load(f)

            # Should have found functions and classes
            assert analysis['summary']['total_functions'] >= 3  # createUser, getUser, saveUser
            assert analysis['classes_summary']['total'] >= 2  # Database, UserService
            assert analysis['functions_summary']['total'] >= 3

            # Should have entry points
            assert len(analysis['entry_points']) > 0

            # Should have call graph data
            assert 'call_graph' in analysis
            assert len(analysis['call_graph']['functions']) > 0

    def test_typescript_call_graph_builder_integration(self):
        """Test TypeScript elements work with CallGraphBuilder."""
        from code_flow_graph.core.ast_extractor import TypeScriptASTExtractor
        from code_flow_graph.core.call_graph_builder import CallGraphBuilder

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create TypeScript file with clear relationships
            ts_file = temp_path / "app.ts"
            ts_file.write_text("""
            interface Database {
                findUser(id: number): User | null;
            }

            class DatabaseImpl implements Database {
                private users: User[] = [];

                public findUser(id: number): User | null {
                    return this.users.find(u => u.id === id) || null;
                }
            }

            class UserService {
                constructor(private database: Database) {}

                public async getUser(id: number): Promise<User | null> {
                    return this.database.findUser(id);
                }

                public async processUser(id: number): Promise<string> {
                    const user = await this.getUser(id);
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

            // Entry point
            async function main(): Promise<void> {
                const db = new DatabaseImpl();
                const userService = new UserService(db);

                const result = await userService.processUser(1);
                console.log(result);
            }
            """)

            # Extract TypeScript elements
            extractor = TypeScriptASTExtractor()
            elements = extractor.extract_from_file(ts_file)

            assert len(elements) >= 6  # Database, DatabaseImpl, UserService, getUser, processUser, main

            # Build call graph
            builder = CallGraphBuilder()
            builder.build_from_elements(elements)

            # Verify call graph was built
            assert len(builder.functions) >= 3  # getUser, processUser, main
            assert len(builder.edges) >= 2  # main->processUser, processUser->getUser

            # Check entry points
            entry_points = builder.get_entry_points()
            assert len(entry_points) >= 1

            main_func = next((f for f in entry_points if f.name == 'main'), None)
            assert main_func is not None

    def test_typescript_vector_store_integration(self):
        """Test TypeScript elements work with vector store."""
        from code_flow_graph.core.ast_extractor import TypeScriptASTExtractor
        from code_flow_graph.core.call_graph_builder import CallGraphBuilder
        from code_flow_graph.core.vector_store import CodeVectorStore

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create TypeScript file
            ts_file = temp_path / "api.ts"
            ts_file.write_text("""
            interface ApiResponse<T> {
                data: T;
                success: boolean;
            }

            class ApiClient {
                private baseUrl: string;

                constructor(baseUrl: string) {
                    this.baseUrl = baseUrl;
                }

                public async getUsers(): Promise<ApiResponse<User[]>> {
                    const response = await fetch(`${this.baseUrl}/users`);
                    return response.json();
                }

                public async createUser(user: User): Promise<ApiResponse<User>> {
                    const response = await fetch(`${this.baseUrl}/users`, {
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

            # Extract and build graph
            extractor = TypeScriptASTExtractor()
            elements = extractor.extract_from_file(ts_file)

            builder = CallGraphBuilder()
            builder.build_from_elements(elements)

            # Test vector store integration
            vector_store_path = temp_path / "vectors"
            vector_store = CodeVectorStore(persist_directory=str(vector_store_path))

            # Add functions to vector store
            sources = {ts_file: ts_file.read_text()}
            doc_ids = vector_store.add_function_nodes_batch(
                list(builder.functions.values()),
                sources
            )

            assert len(doc_ids) > 0

            # Test querying
            results = vector_store.query_functions("functions that handle users", n_results=5)
            assert isinstance(results, list)

            # Verify stats
            stats = vector_store.get_stats()
            assert stats['total_documents'] > 0

    def test_typescript_framework_detection_integration(self):
        """Test framework detection in end-to-end workflow."""
        from code_flow_graph.core.ast_extractor import TypeScriptASTVisitor

        # Test different frameworks
        frameworks = {
            'angular': '''
            import { Component, Injectable } from '@angular/core';

            @Component({
                selector: 'app-user',
                template: '<div>User</div>'
            })
            @Injectable()
            export class UserComponent {
                constructor(private userService: UserService) {}

                ngOnInit(): void {
                    this.userService.getUsers();
                }
            }
            ''',
            'nestjs': '''
            import { Controller, Injectable, Get } from '@nestjs/common';

            @Controller('users')
            @Injectable()
            export class UserController {
                constructor(private userService: UserService) {}

                @Get()
                async findAll(): Promise<User[]> {
                    return this.userService.findAll();
                }
            }
            ''',
            'react': '''
            import React, { useState, useEffect } from 'react';

            interface UserProps {
                userId: number;
            }

            const UserProfile: React.FC<UserProps> = ({ userId }) => {
                const [user, setUser] = useState<User | null>(null);

                useEffect(() => {
                    fetchUser(userId);
                }, [userId]);

                return <div>{user?.name}</div>;
            };
            '''
        }

        for expected_framework, code in frameworks.items():
            visitor = TypeScriptASTVisitor()
            elements = visitor.visit_file(f"test.{expected_framework}.ts", code)

            # Find the main class/function
            main_element = None
            for element in elements:
                if element.name in ['UserComponent', 'UserController', 'UserProfile']:
                    main_element = element
                    break

            assert main_element is not None, f"Could not find main element in {expected_framework} code"
            detected_framework = main_element.metadata.get('framework')
            assert detected_framework == expected_framework, \
                f"Framework detection failed for {expected_framework}: got {detected_framework}"

    def test_typescript_error_handling_integration(self):
        """Test error handling in complete workflow."""
        from code_flow_graph.core.ast_extractor import TypeScriptASTExtractor

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create various problematic files
            (temp_path / "malformed.ts").write_text("""
            class IncompleteClass {
                public brokenMethod( {
                    // Missing closing parenthesis
            """)

            (temp_path / "valid.ts").write_text("""
            interface ValidInterface {
                id: number;
                name: string;
            }

            function validFunction(): string {
                return "valid";
            }
            """)

            # Should handle errors gracefully
            extractor = TypeScriptASTExtractor()
            elements = extractor.extract_from_directory(temp_path)

            # Should extract from valid files despite malformed ones
            assert len(elements) >= 2  # ValidInterface + validFunction

            element_names = {e.name for e in elements}
            assert 'ValidInterface' in element_names
            assert 'validFunction' in element_names

    def test_typescript_performance_with_sample_projects(self):
        """Test performance with sample projects."""
        import time

        sample_projects = [
            'tests/typescript/sample_projects/basic',
            'tests/typescript/sample_projects/angular'
        ]

        for project_path in sample_projects:
            if Path(project_path).exists():
                start_time = time.time()

                result = subprocess.run([
                    'python', '-m', 'code_flow_graph.cli.code_flow_graph',
                    project_path,
                    '--language', 'typescript'
                ], capture_output=True, text=True, cwd='.', timeout=60)

                end_time = time.time()
                duration = end_time - start_time

                assert result.returncode == 0, f"Analysis failed for {project_path}"
                assert duration < 30, f"Analysis took too long for {project_path}: {duration}s"

                print(f"✓ {project_path} analyzed in {duration:.2f}s")

    def test_typescript_mcp_server_compatibility(self):
        """Test MCP server compatibility with TypeScript."""
        # Note: This is a basic test since MCP server currently uses PythonASTExtractor
        # In a full implementation, the MCP server would be updated to support TypeScript

        from code_flow_graph.core.ast_extractor import TypeScriptASTExtractor

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create TypeScript project
            (temp_path / "tsconfig.json").write_text('{"compilerOptions": {"target": "ES2017"}}')

            src_dir = temp_path / "src"
            src_dir.mkdir()

            (src_dir / "main.ts").write_text("""
            interface Config {
                apiUrl: string;
            }

            class Application {
                private config: Config;

                constructor(config: Config) {
                    this.config = config;
                }

                public async run(): Promise<void> {
                    console.log('App running with:', this.config.apiUrl);
                }
            }

            async function main(): Promise<void> {
                const config: Config = { apiUrl: 'http://localhost:3000' };
                const app = new Application(config);
                await app.run();
            }

            main().catch(console.error);
            """)

            # Extract TypeScript elements
            extractor = TypeScriptASTExtractor()
            elements = extractor.extract_from_directory(temp_path)

            # Verify extraction worked
            assert len(elements) >= 3  # Config interface, Application class, main function

            # Verify elements have required attributes for MCP server compatibility
            for element in elements:
                assert hasattr(element, 'name')
                assert hasattr(element, 'kind')
                assert hasattr(element, 'file_path')
                assert hasattr(element, 'line_start')
                assert hasattr(element, 'line_end')

                if hasattr(element, 'parameters'):
                    assert isinstance(element.parameters, list)

                if hasattr(element, 'metadata'):
                    assert isinstance(element.metadata, dict)

    def test_typescript_comprehensive_validation(self):
        """Comprehensive validation of TypeScript implementation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create comprehensive TypeScript project
            (temp_path / "tsconfig.json").write_text("""
            {
              "compilerOptions": {
                "target": "ES2017",
                "module": "commonjs",
                "strict": true,
                "baseUrl": "./src",
                "paths": {
                  "@/*": ["*"]
                }
              },
              "include": ["src/**/*"]
            }
            """)

            src_dir = temp_path / "src"
            src_dir.mkdir()

            # Create comprehensive TypeScript files
            (src_dir / "models.ts").write_text("""
            export interface BaseEntity {
                id: number;
                createdAt: Date;
                updatedAt: Date;
            }

            export interface User extends BaseEntity {
                name: string;
                email: string;
                role: 'admin' | 'user' | 'guest';
            }

            export interface Product extends BaseEntity {
                name: string;
                price: number;
                category: string;
            }

            export type UserRole = User['role'];
            export type ProductCategory = Product['category'];
            """)

            (src_dir / "database.ts").write_text("""
            import { User, Product, BaseEntity } from './models';

            export interface Database {
                findUser(id: number): Promise<User | null>;
                saveUser(user: User): Promise<void>;
                findProducts(category?: string): Promise<Product[]>;
            }

            export class DatabaseImpl implements Database {
                private users: User[] = [];
                private products: Product[] = [];

                public async findUser(id: number): Promise<User | null> {
                    const user = this.users.find(u => u.id === id);
                    return user || null;
                }

                public async saveUser(user: User): Promise<void> {
                    this.users.push(user);
                }

                public async findProducts(category?: string): Promise<Product[]> {
                    if (category) {
                        return this.products.filter(p => p.category === category);
                    }
                    return this.products;
                }
            }
            """)

            (src_dir / "services.ts").write_text("""
            import { User, Product } from './models';
            import { Database, DatabaseImpl } from './database';

            export interface UserService {
                getUser(id: number): Promise<User | null>;
                createUser(userData: Omit<User, keyof BaseEntity>): Promise<User>;
                updateUser(id: number, updates: Partial<User>): Promise<User | null>;
            }

            export interface ProductService {
                getProducts(category?: string): Promise<Product[]>;
                createProduct(productData: Omit<Product, keyof BaseEntity>): Promise<Product>;
            }

            export class UserServiceImpl implements UserService {
                constructor(private database: Database) {}

                public async getUser(id: number): Promise<User | null> {
                    return this.database.findUser(id);
                }

                public async createUser(userData: Omit<User, 'id' | 'createdAt' | 'updatedAt'>): Promise<User> {
                    const now = new Date();
                    const user: User = {
                        id: Date.now(),
                        createdAt: now,
                        updatedAt: now,
                        ...userData
                    };

                    await this.database.saveUser(user);
                    return user;
                }

                public async updateUser(id: number, updates: Partial<User>): Promise<User | null> {
                    const existingUser = await this.database.findUser(id);
                    if (!existingUser) {
                        return null;
                    }

                    const updatedUser: User = {
                        ...existingUser,
                        ...updates,
                        updatedAt: new Date()
                    };

                    await this.database.saveUser(updatedUser);
                    return updatedUser;
                }
            }

            export class ProductServiceImpl implements ProductService {
                constructor(private database: Database) {}

                public async getProducts(category?: string): Promise<Product[]> {
                    return this.database.findProducts(category);
                }

                public async createProduct(productData: Omit<Product, 'id' | 'createdAt' | 'updatedAt'>): Promise<Product> {
                    const now = new Date();
                    const product: Product = {
                        id: Date.now(),
                        createdAt: now,
                        updatedAt: now,
                        ...productData
                    };

                    // In a real implementation, this would save to database
                    return product;
                }
            }
            """)

            (src_dir / "app.ts").write_text("""
            import { User, Product } from './models';
            import { DatabaseImpl } from './database';
            import { UserServiceImpl, ProductServiceImpl } from './services';

            export class Application {
                private userService: UserServiceImpl;
                private productService: ProductServiceImpl;
                private database: DatabaseImpl;

                constructor() {
                    this.database = new DatabaseImpl();
                    this.userService = new UserServiceImpl(this.database);
                    this.productService = new ProductServiceImpl(this.database);
                }

                public async initialize(): Promise<void> {
                    console.log('Initializing application...');

                    // Create sample users
                    await this.createSampleUsers();

                    // Create sample products
                    await this.createSampleProducts();

                    console.log('Application initialized successfully');
                }

                private async createSampleUsers(): Promise<void> {
                    try {
                        await this.userService.createUser({
                            name: 'John Doe',
                            email: 'john@example.com',
                            role: 'admin'
                        });

                        await this.userService.createUser({
                            name: 'Jane Smith',
                            email: 'jane@example.com',
                            role: 'user'
                        });
                    } catch (error) {
                        console.error('Error creating users:', error);
                    }
                }

                private async createSampleProducts(): Promise<void> {
                    try {
                        await this.productService.createProduct({
                            name: 'Laptop',
                            price: 999.99,
                            category: 'Electronics'
                        });

                        await this.productService.createProduct({
                            name: 'Coffee Mug',
                            price: 12.99,
                            category: 'Kitchen'
                        });
                    } catch (error) {
                        console.error('Error creating products:', error);
                    }
                }

                public async run(): Promise<void> {
                    await this.initialize();

                    // Demonstrate functionality
                    const users = await this.userService.getUser(1);
                    const products = await this.productService.getProducts();

                    console.log('Users:', users);
                    console.log('Products:', products);
                }
            }

            // CLI entry point
            async function main(): Promise<void> {
                const app = new Application();
                await app.run();
            }

            // Run application if this file is executed directly
            if (require.main === module) {
                main().catch(console.error);
            }
            """)

            # Test complete workflow
            output_file = temp_path / "comprehensive-analysis.json"

            result = subprocess.run([
                'python', '-m', 'code_flow_graph.cli.code_flow_graph',
                str(temp_path),
                '--language', 'typescript',
                '--output', str(output_file)
            ], capture_output=True, text=True, cwd='.', timeout=120)

            assert result.returncode == 0, f"Analysis failed: {result.stderr}"
            assert output_file.exists()

            # Verify comprehensive analysis
            with open(output_file, 'r') as f:
                analysis = json.load(f)

            # Should have extracted many elements
            assert analysis['summary']['total_functions'] >= 8
            assert analysis['classes_summary']['total'] >= 4  # DatabaseImpl, UserServiceImpl, ProductServiceImpl, Application
            assert analysis['functions_summary']['total'] >= 8

            # Should have interfaces
            interface_elements = [e for e in analysis['call_graph']['functions'].values()
                                if e.get('decorators') == [] and 'Interface' not in e.get('name', '')]
            assert len(interface_elements) > 0

            # Should have call graph edges
            assert len(analysis['call_graph']['edges']) > 0

            # Should have entry points
            assert len(analysis['entry_points']) > 0

            print("✓ Comprehensive TypeScript analysis completed successfully")
            print(f"  - Functions: {analysis['summary']['total_functions']}")
            print(f"  - Classes: {analysis['classes_summary']['total']}")
            print(f"  - Entry points: {len(analysis['entry_points'])}")
            print(f"  - Call graph edges: {len(analysis['call_graph']['edges'])}")