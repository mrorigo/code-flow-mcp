"""
Unit tests for TypeScriptASTVisitor functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from code_flow_graph.core.ast_extractor import TypeScriptASTVisitor, TypeScriptASTExtractor


class TestTypeScriptASTVisitor:
    """Test cases for TypeScriptASTVisitor."""

    def test_visitor_initialization(self, typescript_visitor):
        """Test TypeScriptASTVisitor initializes correctly."""
        assert typescript_visitor.elements == []
        assert typescript_visitor.current_class is None
        assert typescript_visitor.current_file == ""
        assert typescript_visitor.source_lines == []
        assert isinstance(typescript_visitor.typescript_available, bool)

    def test_check_typescript_available_with_node(self, typescript_visitor):
        """Test TypeScript availability detection when Node.js is available."""
        with patch('subprocess.run') as mock_run:
            # Mock successful node --version
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout='v18.0.0'),  # node --version
                MagicMock(returncode=0, stdout='4.9.5')     # npx typescript --version
            ]

            result = typescript_visitor._check_typescript_available()
            assert result is True
            assert typescript_visitor.typescript_available is True
            assert mock_run.call_count == 2

    def test_check_typescript_available_without_node(self, typescript_visitor):
        """Test TypeScript availability detection when Node.js is not available."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("node not found")

            result = typescript_visitor._check_typescript_available()
            assert result is False
            assert typescript_visitor.typescript_available is False

    def test_check_typescript_available_without_typescript(self, typescript_visitor):
        """Test TypeScript availability detection when TypeScript is not available."""
        with patch('subprocess.run') as mock_run:
            # Mock successful node but failed typescript
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout='v18.0.0'),  # node --version
                MagicMock(returncode=1, stdout='')           # npx typescript --version fails
            ]

            result = typescript_visitor._check_typescript_available()
            assert result is False
            assert typescript_visitor.typescript_available is False

    def test_extract_jsdoc_comment_single_line(self, typescript_visitor):
        """Test JSDoc comment extraction from single line."""
        lines = [
            "// Regular comment",
            "/** This is a JSDoc comment */",
            "function testFunction() {",
            "    return true;",
            "}"
        ]

        docstring = typescript_visitor._extract_jsdoc_comment(lines, 2)
        assert docstring == "/** This is a JSDoc comment */"

    def test_extract_jsdoc_comment_multi_line(self, typescript_visitor):
        """Test JSDoc comment extraction from multiple lines."""
        lines = [
            "// Regular comment",
            "/**",
            " * This is a multi-line JSDoc comment",
            " * with multiple lines",
            " */",
            "function testFunction() {",
            "    return true;",
            "}"
        ]

        docstring = typescript_visitor._extract_jsdoc_comment(lines, 5)
        expected = "/**\n * This is a multi-line JSDoc comment\n * with multiple lines\n */"
        assert docstring == expected

    def test_extract_jsdoc_comment_none_found(self, typescript_visitor):
        """Test JSDoc comment extraction when no comment is found."""
        lines = [
            "// Regular comment",
            "function testFunction() {",
            "    return true;",
            "}"
        ]

        docstring = typescript_visitor._extract_jsdoc_comment(lines, 1)
        assert docstring is None

    def test_extract_typescript_parameters_simple(self, typescript_visitor):
        """Test parameter extraction from simple function signature."""
        func_source = "function testFunc(param1: string, param2: number)"

        params = typescript_visitor._extract_typescript_parameters(func_source)
        expected = ["param1: string", "param2: number"]
        assert params == expected

    def test_extract_typescript_parameters_complex_types(self, typescript_visitor):
        """Test parameter extraction with complex TypeScript types."""
        func_source = "function testFunc(param1: string[], param2: Record<string, any>, param3: User | null)"

        params = typescript_visitor._extract_typescript_parameters(func_source)
        expected = ["param1: string[]", "param2: Record<string, any>", "param3: User | null"]
        assert params == expected

    def test_extract_typescript_parameters_no_params(self, typescript_visitor):
        """Test parameter extraction from function with no parameters."""
        func_source = "function testFunc()"

        params = typescript_visitor._extract_typescript_parameters(func_source)
        assert params == []

    def test_calculate_typescript_complexity_simple(self, typescript_visitor):
        """Test complexity calculation for simple function."""
        func_source = """
        function simpleFunc() {
            if (true) {
                return 1;
            }
            return 0;
        }
        """

        complexity = typescript_visitor._calculate_typescript_complexity(func_source)
        assert complexity == 3  # Base + 1 if + 1 for if body

    def test_calculate_typescript_complexity_complex(self, typescript_visitor):
        """Test complexity calculation for complex function."""
        func_source = """
        function complexFunc(a, b) {
            if (a > 0 && b < 10) {
                for (let i = 0; i < a; i++) {
                    if (i % 2 === 0) {
                        console.log(i);
                    } else if (i % 3 === 0) {
                        console.log("Multiple of 3");
                    } else {
                        console.log("Other");
                    }
                }
            } else {
                while (b > 0) {
                    b--;
                }
            }
            return a + b;
        }
        """

        complexity = typescript_visitor._calculate_typescript_complexity(func_source)
        # Base(1) + if(1) + &&(1) + for(1) + if(1) + else if(1) + else(1) + while(1)
        assert complexity == 8

    def test_calculate_nloc_simple(self, typescript_visitor):
        """Test NLOC calculation for simple code."""
        lines = [
            "function testFunc() {",
            "    // Comment line",
            "    if (true) {",
            "        console.log('test');",
            "    }",
            "    return true; // inline comment",
            "}"
        ]

        nloc = typescript_visitor._calculate_nloc(1, 7, lines)
        assert nloc == 4  # Excludes comment line and inline comment

    def test_calculate_nloc_with_jsdoc(self, typescript_visitor):
        """Test NLOC calculation including JSDoc comments."""
        lines = [
            "/**",
            " * This is a JSDoc comment",
            " */",
            "function testFunc() {",
            "    // This is a regular comment",
            "    const x = 1;",
            "    return x;",
            "}"
        ]

        nloc = typescript_visitor._calculate_nloc(1, 8, lines)
        assert nloc == 3  # Function declaration + const + return

    def test_extract_typescript_dependencies(self, typescript_visitor):
        """Test external dependency extraction."""
        func_source = """
        import React from 'react';
        import { useState } from 'react';

        function testFunc() {
            const users = await userService.getUsers();
            const result = lodash.map(users, 'name');
            return result;
        }
        """

        dependencies = typescript_visitor._extract_typescript_dependencies(func_source)
        expected = ['lodash', 'react']  # Only external dependencies, not local services
        assert set(dependencies) == set(expected)

    def test_extract_file_imports_es6(self, typescript_visitor):
        """Test ES6 import extraction."""
        source = """
        import React from 'react';
        import { useState, useEffect } from 'react';
        import * as _ from 'lodash';
        import { UserService } from './user.service';
        import './styles.css';

        function testFunc() {
            // function body
        }
        """

        typescript_visitor._extract_file_imports(source)

        expected_imports = {
            'React': 'react',
            'useState': 'react',
            'useEffect': 'react',
            '_': 'lodash',
            'UserService': './user.service'
        }
        expected_targets = {'useState', 'useEffect'}

        assert typescript_visitor.file_level_imports == expected_imports
        assert typescript_visitor.file_level_import_from_targets == expected_targets

    def test_detect_framework_patterns_angular(self, typescript_visitor):
        """Test Angular framework pattern detection."""
        source = """
        @Component({
            selector: 'app-user',
            template: '<div>User Component</div>'
        })
        @Injectable()
        export class UserComponent {
            constructor(private userService: UserService) {}

            ngOnInit() {
                this.userService.getUsers();
            }
        }
        """

        patterns = typescript_visitor._detect_framework_patterns(source)
        assert patterns['framework'] == 'angular'
        assert 'components' in patterns['features']
        assert 'dependency_injection' in patterns['features']
        assert len(patterns['decorators']) > 0

    def test_detect_framework_patterns_nestjs(self, typescript_visitor):
        """Test NestJS framework pattern detection."""
        source = """
        @Controller('users')
        @Injectable()
        export class UserController {
            constructor(private userService: UserService) {}

            @Get()
            async findAll() {
                return this.userService.findAll();
            }

            @Post()
            async create(@Body() createDto: any) {
                return this.userService.create(createDto);
            }
        }
        """

        patterns = typescript_visitor._detect_framework_patterns(source)
        assert patterns['framework'] == 'nestjs'
        assert 'controllers' in patterns['features']
        assert 'dependency_injection' in patterns['features']

    def test_detect_framework_patterns_react(self, typescript_visitor):
        """Test React framework pattern detection."""
        source = """
        import React, { useState, useEffect } from 'react';

        const UserComponent: React.FC<UserProps> = ({ userId }) => {
            const [user, setUser] = useState<User | null>(null);

            useEffect(() => {
                fetchUser(userId);
            }, [userId]);

            return (
                <div>
                    <h1>{user?.name}</h1>
                    <p>{user?.email}</p>
                </div>
            );
        };
        """

        patterns = typescript_visitor._detect_framework_patterns(source)
        assert patterns['framework'] == 'react'
        assert 'hooks' in patterns['features']
        assert 'jsx' in patterns['features']

    def test_detect_framework_patterns_express(self, typescript_visitor):
        """Test Express framework pattern detection."""
        source = """
        import express from 'express';
        import { UserService } from './user.service';

        const app = express();
        const userService = new UserService();

        app.get('/users', (req, res) => {
            userService.getAll().then(users => res.json(users));
        });

        app.post('/users', (req, res) => {
            userService.create(req.body).then(user => res.json(user));
        });

        export default app;
        """

        patterns = typescript_visitor._detect_framework_patterns(source)
        assert patterns['framework'] == 'express'
        assert 'routing' in patterns['features']
        assert 'middleware' in patterns['features']

    def test_hash_source_snippet(self, typescript_visitor):
        """Test source snippet hashing."""
        lines = [
            "function testFunc() {",
            "    return 'hello';",
            "}"
        ]

        hash_result = typescript_visitor._hash_source_snippet(1, 3, lines)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 32  # MD5 hash length

        # Same content should produce same hash
        hash_result2 = typescript_visitor._hash_source_snippet(1, 3, lines)
        assert hash_result == hash_result2

        # Different content should produce different hash
        lines_modified = [
            "function testFunc() {",
            "    return 'world';",
            "}"
        ]
        hash_result3 = typescript_visitor._hash_source_snippet(1, 3, lines_modified)
        assert hash_result != hash_result3

    def test_parse_complex_types_union(self, typescript_visitor):
        """Test union type parsing."""
        source = """
        type Result<T> = T | null | undefined;
        type Status = 'active' | 'inactive' | 'pending';
        """

        complex_types = typescript_visitor._parse_complex_types(source)

        assert len(complex_types['union_types']) == 2
        assert complex_types['union_types'][0]['name'] == 'Result'
        assert complex_types['union_types'][1]['name'] == 'Status'

    def test_parse_complex_types_interface(self, typescript_visitor):
        """Test interface parsing."""
        source = """
        interface User {
            id: number;
            name: string;
        }

        interface Admin extends User {
            role: string;
        }
        """

        complex_types = typescript_visitor._parse_complex_types(source)

        assert len(complex_types['interfaces']) == 2
        assert complex_types['interfaces'][0]['name'] == 'User'
        assert complex_types['interfaces'][1]['name'] == 'Admin'
        assert complex_types['interfaces'][1]['extends'] == ['User']

    def test_parse_complex_types_enum(self, typescript_visitor):
        """Test enum parsing."""
        source = """
        enum Status {
            ACTIVE = 'active',
            INACTIVE = 'inactive',
            PENDING = 'pending'
        }
        """

        complex_types = typescript_visitor._parse_complex_types(source)

        assert len(complex_types['enums']) == 1
        assert complex_types['enums'][0]['name'] == 'Status'
        # Check that enum members were parsed correctly
        members = complex_types['enums'][0]['members']
        member_names = [member['name'] for member in members]
        assert 'ACTIVE' in member_names

    def test_parse_complex_types_utility_types(self, typescript_visitor):
        """Test utility type parsing."""
        source = """
        type PartialUser = Partial<User>;
        type ReadonlyUser = Readonly<User>;
        type UserWithoutEmail = Omit<User, 'email'>;
        """

        complex_types = typescript_visitor._parse_complex_types(source)

        assert len(complex_types['utility_types']) == 3
        # Check that utility types were detected (the structure might be different)
        assert len(complex_types['utility_types']) == 3
        utility_names = [item['name'] for item in complex_types['utility_types']]
        assert 'PartialUser' in utility_names
        assert complex_types['utility_types'][1]['utility'] == 'Readonly'
        assert complex_types['utility_types'][2]['utility'] == 'Omit'

    def test_visit_file_basic_function(self, typescript_visitor, temp_dir):
        """Test visiting a file with a basic function."""
        ts_file = temp_dir / "test.ts"
        source = """
        function greetUser(name: string): string {
            return `Hello, ${name}!`;
        }
        """

        with open(ts_file, 'w') as f:
            f.write(source)

        elements = typescript_visitor.visit_file(str(ts_file), source)

        assert len(elements) >= 1
        func_element = next((e for e in elements if e.name == 'greetUser'), None)
        assert func_element is not None
        assert func_element.kind == 'function'
        assert 'name: string' in func_element.parameters
        assert func_element.return_type == 'string'

    def test_visit_file_with_class(self, typescript_visitor, temp_dir):
        """Test visiting a file with a class."""
        ts_file = temp_dir / "test.ts"
        source = """
        class UserService {
            private users: User[] = [];

            public getUser(id: number): User | null {
                return this.users.find(u => u.id === id) || null;
            }
        }
        """

        with open(ts_file, 'w') as f:
            f.write(source)

        elements = typescript_visitor.visit_file(str(ts_file), source)

        assert len(elements) >= 2  # Class + method
        class_element = next((e for e in elements if e.name == 'UserService'), None)
        assert class_element is not None
        assert class_element.kind == 'class'

        func_element = next((e for e in elements if e.name == 'getUser'), None)
        assert func_element is not None
        assert func_element.is_method is True
        assert func_element.class_name == 'UserService'

    def test_visit_file_with_interface(self, typescript_visitor, temp_dir):
        """Test visiting a file with an interface."""
        ts_file = temp_dir / "test.ts"
        source = """
        interface User {
            id: number;
            name: string;
            email: string;
        }
        """

        with open(ts_file, 'w') as f:
            f.write(source)

        elements = typescript_visitor.visit_file(str(ts_file), source)

        interface_element = next((e for e in elements if e.name == 'User'), None)
        assert interface_element is not None
        assert interface_element.kind == 'interface'
        assert interface_element.metadata.get('typescript_kind') == 'interface'


class TestTypeScriptASTExtractor:
    """Test cases for TypeScriptASTExtractor."""

    def test_extractor_initialization(self, typescript_extractor):
        """Test TypeScriptASTExtractor initializes correctly."""
        assert isinstance(typescript_extractor.visitor, TypeScriptASTVisitor)
        assert typescript_extractor.project_root is None
        assert typescript_extractor.tsconfig is None
        assert typescript_extractor.project_references == []
        assert typescript_extractor.path_mappings == {}

    def test_find_tsconfig_in_directory(self, typescript_extractor, temp_dir):
        """Test tsconfig.json file detection."""
        # Create tsconfig.json
        tsconfig = temp_dir / "tsconfig.json"
        tsconfig.write_text('{"compilerOptions": {"target": "ES2020"}}')

        result = typescript_extractor._find_tsconfig(temp_dir)
        assert result == tsconfig

    def test_find_tsconfig_in_parent_directory(self, typescript_extractor, temp_dir):
        """Test tsconfig.json detection in parent directories."""
        # Create nested directory structure
        subdir = temp_dir / "src" / "components"
        subdir.mkdir(parents=True)

        # Create tsconfig.json in root
        tsconfig = temp_dir / "tsconfig.json"
        tsconfig.write_text('{"compilerOptions": {"target": "ES2020"}}')

        result = typescript_extractor._find_tsconfig(subdir)
        assert result == tsconfig

    def test_find_tsconfig_not_found(self, typescript_extractor, temp_dir):
        """Test tsconfig.json detection when file doesn't exist."""
        result = typescript_extractor._find_tsconfig(temp_dir)
        assert result is None

    def test_parse_tsconfig_valid(self, typescript_extractor, temp_dir):
        """Test parsing of valid tsconfig.json."""
        tsconfig = temp_dir / "tsconfig.json"
        config_content = {
            "compilerOptions": {
                "target": "ES2020",
                "strict": True
            },
            "include": ["src/**/*"]
        }
        import json
        tsconfig.write_text(json.dumps(config_content))

        result = typescript_extractor._parse_tsconfig(tsconfig)
        assert result["compilerOptions"]["target"] == "ES2020"
        assert result["compilerOptions"]["strict"] is True
        assert "include" in result

    def test_parse_tsconfig_invalid_json(self, typescript_extractor, temp_dir):
        """Test parsing of invalid tsconfig.json."""
        tsconfig = temp_dir / "tsconfig.json"
        tsconfig.write_text('{"invalid": json}')

        result = typescript_extractor._parse_tsconfig(tsconfig)
        assert result == {}

    def test_setup_project_integration(self, typescript_extractor, temp_dir):
        """Test project integration setup."""
        # Create tsconfig.json
        tsconfig = temp_dir / "tsconfig.json"
        config_content = {
            "compilerOptions": {
                "target": "ES2020",
                "baseUrl": "./src",
                "paths": {
                    "@/*": ["*"],
                    "@/components/*": ["components/*"]
                }
            },
            "references": [
                {"path": "./packages/core"},
                {"path": "./packages/ui"}
            ]
        }
        tsconfig.write_text(str(config_content).replace("'", '"'))

        typescript_extractor._setup_project_integration(temp_dir)

        assert typescript_extractor.tsconfig is not None
        assert typescript_extractor.project_references == config_content["references"]
        assert typescript_extractor.path_mappings["baseUrl"] == "./src"
        assert typescript_extractor.path_mappings["paths"] == config_content["compilerOptions"]["paths"]

    def test_resolve_path_mapping(self, typescript_extractor, temp_dir):
        """Test path mapping resolution."""
        typescript_extractor.path_mappings = {
            "baseUrl": "./src",
            "paths": {
                "@/*": ["*"],
                "@/components/*": ["components/*"]
            }
        }

        resolved = typescript_extractor._resolve_path_mapping("@/components/Button")
        assert resolved == "components/Button"

    def test_extract_from_file_basic(self, typescript_extractor, temp_dir):
        """Test extraction from a single TypeScript file."""
        ts_file = temp_dir / "test.ts"
        source = """
        interface User {
            id: number;
            name: string;
        }

        function getUser(id: number): User | null {
            return null;
        }
        """

        ts_file.write_text(source)

        elements = typescript_extractor.extract_from_file(ts_file)

        assert len(elements) >= 2
        interface_element = next((e for e in elements if e.name == 'User'), None)
        func_element = next((e for e in elements if e.name == 'getUser'), None)

        assert interface_element is not None
        assert func_element is not None

    def test_extract_from_empty_file(self, typescript_extractor, temp_dir):
        """Test extraction from empty TypeScript file."""
        ts_file = temp_dir / "empty.ts"
        ts_file.write_text("")

        elements = typescript_extractor.extract_from_file(ts_file)
        assert elements == []

    def test_extract_from_nonexistent_file(self, typescript_extractor):
        """Test extraction from nonexistent file."""
        nonexistent = Path("nonexistent.ts")

        elements = typescript_extractor.extract_from_file(nonexistent)
        assert elements == []

    def test_extract_from_non_typescript_file(self, typescript_extractor, temp_dir):
        """Test extraction from non-TypeScript file."""
        js_file = temp_dir / "test.js"
        js_file.write_text("console.log('hello');")

        elements = typescript_extractor.extract_from_file(js_file)
        assert elements == []