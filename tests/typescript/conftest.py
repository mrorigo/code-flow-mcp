"""
Pytest configuration and fixtures for TypeScript testing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from code_flow_graph.core.typescript_extractor import TypeScriptASTExtractor, TypeScriptASTVisitor
from code_flow_graph.core.call_graph_builder import CallGraphBuilder
from code_flow_graph.core.vector_store import CodeVectorStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def typescript_visitor():
    """Create a TypeScriptASTVisitor instance for testing."""
    return TypeScriptASTVisitor()


@pytest.fixture
def typescript_extractor():
    """Create a TypeScriptASTExtractor instance for testing."""
    return TypeScriptASTExtractor()


@pytest.fixture
def call_graph_builder():
    """Create a CallGraphBuilder instance for testing."""
    builder = CallGraphBuilder()
    return builder


@pytest.fixture
def vector_store(temp_dir):
    """Create a CodeVectorStore instance for testing."""
    store_path = temp_dir / "test_vectors"
    return CodeVectorStore(persist_directory=str(store_path))


@pytest.fixture
def sample_typescript_code():
    """Sample TypeScript code for testing."""
    return '''
interface User {
    id: number;
    name: string;
    email: string;
}

class UserService {
    private users: User[] = [];

    public async getUser(id: number): Promise<User | null> {
        const user = this.users.find(u => u.id === id);
        if (!user) {
            return null;
        }
        return user;
    }

    public createUser(userData: Omit<User, 'id'>): User {
        const newUser: User = {
            id: Date.now(),
            ...userData
        };
        this.users.push(newUser);
        return newUser;
    }
}

function helperFunction(value: string): boolean {
    return value.length > 0;
}

export { User, UserService, helperFunction };
'''


@pytest.fixture
def sample_angular_code():
    """Sample Angular TypeScript code for testing."""
    return '''
import { Component, Input, OnInit } from '@angular/core';

@Component({
    selector: 'app-user-card',
    template: `
        <div class="user-card">
            <h3>{{user.name}}</h3>
            <p>{{user.email}}</p>
        </div>
    `,
    styles: [`
        .user-card { border: 1px solid #ccc; padding: 1rem; }
    `]
})
export class UserCardComponent implements OnInit {
    @Input() user: any;

    ngOnInit(): void {
        console.log('Component initialized');
    }
}
'''


@pytest.fixture
def sample_nestjs_code():
    """Sample NestJS TypeScript code for testing."""
    return '''
import { Controller, Get, Post, Body, Param } from '@nestjs/common';
import { UserService } from './user.service';
import { User } from './user.interface';

@Controller('users')
export class UserController {
    constructor(private readonly userService: UserService) {}

    @Get()
    async findAll(): Promise<User[]> {
        return this.userService.findAll();
    }

    @Get(':id')
    async findOne(@Param('id') id: string): Promise<User> {
        return this.userService.findOne(+id);
    }

    @Post()
    async create(@Body() createUserDto: any): Promise<User> {
        return this.userService.create(createUserDto);
    }
}
'''


@pytest.fixture
def sample_react_code():
    """Sample React TypeScript code for testing."""
    return '''
import React, { useState, useEffect } from 'react';
import { User } from './types';

interface UserProfileProps {
    userId: number;
    onUserLoad?: (user: User) => void;
}

const UserProfile: React.FC<UserProfileProps> = ({ userId, onUserLoad }) => {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState<boolean>(true);

    useEffect(() => {
        fetchUser(userId);
    }, [userId]);

    const fetchUser = async (id: number): Promise<void> => {
        try {
            setLoading(true);
            const response = await fetch(`/api/users/${id}`);
            const userData = await response.json();
            setUser(userData);
            onUserLoad?.(userData);
        } catch (error) {
            console.error('Failed to fetch user:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div>Loading...</div>;
    if (!user) return <div>User not found</div>;

    return (
        <div className="user-profile">
            <h1>{user.name}</h1>
            <p>{user.email}</p>
        </div>
    );
};

export default UserProfile;
'''


@pytest.fixture
def sample_express_code():
    """Sample Express TypeScript code for testing."""
    return '''
import express, { Request, Response, NextFunction } from 'express';
import { UserService } from './user.service';

const app = express();
const userService = new UserService();

app.use(express.json());

// Middleware
const logger = (req: Request, res: Response, next: NextFunction) => {
    console.log(`${req.method} ${req.path}`);
    next();
};

app.use(logger);

// Routes
app.get('/users', async (req: Request, res: Response) => {
    try {
        const users = await userService.getAllUsers();
        res.json(users);
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.get('/users/:id', async (req: Request, res: Response) => {
    try {
        const user = await userService.getUserById(parseInt(req.params.id));
        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.json(user);
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.post('/users', async (req: Request, res: Response) => {
    try {
        const newUser = await userService.createUser(req.body);
        res.status(201).json(newUser);
    } catch (error) {
        res.status(400).json({ error: 'Bad request' });
    }
});

export default app;
'''


@pytest.fixture
def tsconfig_json():
    """Sample tsconfig.json for testing."""
    return {
        "compilerOptions": {
            "target": "ES2020",
            "module": "commonjs",
            "lib": ["ES2020"],
            "outDir": "./dist",
            "rootDir": "./src",
            "strict": True,
            "esModuleInterop": True,
            "skipLibCheck": True,
            "forceConsistentCasingInFileNames": True,
            "baseUrl": "./src",
            "paths": {
                "@/*": ["*"],
                "@/components/*": ["components/*"],
                "@/services/*": ["services/*"]
            }
        },
        "include": [
            "src/**/*"
        ],
        "exclude": [
            "node_modules",
            "dist"
        ]
    }


@pytest.fixture
def malformed_typescript_code():
    """Sample malformed TypeScript code for error testing."""
    return '''
class IncompleteClass {
    public brokenMethod( {
        // Missing closing parenthesis

    public anotherBrokenMethod(param: string {
        // Missing closing parenthesis in parameter

    private invalidProperty: ; // Missing type

    constructor() {
        this.invalidProperty = "test";
        // Property initialized but type not specified
    }
}

function malformedFunction(param1: string, param2: number { // Missing closing parenthesis
    console.log("This function is malformed");
    return param1 + param2; // This will cause type error
}

interface BrokenInterface {
    id: number;
    name: string
    // Missing semicolon

export { IncompleteClass, malformedFunction };
'''


@pytest.fixture
def large_typescript_file():
    """Large TypeScript file for performance testing."""
    lines = []
    lines.append("// Large TypeScript file for performance testing")
    lines.append("")

    # Generate 50 interfaces
    for i in range(50):
        lines.append(f"interface Interface{i} {{")
        lines.append(f"    id{i}: number;")
        lines.append(f"    name{i}: string;")
        lines.append(f"    data{i}: any;")
        lines.append("}")
        lines.append("")

    # Generate 50 classes
    for i in range(50):
        lines.append(f"class Class{i} implements Interface{i % 5} {{")
        lines.append(f"    private id{i}: number;")
        lines.append(f"    private name{i}: string;")
        lines.append(f"    private data{i}: any;")
        lines.append("    ")
        lines.append(f"    constructor(id: number, name: string, data: any) {{")
        lines.append(f"        this.id{i} = id;")
        lines.append(f"        this.name{i} = name;")
        lines.append(f"        this.data{i} = data;")
        lines.append("    }")
        lines.append("    ")
        lines.append(f"    public getId{i}(): number {{")
        lines.append(f"        return this.id{i};")
        lines.append("    }")
        lines.append("    ")
        lines.append(f"    public setName{i}(name: string): void {{")
        lines.append(f"        this.name{i} = name;")
        lines.append("    }")
        lines.append("}")
        lines.append("")

    # Generate 50 functions
    for i in range(50):
        lines.append(f"function function{i}(param{i}: Interface{i % 5}): Class{i % 3} {{")
        lines.append(f"    console.log('Processing param{i}:', param{i});")
        lines.append(f"    const instance{i} = new Class{i % 3}({i}, 'test{i}', {{ data: {i} }});")
        lines.append(f"    return instance{i};")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)