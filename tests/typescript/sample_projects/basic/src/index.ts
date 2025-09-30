// Basic TypeScript project for integration testing

export interface User {
    id: number;
    name: string;
    email: string;
    createdAt: Date;
}

export interface Product {
    id: number;
    name: string;
    price: number;
    category: string;
}

export class Database {
    private users: User[] = [];
    private products: Product[] = [];

    public async findUser(id: number): Promise<User | null> {
        const user = this.users.find(u => u.id === id);
        return user || null;
    }

    public async findUsersByName(name: string): Promise<User[]> {
        return this.users.filter(u => u.name.includes(name));
    }

    public async createUser(userData: Omit<User, 'id' | 'createdAt'>): Promise<User> {
        const newUser: User = {
            id: Date.now(),
            createdAt: new Date(),
            ...userData
        };

        this.users.push(newUser);
        return newUser;
    }

    public async findProduct(id: number): Promise<Product | null> {
        const product = this.products.find(p => p.id === id);
        return product || null;
    }

    public async createProduct(productData: Omit<Product, 'id'>): Promise<Product> {
        const newProduct: Product = {
            id: Date.now(),
            ...productData
        };

        this.products.push(newProduct);
        return newProduct;
    }
}

export class UserService {
    constructor(private database: Database) {}

    public async getUserById(id: number): Promise<User | null> {
        return this.database.findUser(id);
    }

    public async searchUsers(query: string): Promise<User[]> {
        return this.database.findUsersByName(query);
    }

    public async registerUser(userData: Omit<User, 'id' | 'createdAt'>): Promise<User> {
        // Validate user data
        if (!userData.name || !userData.email) {
            throw new Error('Name and email are required');
        }

        // Check if user already exists
        const existingUsers = await this.database.findUsersByName(userData.name);
        if (existingUsers.some(u => u.email === userData.email)) {
            throw new Error('User with this email already exists');
        }

        return this.database.createUser(userData);
    }
}

export class ProductService {
    constructor(private database: Database) {}

    public async getProductById(id: number): Promise<Product | null> {
        return this.database.findProduct(id);
    }

    public async createProduct(productData: Omit<Product, 'id'>): Promise<Product> {
        if (!productData.name || productData.price <= 0) {
            throw new Error('Valid name and positive price are required');
        }

        return this.database.createProduct(productData);
    }
}

export class Application {
    private userService: UserService;
    private productService: ProductService;

    constructor() {
        const database = new Database();
        this.userService = new UserService(database);
        this.productService = new ProductService(database);
    }

    public async initialize(): Promise<void> {
        console.log('Application initializing...');

        // Create sample data
        await this.createSampleData();

        console.log('Application initialized successfully');
    }

    private async createSampleData(): Promise<void> {
        try {
            // Create sample users
            await this.userService.registerUser({
                name: 'John Doe',
                email: 'john@example.com'
            });

            await this.userService.registerUser({
                name: 'Jane Smith',
                email: 'jane@example.com'
            });

            // Create sample products
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

            console.log('Sample data created');
        } catch (error) {
            console.error('Error creating sample data:', error);
        }
    }

    public getUserService(): UserService {
        return this.userService;
    }

    public getProductService(): ProductService {
        return this.productService;
    }
}

// CLI entry point
export async function main(): Promise<void> {
    const app = new Application();

    try {
        await app.initialize();

        const userService = app.getUserService();
        const users = await userService.searchUsers('John');

        console.log('Found users:', users);

        const productService = app.getProductService();
        const products = await Promise.all([
            productService.getProductById(1),
            productService.getProductById(2)
        ]);

        console.log('Found products:', products);
    } catch (error) {
        console.error('Application error:', error);
        process.exit(1);
    }
}

// Run the application if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}