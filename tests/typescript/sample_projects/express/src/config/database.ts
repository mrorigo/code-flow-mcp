import { DataSource } from 'typeorm';
import { User } from '../models/User';
import { Product } from '../models/Product';
import { Order } from '../models/Order';

// Database configuration interface
export interface DatabaseConfig {
  host: string;
  port: number;
  username: string;
  password: string;
  database: string;
  synchronize: boolean;
  logging: boolean;
}

// Default configuration
const defaultConfig: DatabaseConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432'),
  username: process.env.DB_USERNAME || 'postgres',
  password: process.env.DB_PASSWORD || 'password',
  database: process.env.DB_NAME || 'express_api',
  synchronize: process.env.NODE_ENV !== 'production',
  logging: process.env.NODE_ENV === 'development'
};

// Create TypeORM DataSource
export const AppDataSource = new DataSource({
  type: 'sqlite', // Using SQLite for simplicity in sample project
  database: process.env.DB_NAME || 'express_api.db',
  synchronize: defaultConfig.synchronize,
  logging: defaultConfig.logging,
  entities: [User, Product, Order],
  migrations: ['src/migrations/*.ts'],
  subscribers: ['src/subscribers/*.ts'],
  cache: {
    type: 'redis',
    options: {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      password: process.env.REDIS_PASSWORD
    },
    duration: 60000 // 1 minute
  },
  ssl: process.env.NODE_ENV === 'production' ? {
    rejectUnauthorized: false
  } : false
});

// Database connection health check
export async function checkDatabaseConnection(): Promise<boolean> {
  try {
    await AppDataSource.initialize();
    console.log('Database connection established successfully');
    return true;
  } catch (error) {
    console.error('Database connection failed:', error);
    return false;
  }
}

// Initialize database with sample data
export async function initializeSampleData(): Promise<void> {
  try {
    // Check if we already have data
    const userRepository = AppDataSource.getRepository(User);
    const existingUsers = await userRepository.count();

    if (existingUsers > 0) {
      console.log('Sample data already exists, skipping initialization');
      return;
    }

    console.log('Initializing sample data...');

    // Create sample users
    const adminUser = userRepository.create({
      firstName: 'Admin',
      lastName: 'User',
      email: 'admin@example.com',
      password: 'hashed_password_here', // In real app, this would be hashed
      role: 'admin' as any,
      isActive: true,
      preferences: {
        theme: 'dark',
        notifications: true
      }
    });

    const regularUser = userRepository.create({
      firstName: 'John',
      lastName: 'Doe',
      email: 'john@example.com',
      password: 'hashed_password_here',
      role: 'user' as any,
      isActive: true,
      phoneNumber: '+1234567890',
      preferences: {
        theme: 'light',
        notifications: true
      }
    });

    await userRepository.save([adminUser, regularUser]);

    // Create sample products
    const productRepository = AppDataSource.getRepository(Product);
    const sampleProducts = [
      {
        name: 'Wireless Bluetooth Headphones',
        description: 'High-quality wireless headphones with noise cancellation',
        price: 199.99,
        stockQuantity: 50,
        category: 'electronics' as any,
        status: 'active' as any,
        imageUrl: 'https://example.com/headphones.jpg',
        specifications: {
          batteryLife: '30 hours',
          connectivity: 'Bluetooth 5.0',
          weight: '250g'
        },
        tags: ['wireless', 'bluetooth', 'audio'],
        discountPercentage: 0.1 // 10% discount
      },
      {
        name: 'Organic Cotton T-Shirt',
        description: 'Comfortable and sustainable cotton t-shirt',
        price: 29.99,
        stockQuantity: 100,
        category: 'clothing' as any,
        status: 'active' as any,
        imageUrl: 'https://example.com/tshirt.jpg',
        specifications: {
          material: '100% organic cotton',
          sizes: ['S', 'M', 'L', 'XL'],
          color: 'White'
        },
        tags: ['organic', 'cotton', 'clothing']
      },
      {
        name: 'TypeScript Programming Book',
        description: 'Comprehensive guide to TypeScript programming',
        price: 49.99,
        stockQuantity: 25,
        category: 'books' as any,
        status: 'active' as any,
        imageUrl: 'https://example.com/typescript-book.jpg',
        specifications: {
          pages: 450,
          language: 'English',
          level: 'Intermediate to Advanced'
        },
        tags: ['programming', 'typescript', 'book']
      }
    ];

    const products = productRepository.create(sampleProducts);
    await productRepository.save(products);

    // Create sample orders
    const orderRepository = AppDataSource.getRepository(Order);
    const sampleOrders = [
      {
        userId: regularUser.id,
        productId: products[0].id,
        quantity: 2,
        unitPrice: products[0].finalPrice,
        totalAmount: products[0].finalPrice * 2,
        status: 'delivered' as any,
        paymentStatus: 'completed' as any,
        shippingAddress: '123 Main St, Anytown, AT 12345',
        trackingNumber: 'TRK123456789',
        deliveredAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
        metadata: {
          paymentMethod: 'credit_card',
          orderSource: 'website'
        }
      },
      {
        userId: regularUser.id,
        productId: products[1].id,
        quantity: 1,
        unitPrice: products[1].finalPrice,
        totalAmount: products[1].finalPrice,
        status: 'shipped' as any,
        paymentStatus: 'completed' as any,
        shippingAddress: '123 Main St, Anytown, AT 12345',
        trackingNumber: 'TRK987654321',
        shippedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
        metadata: {
          paymentMethod: 'paypal',
          orderSource: 'mobile_app'
        }
      },
      {
        userId: regularUser.id,
        productId: products[2].id,
        quantity: 1,
        unitPrice: products[2].finalPrice,
        totalAmount: products[2].finalPrice,
        status: 'pending' as any,
        paymentStatus: 'pending' as any,
        shippingAddress: '123 Main St, Anytown, AT 12345',
        metadata: {
          paymentMethod: 'bank_transfer',
          orderSource: 'website'
        }
      }
    ];

    const orders = orderRepository.create(sampleOrders);
    await orderRepository.save(orders);

    console.log('Sample data initialized successfully');
    console.log(`Created ${await userRepository.count()} users`);
    console.log(`Created ${await productRepository.count()} products`);
    console.log(`Created ${await orderRepository.count()} orders`);

  } catch (error) {
    console.error('Failed to initialize sample data:', error);
    throw error;
  }
}

// Database utilities
export class DatabaseUtils {
  static async clearAllTables(): Promise<void> {
    try {
      await AppDataSource.query('DELETE FROM orders');
      await AppDataSource.query('DELETE FROM products');
      await AppDataSource.query('DELETE FROM users');
      console.log('All tables cleared');
    } catch (error) {
      console.error('Failed to clear tables:', error);
      throw error;
    }
  }

  static async getDatabaseStats(): Promise<{
    users: number;
    products: number;
    orders: number;
  }> {
    const userRepository = AppDataSource.getRepository(User);
    const productRepository = AppDataSource.getRepository(Product);
    const orderRepository = AppDataSource.getRepository(Order);

    return {
      users: await userRepository.count(),
      products: await productRepository.count(),
      orders: await orderRepository.count()
    };
  }
}