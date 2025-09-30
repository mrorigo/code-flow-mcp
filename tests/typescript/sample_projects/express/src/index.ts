import express, { Application, Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';

import { errorHandler } from './middleware/errorHandler';
import { notFoundHandler } from './middleware/notFoundHandler';
import { AppDataSource } from './config/database';
import { authMiddleware } from './middleware/auth';

import userRoutes from './routes/userRoutes';
import productRoutes from './routes/productRoutes';
import orderRoutes from './routes/orderRoutes';
import authRoutes from './routes/authRoutes';

import { ApiResponse, ApiError } from './types/api';
import { User } from './models/User';
import { Product } from './models/Product';
import { Order } from './models/Order';

class ExpressApplication {
  private app: Application;
  private port: number;

  constructor() {
    this.app = express();
    this.port = process.env.PORT ? parseInt(process.env.PORT) : 3000;
    this.configureMiddleware();
    this.configureRoutes();
    this.configureErrorHandling();
  }

  private configureMiddleware(): void {
    // Security middleware
    this.app.use(helmet());

    // Rate limiting
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: {
        success: false,
        error: 'Too many requests from this IP, please try again later.'
      } as any
    });
    this.app.use('/api/', limiter);

    // Body parsing middleware
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));

    // CORS configuration
    this.app.use(cors({
      origin: process.env.CORS_ORIGIN?.split(',') || ['http://localhost:3000'],
      credentials: true
    }));

    // Compression middleware
    this.app.use(compression());

    // Logging middleware
    this.app.use(morgan('combined'));
  }

  private configureRoutes(): void {
    // Health check endpoint
    this.app.get('/health', (req: Request, res: Response) => {
      const response: ApiResponse<string> = {
        success: true,
        data: 'Server is running',
        timestamp: new Date().toISOString()
      };
      res.json(response);
    });

    // API routes
    this.app.use('/api/auth', authRoutes);
    this.app.use('/api/users', authMiddleware, userRoutes);
    this.app.use('/api/products', productRoutes);
    this.app.use('/api/orders', authMiddleware, orderRoutes);

    // API info endpoint
    this.app.get('/api', (req: Request, res: Response) => {
      const response: ApiResponse<any> = {
        success: true,
        data: {
          name: 'Express TypeScript API',
          version: '1.0.0',
          description: 'RESTful API with TypeScript, Express.js, and TypeORM',
          endpoints: {
            auth: '/api/auth',
            users: '/api/users',
            products: '/api/products',
            orders: '/api/orders'
          }
        },
        timestamp: new Date().toISOString()
      };
      res.json(response);
    });
  }

  private configureErrorHandling(): void {
    // 404 handler
    this.app.use(notFoundHandler);

    // Global error handler
    this.app.use(errorHandler);
  }

  public async initialize(): Promise<void> {
    try {
      // Initialize database connection
      await AppDataSource.initialize();
      console.log('Database connection established successfully');

      // Start the server
      this.app.listen(this.port, () => {
        console.log(`Server is running on port ${this.port}`);
        console.log(`Health check available at http://localhost:${this.port}/health`);
        console.log(`API info available at http://localhost:${this.port}/api`);
      });
    } catch (error) {
      console.error('Failed to initialize application:', error);
      process.exit(1);
    }
  }

  public getApp(): Application {
    return this.app;
  }
}

// Create and start the application
const application = new ExpressApplication();

// Handle graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully');
  await AppDataSource.destroy();
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('SIGINT received, shutting down gracefully');
  await AppDataSource.destroy();
  process.exit(0);
});

// Start the application if this file is run directly
if (require.main === module) {
  application.initialize().catch(console.error);
}

export default application;