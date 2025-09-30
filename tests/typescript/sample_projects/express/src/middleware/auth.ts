import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import { AuthenticatedRequest, UserRole, Permission, ApiError, HttpStatus } from '../types/api';
import { AppDataSource } from '../config/database';
import { User } from '../models/User';

// JWT configuration
const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-jwt-key-change-in-production';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '24h';
const REFRESH_TOKEN_SECRET = process.env.REFRESH_TOKEN_SECRET || 'your-refresh-token-secret';
const REFRESH_TOKEN_EXPIRES_IN = process.env.REFRESH_TOKEN_EXPIRES_IN || '7d';

// Token payload interface
interface JwtPayload {
  userId: number;
  email: string;
  role: UserRole;
  permissions: Permission[];
  iat?: number;
  exp?: number;
}

interface RefreshTokenPayload {
  userId: number;
  tokenVersion: number;
  iat?: number;
  exp?: number;
}

// Authentication middleware
export const authMiddleware = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.startsWith('Bearer ')
      ? authHeader.slice(7)
      : null;

    if (!token) {
      const error: ApiError = {
        success: false,
        error: 'Access token is required',
        message: 'Please provide a valid access token',
        code: 'AUTH_TOKEN_MISSING',
        timestamp: new Date().toISOString()
      };
      res.status(HttpStatus.UNAUTHORIZED).json(error);
      return;
    }

    try {
      // Verify JWT token
      const decoded = jwt.verify(token, JWT_SECRET) as JwtPayload;

      // Get user from database to ensure they still exist and are active
      const userRepository = AppDataSource.getRepository(User);
      const user = await userRepository.findOne({
        where: { id: decoded.userId, isActive: true }
      });

      if (!user) {
        const error: ApiError = {
          success: false,
          error: 'User not found or inactive',
          message: 'Please log in again',
          code: 'USER_NOT_FOUND',
          timestamp: new Date().toISOString()
        };
        res.status(HttpStatus.UNAUTHORIZED).json(error);
        return;
      }

      // Add user to request object
      (req as AuthenticatedRequest).user = {
        id: user.id,
        email: user.email,
        role: user.role,
        permissions: user.permissions
      };

      next();
    } catch (jwtError: any) {
      const error: ApiError = {
        success: false,
        error: 'Invalid or expired token',
        message: 'Please log in again',
        code: 'AUTH_TOKEN_INVALID',
        timestamp: new Date().toISOString()
      };
      res.status(HttpStatus.UNAUTHORIZED).json(error);
    }
  } catch (error) {
    console.error('Authentication middleware error:', error);
    const apiError: ApiError = {
      success: false,
      error: 'Authentication failed',
      message: 'Internal server error during authentication',
      code: 'AUTH_INTERNAL_ERROR',
      timestamp: new Date().toISOString()
    };
    res.status(HttpStatus.INTERNAL_SERVER_ERROR).json(apiError);
  }
};

// Role-based authorization middleware
export const requireRole = (roles: UserRole | UserRole[]) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const user = (req as AuthenticatedRequest).user;
    const allowedRoles = Array.isArray(roles) ? roles : [roles];

    if (!user || !allowedRoles.includes(user.role)) {
      const error: ApiError = {
        success: false,
        error: 'Insufficient permissions',
        message: `Required roles: ${allowedRoles.join(', ')}`,
        code: 'INSUFFICIENT_ROLE',
        timestamp: new Date().toISOString()
      };
      res.status(HttpStatus.FORBIDDEN).json(error);
      return;
    }

    next();
  };
};

// Permission-based authorization middleware
export const requirePermission = (permission: Permission | Permission[]) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const user = (req as AuthenticatedRequest).user;
    const requiredPermissions = Array.isArray(permission) ? permission : [permission];

    if (!user) {
      const error: ApiError = {
        success: false,
        error: 'Authentication required',
        message: 'Please log in to access this resource',
        code: 'AUTH_REQUIRED',
        timestamp: new Date().toISOString()
      };
      res.status(HttpStatus.UNAUTHORIZED).json(error);
      return;
    }

    const hasPermission = requiredPermissions.every(p => user.permissions.includes(p));

    if (!hasPermission) {
      const error: ApiError = {
        success: false,
        error: 'Insufficient permissions',
        message: `Required permissions: ${requiredPermissions.join(', ')}`,
        code: 'INSUFFICIENT_PERMISSION',
        timestamp: new Date().toISOString()
      };
      res.status(HttpStatus.FORBIDDEN).json(error);
      return;
    }

    next();
  };
};

// Admin-only middleware (convenience wrapper)
export const requireAdmin = requireRole(UserRole.ADMIN);

// Optional authentication middleware (doesn't fail if no token provided)
export const optionalAuth = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.startsWith('Bearer ')
      ? authHeader.slice(7)
      : null;

    if (token) {
      try {
        const decoded = jwt.verify(token, JWT_SECRET) as JwtPayload;
        const userRepository = AppDataSource.getRepository(User);
        const user = await userRepository.findOne({
          where: { id: decoded.userId, isActive: true }
        });

        if (user) {
          (req as AuthenticatedRequest).user = {
            id: user.id,
            email: user.email,
            role: user.role,
            permissions: user.permissions
          };
        }
      } catch (error) {
        // Ignore JWT errors in optional auth
      }
    }

    next();
  } catch (error) {
    next(); // Continue without authentication
  }
};

// Resource ownership middleware
export const requireOwnership = (userIdField: string = 'userId') => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const user = (req as AuthenticatedRequest).user;

    if (!user) {
      const error: ApiError = {
        success: false,
        error: 'Authentication required',
        message: 'Please log in to access this resource',
        code: 'AUTH_REQUIRED',
        timestamp: new Date().toISOString()
      };
      res.status(HttpStatus.UNAUTHORIZED).json(error);
      return;
    }

    const resourceUserId = (req as any).params[userIdField] || (req as any).body[userIdField];

    if (!resourceUserId) {
      const error: ApiError = {
        success: false,
        error: 'Resource user ID not found',
        message: `Cannot determine resource ownership`,
        code: 'RESOURCE_USER_ID_MISSING',
        timestamp: new Date().toISOString()
      };
      res.status(HttpStatus.BAD_REQUEST).json(error);
      return;
    }

    // Admins can access any resource
    if (user.role === UserRole.ADMIN) {
      next();
      return;
    }

    // Users can only access their own resources
    if (user.id !== parseInt(resourceUserId)) {
      const error: ApiError = {
        success: false,
        error: 'Access denied',
        message: 'You can only access your own resources',
        code: 'ACCESS_DENIED',
        timestamp: new Date().toISOString()
      };
      res.status(HttpStatus.FORBIDDEN).json(error);
      return;
    }

    next();
  };
};

// Authentication utility functions
export class AuthUtils {
  static async hashPassword(password: string): Promise<string> {
    const saltRounds = 12;
    return bcrypt.hash(password, saltRounds);
  }

  static async verifyPassword(password: string, hashedPassword: string): Promise<boolean> {
    return bcrypt.compare(password, hashedPassword);
  }

  static generateAccessToken(user: User): string {
    const payload: JwtPayload = {
      userId: user.id,
      email: user.email,
      role: user.role,
      permissions: user.permissions
    };

    return jwt.sign(payload, JWT_SECRET, {
      expiresIn: JWT_EXPIRES_IN,
      issuer: 'express-api',
      audience: 'express-api-users'
    });
  }

  static generateRefreshToken(user: User, tokenVersion: number = 1): string {
    const payload: RefreshTokenPayload = {
      userId: user.id,
      tokenVersion
    };

    return jwt.sign(payload, REFRESH_TOKEN_SECRET, {
      expiresIn: REFRESH_TOKEN_EXPIRES_IN,
      issuer: 'express-api',
      audience: 'express-api-users'
    });
  }

  static verifyRefreshToken(token: string): RefreshTokenPayload | null {
    try {
      return jwt.verify(token, REFRESH_TOKEN_SECRET) as RefreshTokenPayload;
    } catch (error) {
      return null;
    }
  }

  static extractTokenFromHeader(authHeader: string | undefined): string | null {
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return null;
    }
    return authHeader.slice(7);
  }
}