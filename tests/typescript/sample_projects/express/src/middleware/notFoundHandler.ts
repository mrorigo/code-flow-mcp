import { Request, Response, NextFunction } from 'express';
import { AppError, NotFoundError } from './errorHandler';

// 404 handler middleware
export const notFoundHandler = (req: Request, res: Response, next: NextFunction): void => {
  const error = new NotFoundError(`Route ${req.originalUrl} not found`);
  next(error);
};

// API 404 handler for API routes
export const apiNotFoundHandler = (req: Request, res: Response, next: NextFunction): void => {
  if (req.path.startsWith('/api/')) {
    const error = new NotFoundError(`API endpoint ${req.originalUrl} not found`);
    next(error);
  } else {
    next();
  }
};