import { Request, Response, NextFunction } from 'express';
import { ApiError, HttpStatus } from '../types/api';

// Custom error class for API errors
export class AppError extends Error {
  public statusCode: number;
  public isOperational: boolean;
  public errorCode?: string;
  public details?: any;

  constructor(
    message: string,
    statusCode: number = HttpStatus.INTERNAL_SERVER_ERROR,
    errorCode?: string,
    details?: any,
    isOperational: boolean = true
  ) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    this.errorCode = errorCode;
    this.details = details;

    Error.captureStackTrace(this, this.constructor);
  }
}

// Validation error class
export class ValidationError extends AppError {
  constructor(message: string, details?: any) {
    super(message, HttpStatus.UNPROCESSABLE_ENTITY, 'VALIDATION_ERROR', details);
  }
}

// Not found error class
export class NotFoundError extends AppError {
  constructor(resource: string = 'Resource') {
    super(`${resource} not found`, HttpStatus.NOT_FOUND, 'NOT_FOUND');
  }
}

// Unauthorized error class
export class UnauthorizedError extends AppError {
  constructor(message: string = 'Unauthorized access') {
    super(message, HttpStatus.UNAUTHORIZED, 'UNAUTHORIZED');
  }
}

// Forbidden error class
export class ForbiddenError extends AppError {
  constructor(message: string = 'Access forbidden') {
    super(message, HttpStatus.FORBIDDEN, 'FORBIDDEN');
  }
}

// Conflict error class
export class ConflictError extends AppError {
  constructor(message: string, details?: any) {
    super(message, HttpStatus.CONFLICT, 'CONFLICT', details);
  }
}

// Global error handler middleware
export const errorHandler = (
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  let statusCode = HttpStatus.INTERNAL_SERVER_ERROR;
  let message = 'Internal server error';
  let errorCode = 'INTERNAL_ERROR';
  let details = undefined;

  // Handle known error types
  if (error instanceof AppError) {
    statusCode = error.statusCode;
    message = error.message;
    errorCode = error.errorCode || errorCode;
    details = error.details;
  } else if (error.name === 'ValidationError') {
    // Handle TypeORM validation errors
    statusCode = HttpStatus.UNPROCESSABLE_ENTITY;
    errorCode = 'DATABASE_VALIDATION_ERROR';
    message = 'Database validation failed';
    details = error;
  } else if (error.name === 'QueryFailedError') {
    // Handle TypeORM query errors
    statusCode = HttpStatus.BAD_REQUEST;
    errorCode = 'DATABASE_QUERY_ERROR';
    message = 'Database operation failed';
    details = process.env.NODE_ENV === 'development' ? error.message : undefined;
  } else if (error.name === 'JsonWebTokenError') {
    // Handle JWT errors
    statusCode = HttpStatus.UNAUTHORIZED;
    errorCode = 'INVALID_TOKEN';
    message = 'Invalid authentication token';
  } else if (error.name === 'TokenExpiredError') {
    // Handle expired JWT tokens
    statusCode = HttpStatus.UNAUTHORIZED;
    errorCode = 'TOKEN_EXPIRED';
    message = 'Authentication token has expired';
  } else if (error.message?.includes('ENOENT')) {
    // Handle file not found errors
    statusCode = HttpStatus.NOT_FOUND;
    errorCode = 'FILE_NOT_FOUND';
    message = 'File not found';
  } else if (error.message?.includes('EACCES')) {
    // Handle permission errors
    statusCode = HttpStatus.FORBIDDEN;
    errorCode = 'PERMISSION_DENIED';
    message = 'Permission denied';
  }

  // Log error for debugging (only in development or for operational errors)
  if (process.env.NODE_ENV === 'development' || (error instanceof AppError && error.isOperational)) {
    console.error('Error occurred:', {
      name: error.name,
      message: error.message,
      statusCode,
      errorCode,
      stack: error.stack,
      url: req.url,
      method: req.method,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      timestamp: new Date().toISOString()
    });
  }

  // Don't expose internal errors in production
  if (statusCode === HttpStatus.INTERNAL_SERVER_ERROR && process.env.NODE_ENV === 'production') {
    message = 'Something went wrong';
    details = undefined;
  }

  // Prepare error response
  const errorResponse: ApiError = {
    success: false,
    error: message,
    message: statusCode >= 500 ? 'Please try again later' : message,
    code: errorCode,
    details: details,
    timestamp: new Date().toISOString()
  };

  // Handle specific error types with additional headers or special handling
  if (error instanceof UnauthorizedError) {
    res.set('WWW-Authenticate', 'Bearer');
  }

  res.status(statusCode).json(errorResponse);
};

// 404 handler middleware
export const notFoundHandler = (req: Request, res: Response, next: NextFunction): void => {
  const error = new NotFoundError(`Route ${req.originalUrl} not found`);
  next(error);
};

// Async error wrapper
export const asyncHandler = (fn: Function) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

// Validation error formatter
export const formatValidationErrors = (errors: any[]): any => {
  return errors.map(error => ({
    field: error.property,
    value: error.value,
    constraints: error.constraints,
    children: error.children?.length ? formatValidationErrors(error.children) : undefined
  }));
};

// Request timeout handler
export const handleTimeout = (req: Request, res: Response, next: NextFunction): void => {
  const timeout = setTimeout(() => {
    const error = new AppError(
      'Request timeout',
      HttpStatus.REQUEST_TIMEOUT,
      'REQUEST_TIMEOUT'
    );
    next(error);
  }, 30000); // 30 seconds

  res.on('finish', () => {
    clearTimeout(timeout);
  });

  next();
};

// Development error handler (with more detailed error information)
export const developmentErrorHandler = (
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const statusCode = error instanceof AppError ? error.statusCode : HttpStatus.INTERNAL_SERVER_ERROR;
  const message = error.message;
  const errorCode = error instanceof AppError ? error.errorCode : 'INTERNAL_ERROR';

  console.error('Development Error:', {
    name: error.name,
    message: error.message,
    stack: error.stack,
    statusCode,
    errorCode,
    body: req.body,
    params: req.params,
    query: req.query,
    headers: req.headers,
    url: req.url,
    method: req.method
  });

  const errorResponse: ApiError = {
    success: false,
    error: message,
    message: message,
    code: errorCode,
    details: {
      stack: error.stack,
      body: req.body,
      params: req.params,
      query: req.query
    },
    timestamp: new Date().toISOString()
  };

  res.status(statusCode).json(errorResponse);
};