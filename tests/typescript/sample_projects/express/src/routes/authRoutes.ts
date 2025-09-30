import { Router, Request, Response } from 'express';
import { AppDataSource } from '../config/database';
import { User } from '../models/User';
import { AuthUtils } from '../middleware/auth';
import { AuthenticatedRequest } from '../types/api';
import { asyncHandler, ValidationError } from '../middleware/errorHandler';
import { ApiResponse, HttpStatus } from '../types/api';

const router = Router();

// Validation rules
const registerValidation = [
  body('firstName').trim().isLength({ min: 1 }).withMessage('First name is required'),
  body('lastName').trim().isLength({ min: 1 }).withMessage('Last name is required'),
  body('email').isEmail().normalizeEmail().withMessage('Valid email is required'),
  body('password').isLength({ min: 8 }).withMessage('Password must be at least 8 characters'),
  body('role').optional().isIn(['user', 'admin', 'moderator']).withMessage('Invalid role')
];

const loginValidation = [
  body('email').isEmail().normalizeEmail().withMessage('Valid email is required'),
  body('password').notEmpty().withMessage('Password is required')
];

// Register endpoint
router.post('/register', registerValidation, asyncHandler(async (req: Request, res: Response) => {
  // Check validation results
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    throw new ValidationError('Validation failed', errors.array());
  }

  const { firstName, lastName, email, password, role = 'user' } = req.body;

  const userRepository = AppDataSource.getRepository(User);

  // Check if user already exists
  const existingUser = await userRepository.findOne({ where: { email } });
  if (existingUser) {
    throw new ValidationError('User already exists with this email');
  }

  // Hash password
  const hashedPassword = await AuthUtils.hashPassword(password);

  // Create new user
  const newUser = userRepository.create({
    firstName,
    lastName,
    email,
    password: hashedPassword,
    role: role as any
  });

  const savedUser = await userRepository.save(newUser);

  // Generate tokens
  const accessToken = AuthUtils.generateAccessToken(savedUser);
  const refreshToken = AuthUtils.generateRefreshToken(savedUser);

  const response: ApiResponse<any> = {
    success: true,
    data: {
      user: savedUser.toSafeJSON(),
      accessToken,
      refreshToken
    },
    message: 'User registered successfully',
    timestamp: new Date().toISOString()
  };

  res.status(HttpStatus.CREATED).json(response);
}));

// Login endpoint
router.post('/login', loginValidation, asyncHandler(async (req: Request, res: Response) => {
  // Check validation results
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    throw new ValidationError('Validation failed', errors.array());
  }

  const { email, password } = req.body;

  const userRepository = AppDataSource.getRepository(User);

  // Find user by email
  const user = await userRepository.findOne({
    where: { email, isActive: true }
  });

  if (!user) {
    throw new ValidationError('Invalid email or password');
  }

  // Verify password
  const isValidPassword = await AuthUtils.verifyPassword(password, user.password);
  if (!isValidPassword) {
    throw new ValidationError('Invalid email or password');
  }

  // Generate tokens
  const accessToken = AuthUtils.generateAccessToken(user);
  const refreshToken = AuthUtils.generateRefreshToken(user);

  const response: ApiResponse<any> = {
    success: true,
    data: {
      user: user.toSafeJSON(),
      accessToken,
      refreshToken
    },
    message: 'Login successful',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Refresh token endpoint
router.post('/refresh', asyncHandler(async (req: Request, res: Response) => {
  const { refreshToken } = req.body;

  if (!refreshToken) {
    throw new ValidationError('Refresh token is required');
  }

  // Verify refresh token
  const decoded = AuthUtils.verifyRefreshToken(refreshToken);
  if (!decoded) {
    throw new ValidationError('Invalid refresh token');
  }

  // Get user from database
  const userRepository = AppDataSource.getRepository(User);
  const user = await userRepository.findOne({
    where: { id: decoded.userId, isActive: true }
  });

  if (!user) {
    throw new ValidationError('User not found');
  }

  // Generate new tokens
  const newAccessToken = AuthUtils.generateAccessToken(user);
  const newRefreshToken = AuthUtils.generateRefreshToken(user, decoded.tokenVersion + 1);

  const response: ApiResponse<any> = {
    success: true,
    data: {
      accessToken: newAccessToken,
      refreshToken: newRefreshToken
    },
    message: 'Tokens refreshed successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Logout endpoint
router.post('/logout', asyncHandler(async (req: Request, res: Response) => {
  // In a stateless JWT implementation, logout is typically handled on the client-side
  // by removing the tokens from storage. However, you could implement token blacklisting here.

  const response: ApiResponse<null> = {
    success: true,
    data: null,
    message: 'Logout successful',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Get current user profile
router.get('/me', asyncHandler(async (req: Request, res: Response) => {
  const user = (req as AuthenticatedRequest).user;

  const userRepository = AppDataSource.getRepository(User);
  const currentUser = await userRepository.findOne({
    where: { id: user.id }
  });

  if (!currentUser) {
    throw new ValidationError('User not found');
  }

  const response: ApiResponse<any> = {
    success: true,
    data: currentUser.toSafeJSON(),
    message: 'Profile retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Change password
router.post('/change-password', asyncHandler(async (req: Request, res: Response) => {
  const user = (req as AuthenticatedRequest).user;
  const { currentPassword, newPassword } = req.body;

  if (!currentPassword || !newPassword) {
    throw new ValidationError('Current password and new password are required');
  }

  const userRepository = AppDataSource.getRepository(User);
  const currentUser = await userRepository.findOne({
    where: { id: user.id }
  });

  if (!currentUser) {
    throw new ValidationError('User not found');
  }

  // Verify current password
  const isValidPassword = await AuthUtils.verifyPassword(currentPassword, currentUser.password);
  if (!isValidPassword) {
    throw new ValidationError('Current password is incorrect');
  }

  // Hash new password
  const hashedNewPassword = await AuthUtils.hashPassword(newPassword);

  // Update password
  currentUser.password = hashedNewPassword;
  await userRepository.save(currentUser);

  const response: ApiResponse<null> = {
    success: true,
    data: null,
    message: 'Password changed successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

export default router;