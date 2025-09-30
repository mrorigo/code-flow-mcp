import { Router, Request, Response } from 'express';
import { AppDataSource } from '../config/database';
import { User } from '../models/User';
import { AuthUtils } from '../middleware/auth';
import { AuthenticatedRequest } from '../types/api';
import { asyncHandler, ValidationError } from '../middleware/errorHandler';
import { ApiResponse, HttpStatus } from '../types/api';

const router = Router();

// Get all users (Admin only)
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const userRepository = AppDataSource.getRepository(User);
  const users = await userRepository.find({
    select: ['id', 'firstName', 'lastName', 'email', 'role', 'isActive', 'createdAt', 'updatedAt']
  });

  const response: ApiResponse<User[]> = {
    success: true,
    data: users,
    message: 'Users retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Get user by ID
router.get('/:id', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;
  const userId = parseInt(id);

  if (isNaN(userId)) {
    throw new ValidationError('Invalid user ID');
  }

  const userRepository = AppDataSource.getRepository(User);
  const user = await userRepository.findOne({
    where: { id: userId },
    select: ['id', 'firstName', 'lastName', 'email', 'role', 'isActive', 'createdAt', 'updatedAt']
  });

  if (!user) {
    throw new ValidationError('User not found');
  }

  const response: ApiResponse<User> = {
    success: true,
    data: user,
    message: 'User retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Update user
router.put('/:id', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;
  const userId = parseInt(id);
  const updates = req.body;

  if (isNaN(userId)) {
    throw new ValidationError('Invalid user ID');
  }

  const userRepository = AppDataSource.getRepository(User);
  const user = await userRepository.findOne({ where: { id: userId } });

  if (!user) {
    throw new ValidationError('User not found');
  }

  // Update allowed fields
  const allowedFields = ['firstName', 'lastName', 'phoneNumber', 'avatar', 'preferences'];
  const filteredUpdates: any = {};

  allowedFields.forEach(field => {
    if (updates[field] !== undefined) {
      filteredUpdates[field] = updates[field];
    }
  });

  // Merge updates
  Object.assign(user, filteredUpdates);
  const updatedUser = await userRepository.save(user);

  const response: ApiResponse<User> = {
    success: true,
    data: updatedUser.toSafeJSON(),
    message: 'User updated successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Deactivate user (Admin only)
router.delete('/:id', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;
  const userId = parseInt(id);

  if (isNaN(userId)) {
    throw new ValidationError('Invalid user ID');
  }

  const userRepository = AppDataSource.getRepository(User);
  const user = await userRepository.findOne({ where: { id: userId } });

  if (!user) {
    throw new ValidationError('User not found');
  }

  user.isActive = false;
  await userRepository.save(user);

  const response: ApiResponse<null> = {
    success: true,
    data: null,
    message: 'User deactivated successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Get current user profile
router.get('/profile/me', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const currentUser = req.user;

  const userRepository = AppDataSource.getRepository(User);
  const user = await userRepository.findOne({
    where: { id: currentUser.id }
  });

  if (!user) {
    throw new ValidationError('User not found');
  }

  const response: ApiResponse<any> = {
    success: true,
    data: {
      id: user.id,
      firstName: user.firstName,
      lastName: user.lastName,
      email: user.email,
      role: user.role,
      isActive: user.isActive,
      phoneNumber: user.phoneNumber,
      avatar: user.avatar,
      preferences: user.preferences,
      createdAt: user.createdAt,
      updatedAt: user.updatedAt
    },
    message: 'Profile retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Update current user profile
router.put('/profile/me', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const currentUser = req.user;
  const updates = req.body;

  const userRepository = AppDataSource.getRepository(User);
  const user = await userRepository.findOne({ where: { id: currentUser.id } });

  if (!user) {
    throw new ValidationError('User not found');
  }

  // Update allowed fields
  const allowedFields = ['firstName', 'lastName', 'phoneNumber', 'avatar', 'preferences'];
  allowedFields.forEach(field => {
    if (updates[field] !== undefined) {
      (user as any)[field] = updates[field];
    }
  });

  const updatedUser = await userRepository.save(user);

  const response: ApiResponse<any> = {
    success: true,
    data: {
      id: updatedUser.id,
      firstName: updatedUser.firstName,
      lastName: updatedUser.lastName,
      email: updatedUser.email,
      role: updatedUser.role,
      isActive: updatedUser.isActive,
      phoneNumber: updatedUser.phoneNumber,
      avatar: updatedUser.avatar,
      preferences: updatedUser.preferences,
      createdAt: updatedUser.createdAt,
      updatedAt: updatedUser.updatedAt
    },
    message: 'Profile updated successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

export default router;