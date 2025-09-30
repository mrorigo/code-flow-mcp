import { Router, Request, Response } from 'express';
import { AppDataSource } from '../config/database';
import { Order, OrderStatus } from '../models/Order';
import { Product } from '../models/Product';
import { AuthenticatedRequest } from '../types/api';
import { asyncHandler, ValidationError } from '../middleware/errorHandler';
import { ApiResponse, HttpStatus } from '../types/api';

const router = Router();

// Get all orders for current user
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const user = req.user;
  const {
    status,
    page = 1,
    limit = 10
  } = req.query;

  const orderRepository = AppDataSource.getRepository(Order);
  const queryBuilder = orderRepository.createQueryBuilder('order')
    .leftJoinAndSelect('order.product', 'product')
    .where('order.userId = :userId', { userId: user.id });

  if (status) {
    queryBuilder.andWhere('order.status = :status', { status });
  }

  const pageNum = parseInt(page as string);
  const limitNum = parseInt(limit as string);
  const offset = (pageNum - 1) * limitNum;

  queryBuilder.skip(offset).take(limitNum).orderBy('order.createdAt', 'DESC');

  const [orders, total] = await queryBuilder.getManyAndCount();

  const response: ApiResponse<any> = {
    success: true,
    data: {
      orders: orders.map(o => o.toListItem()),
      pagination: {
        page: pageNum,
        limit: limitNum,
        total,
        totalPages: Math.ceil(total / limitNum),
        hasNext: offset + limitNum < total,
        hasPrev: pageNum > 1
      }
    },
    message: 'Orders retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Get order by ID
router.get('/:id', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;
  const orderId = parseInt(id);
  const user = req.user;

  if (isNaN(orderId)) {
    throw new ValidationError('Invalid order ID');
  }

  const orderRepository = AppDataSource.getRepository(Order);
  const order = await orderRepository.findOne({
    where: { id: orderId, userId: user.id },
    relations: ['product']
  });

  if (!order) {
    throw new ValidationError('Order not found');
  }

  const response: ApiResponse<any> = {
    success: true,
    data: order.toDetailView(),
    message: 'Order retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Create new order
router.post('/', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const user = req.user;
  const { productId, quantity, shippingAddress } = req.body;

  if (!productId || !quantity) {
    throw new ValidationError('Product ID and quantity are required');
  }

  const productRepository = AppDataSource.getRepository(Product);
  const product = await productRepository.findOne({ where: { id: productId } });

  if (!product) {
    throw new ValidationError('Product not found');
  }

  if (!product.canPurchase(quantity)) {
    throw new ValidationError('Product is not available in requested quantity');
  }

  const orderRepository = AppDataSource.getRepository(Order);

  const newOrder = orderRepository.create({
    userId: user.id,
    productId,
    quantity,
    unitPrice: product.finalPrice,
    totalAmount: product.finalPrice * quantity,
    status: OrderStatus.PENDING,
    shippingAddress: shippingAddress || 'Default shipping address',
    metadata: {
      createdBy: user.email,
      productName: product.name
    }
  });

  const savedOrder = await orderRepository.save(newOrder);

  // Reduce product stock
  product.reduceStock(quantity);
  await productRepository.save(product);

  // Load the order with product details
  const orderWithProduct = await orderRepository.findOne({
    where: { id: savedOrder.id },
    relations: ['product']
  });

  const response: ApiResponse<any> = {
    success: true,
    data: orderWithProduct?.toDetailView(),
    message: 'Order created successfully',
    timestamp: new Date().toISOString()
  };

  res.status(HttpStatus.CREATED).json(response);
}));

// Cancel order
router.post('/:id/cancel', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;
  const orderId = parseInt(id);
  const user = req.user;
  const { reason } = req.body;

  if (isNaN(orderId)) {
    throw new ValidationError('Invalid order ID');
  }

  const orderRepository = AppDataSource.getRepository(Order);
  const order = await orderRepository.findOne({
    where: { id: orderId, userId: user.id },
    relations: ['product']
  });

  if (!order) {
    throw new ValidationError('Order not found');
  }

  if (!order.canBeCancelled) {
    throw new ValidationError('Order cannot be cancelled in current status');
  }

  order.cancel(reason);

  // Return stock to product
  if (order.product) {
    order.product.increaseStock(order.quantity);
    await AppDataSource.getRepository(Product).save(order.product);
  }

  await orderRepository.save(order);

  const response: ApiResponse<any> = {
    success: true,
    data: order.toDetailView(),
    message: 'Order cancelled successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Get order statistics for current user
router.get('/stats/summary', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const user = req.user;

  const orderRepository = AppDataSource.getRepository(Order);

  const stats = await orderRepository
    .createQueryBuilder('order')
    .select([
      'COUNT(*) as totalOrders',
      'SUM(CASE WHEN status = :delivered THEN 1 ELSE 0 END) as completedOrders',
      'SUM(CASE WHEN status = :pending THEN 1 ELSE 0 END) as pendingOrders',
      'SUM(totalAmount) as totalSpent'
    ])
    .setParameters({
      delivered: OrderStatus.DELIVERED,
      pending: OrderStatus.PENDING
    })
    .where('order.userId = :userId', { userId: user.id })
    .getRawOne();

  const response: ApiResponse<any> = {
    success: true,
    data: {
      totalOrders: parseInt(stats.totalOrders) || 0,
      completedOrders: parseInt(stats.completedOrders) || 0,
      pendingOrders: parseInt(stats.pendingOrders) || 0,
      totalSpent: parseFloat(stats.totalSpent) || 0
    },
    message: 'Order statistics retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

export default router;