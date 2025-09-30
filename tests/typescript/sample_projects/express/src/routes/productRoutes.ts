import { Router, Request, Response } from 'express';
import { AppDataSource } from '../config/database';
import { Product, ProductStatus, ProductCategory } from '../models/Product';
import { AuthenticatedRequest } from '../types/api';
import { asyncHandler, ValidationError } from '../middleware/errorHandler';
import { ApiResponse, HttpStatus } from '../types/api';

const router = Router();

// Get all products with optional filtering
router.get('/', asyncHandler(async (req: Request, res: Response) => {
  const {
    category,
    status,
    minPrice,
    maxPrice,
    search,
    page = 1,
    limit = 10,
    sortBy = 'createdAt',
    sortOrder = 'DESC'
  } = req.query;

  const productRepository = AppDataSource.getRepository(Product);
  const queryBuilder = productRepository.createQueryBuilder('product');

  // Apply filters
  if (category) {
    queryBuilder.andWhere('product.category = :category', { category });
  }

  if (status) {
    queryBuilder.andWhere('product.status = :status', { status });
  }

  if (minPrice) {
    queryBuilder.andWhere('product.price >= :minPrice', { minPrice: parseFloat(minPrice as string) });
  }

  if (maxPrice) {
    queryBuilder.andWhere('product.price <= :maxPrice', { maxPrice: parseFloat(maxPrice as string) });
  }

  if (search) {
    queryBuilder.andWhere('(product.name LIKE :search OR product.description LIKE :search)', {
      search: `%${search}%`
    });
  }

  // Apply pagination
  const pageNum = parseInt(page as string);
  const limitNum = parseInt(limit as string);
  const offset = (pageNum - 1) * limitNum;

  queryBuilder.skip(offset).take(limitNum);

  // Apply sorting
  const validSortFields = ['name', 'price', 'createdAt', 'updatedAt', 'stockQuantity'];
  const sortField = validSortFields.includes(sortBy as string) ? sortBy as string : 'createdAt';
  const sortDirection = sortOrder === 'ASC' ? 'ASC' : 'DESC';

  queryBuilder.orderBy(`product.${sortField}`, sortDirection);

  const [products, total] = await queryBuilder.getManyAndCount();

  const response: ApiResponse<any> = {
    success: true,
    data: {
      products: products.map(p => p.toListItem()),
      pagination: {
        page: pageNum,
        limit: limitNum,
        total,
        totalPages: Math.ceil(total / limitNum),
        hasNext: offset + limitNum < total,
        hasPrev: pageNum > 1
      }
    },
    message: 'Products retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Get product by ID
router.get('/:id', asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;
  const productId = parseInt(id);

  if (isNaN(productId)) {
    throw new ValidationError('Invalid product ID');
  }

  const productRepository = AppDataSource.getRepository(Product);
  const product = await productRepository.findOne({
    where: { id: productId }
  });

  if (!product) {
    throw new ValidationError('Product not found');
  }

  // Increment view count
  product.viewCount += 1;
  await productRepository.save(product);

  const response: ApiResponse<any> = {
    success: true,
    data: product.toDetailView(),
    message: 'Product retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Create new product (Admin/Moderator only)
router.post('/', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const productData = req.body;

  // Validate required fields
  if (!productData.name || !productData.price) {
    throw new ValidationError('Product name and price are required');
  }

  const productRepository = AppDataSource.getRepository(Product);

  const newProduct = productRepository.create({
    name: productData.name,
    description: productData.description,
    price: parseFloat(productData.price),
    stockQuantity: productData.stockQuantity || 0,
    category: productData.category || ProductCategory.OTHER,
    status: productData.status || ProductStatus.ACTIVE,
    imageUrl: productData.imageUrl,
    specifications: productData.specifications || {},
    tags: productData.tags || [],
    discountPercentage: productData.discountPercentage || 0,
    createdBy: { id: req.user.id } as any
  });

  const savedProduct = await productRepository.save(newProduct);

  const response: ApiResponse<any> = {
    success: true,
    data: savedProduct.toDetailView(),
    message: 'Product created successfully',
    timestamp: new Date().toISOString()
  };

  res.status(HttpStatus.CREATED).json(response);
}));

// Update product (Admin/Moderator only)
router.put('/:id', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;
  const productId = parseInt(id);
  const updates = req.body;

  if (isNaN(productId)) {
    throw new ValidationError('Invalid product ID');
  }

  const productRepository = AppDataSource.getRepository(Product);
  const product = await productRepository.findOne({ where: { id: productId } });

  if (!product) {
    throw new ValidationError('Product not found');
  }

  // Update allowed fields
  const allowedFields = [
    'name', 'description', 'price', 'stockQuantity', 'category',
    'status', 'imageUrl', 'specifications', 'tags', 'discountPercentage'
  ];

  allowedFields.forEach(field => {
    if (updates[field] !== undefined) {
      (product as any)[field] = updates[field];
    }
  });

  const updatedProduct = await productRepository.save(product);

  const response: ApiResponse<any> = {
    success: true,
    data: updatedProduct.toDetailView(),
    message: 'Product updated successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Delete product (Admin only)
router.delete('/:id', asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;
  const productId = parseInt(id);

  if (isNaN(productId)) {
    throw new ValidationError('Invalid product ID');
  }

  const productRepository = AppDataSource.getRepository(Product);
  const product = await productRepository.findOne({ where: { id: productId } });

  if (!product) {
    throw new ValidationError('Product not found');
  }

  await productRepository.remove(product);

  const response: ApiResponse<null> = {
    success: true,
    data: null,
    message: 'Product deleted successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Get product categories
router.get('/categories/list', asyncHandler(async (req: Request, res: Response) => {
  const categories = Object.values(ProductCategory);

  const response: ApiResponse<string[]> = {
    success: true,
    data: categories,
    message: 'Product categories retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

// Search products
router.get('/search/suggestions', asyncHandler(async (req: Request, res: Response) => {
  const { q: query } = req.query;

  if (!query || typeof query !== 'string') {
    throw new ValidationError('Search query is required');
  }

  const productRepository = AppDataSource.getRepository(Product);
  const products = await productRepository
    .createQueryBuilder('product')
    .where('product.name LIKE :query', { query: `%${query}%` })
    .orWhere('product.description LIKE :query', { query: `%${query}%` })
    .andWhere('product.status = :status', { status: ProductStatus.ACTIVE })
    .select(['product.id', 'product.name', 'product.price', 'product.imageUrl'])
    .limit(10)
    .getMany();

  const suggestions = products.map(product => ({
    id: product.id,
    name: product.name,
    price: product.price,
    imageUrl: product.imageUrl
  }));

  const response: ApiResponse<any[]> = {
    success: true,
    data: suggestions,
    message: 'Search suggestions retrieved successfully',
    timestamp: new Date().toISOString()
  };

  res.json(response);
}));

export default router;