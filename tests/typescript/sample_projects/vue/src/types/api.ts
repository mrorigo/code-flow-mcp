// API Response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  timestamp: string;
  meta?: {
    pagination?: PaginationInfo;
    [key: string]: any;
  };
}

export interface PaginationInfo {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// User types
export interface User {
  id: number;
  firstName: string;
  lastName: string;
  email: string;
  role: UserRole;
  isActive: boolean;
  avatar?: string;
  phoneNumber?: string;
  preferences: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}

export type UserRole = 'admin' | 'user' | 'moderator';

export interface CreateUserRequest {
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  role?: UserRole;
}

export interface UpdateUserRequest {
  firstName?: string;
  lastName?: string;
  phoneNumber?: string;
  avatar?: string;
  preferences?: Record<string, any>;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface AuthResponse {
  user: User;
  accessToken: string;
  refreshToken: string;
}

// Product types
export interface Product {
  id: number;
  name: string;
  description?: string;
  price: number;
  finalPrice: number;
  stockQuantity: number;
  category: ProductCategory;
  status: ProductStatus;
  imageUrl?: string;
  specifications: Record<string, any>;
  tags: string[];
  discountPercentage: number;
  isOnSale: boolean;
  isAvailable: boolean;
  availabilityStatus: string;
  viewCount: number;
  createdAt: string;
  updatedAt: string;
}

export type ProductCategory =
  | 'electronics'
  | 'clothing'
  | 'books'
  | 'home'
  | 'sports'
  | 'other';

export type ProductStatus =
  | 'active'
  | 'inactive'
  | 'out_of_stock'
  | 'discontinued';

export interface ProductFilters {
  category?: ProductCategory;
  status?: ProductStatus;
  minPrice?: number;
  maxPrice?: number;
  search?: string;
}

export interface ProductSearchRequest extends ProductFilters {
  page?: number;
  limit?: number;
  sortBy?: string;
  sortOrder?: 'ASC' | 'DESC';
}

export interface CreateProductRequest {
  name: string;
  description?: string;
  price: number;
  stockQuantity?: number;
  category?: ProductCategory;
  status?: ProductStatus;
  imageUrl?: string;
  specifications?: Record<string, any>;
  tags?: string[];
  discountPercentage?: number;
}

export interface UpdateProductRequest extends Partial<CreateProductRequest> {}

// Order types
export interface Order {
  id: number;
  orderNumber: string;
  quantity: number;
  unitPrice: number;
  totalAmount: number;
  subtotal: number;
  status: OrderStatus;
  paymentStatus: PaymentStatus;
  shippingAddress?: string;
  trackingNumber?: string;
  notes?: string;
  shippedAt?: string;
  deliveredAt?: string;
  metadata: Record<string, any>;
  isCompleted: boolean;
  isCancelled: boolean;
  isPending: boolean;
  canBeCancelled: boolean;
  canBeShipped: boolean;
  daysSinceCreation: number;
  createdAt: string;
  updatedAt: string;
  // Populated relations
  product?: Product;
}

export type OrderStatus =
  | 'pending'
  | 'confirmed'
  | 'processing'
  | 'shipped'
  | 'delivered'
  | 'cancelled'
  | 'refunded';

export type PaymentStatus =
  | 'pending'
  | 'completed'
  | 'failed'
  | 'refunded';

export interface CreateOrderRequest {
  productId: number;
  quantity: number;
  shippingAddress?: string;
}

export interface OrderSummary {
  totalOrders: number;
  completedOrders: number;
  pendingOrders: number;
  totalSpent: number;
}

// Generic CRUD types
export interface ListResponse<T> {
  data: T[];
  pagination: PaginationInfo;
}

export interface CreateResponse<T> {
  data: T;
  message: string;
}

export interface UpdateResponse<T> {
  data: T;
  message: string;
}

// Form validation types
export interface ValidationRule {
  required?: boolean;
  min?: number;
  max?: number;
  pattern?: RegExp;
  custom?: (value: any) => boolean | string;
  message?: string;
}

export interface FormField<T = any> {
  value: T;
  error?: string;
  touched: boolean;
  rules: ValidationRule[];
}

export interface FormState {
  [key: string]: FormField;
}

// Component prop types
export interface BaseComponentProps {
  class?: string;
  style?: string | Record<string, string>;
  disabled?: boolean;
  loading?: boolean;
}

// Table/List component types
export interface TableColumn<T = any> {
  key: keyof T;
  label: string;
  sortable?: boolean;
  width?: number;
  align?: 'left' | 'center' | 'right';
  formatter?: (value: any, row: T) => string;
  render?: (value: any, row: T) => any;
}

export interface TableProps<T = any> {
  data: T[];
  columns: TableColumn<T>[];
  loading?: boolean;
  pagination?: boolean;
  pageSize?: number;
  total?: number;
  currentPage?: number;
  onSort?: (key: keyof T, order: 'ASC' | 'DESC') => void;
  onPageChange?: (page: number) => void;
  onRowClick?: (row: T) => void;
}

// Modal/Dialog types
export interface ModalProps {
  visible: boolean;
  title?: string;
  width?: number | string;
  closable?: boolean;
  maskClosable?: boolean;
  confirmLoading?: boolean;
  onClose?: () => void;
  onConfirm?: () => void;
  onCancel?: () => void;
}

// Notification types
export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface NotificationOptions {
  type: NotificationType;
  title?: string;
  message: string;
  duration?: number;
  showClose?: boolean;
}

// Utility types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Nullable<T> = T | null;

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// API Error types
export interface ApiError {
  success: false;
  error: string;
  message: string;
  code?: string;
  details?: any;
  timestamp: string;
}

// Store state types
export interface LoadingState {
  [key: string]: boolean;
}

export interface ErrorState {
  [key: string]: string | null;
}

// Route meta types
export interface RouteMeta {
  title?: string;
  requiresAuth?: boolean;
  roles?: UserRole[];
  permissions?: string[];
  breadcrumb?: string;
  icon?: string;
  hidden?: boolean;
}

// Event types
export interface CustomEventPayload<T = any> {
  type: string;
  payload: T;
}

// Theme types
export type Theme = 'light' | 'dark' | 'auto';

export interface ThemeConfig {
  theme: Theme;
  primaryColor: string;
  fontSize: 'small' | 'medium' | 'large';
  compact: boolean;
}