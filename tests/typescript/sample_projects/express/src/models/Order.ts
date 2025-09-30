import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  JoinColumn,
  Index,
  BeforeInsert,
  BeforeUpdate
} from 'typeorm';
import { Exclude, Expose } from 'class-transformer';
import { IsNotEmpty, IsNumber, IsEnum, IsOptional, Min } from 'class-validator';
import { User } from './User';
import { Product } from './Product';

export enum OrderStatus {
  PENDING = 'pending',
  CONFIRMED = 'confirmed',
  PROCESSING = 'processing',
  SHIPPED = 'shipped',
  DELIVERED = 'delivered',
  CANCELLED = 'cancelled',
  REFUNDED = 'refunded'
}

export enum PaymentStatus {
  PENDING = 'pending',
  COMPLETED = 'completed',
  FAILED = 'failed',
  REFUNDED = 'refunded'
}

@Entity('orders')
@Index(['userId'])
@Index(['status'])
@Index(['paymentStatus'])
@Index(['createdAt'])
export class Order {
  @PrimaryGeneratedColumn()
  @Expose()
  id: number;

  @Column({ type: 'varchar', length: 100 })
  @IsNotEmpty({ message: 'Order number is required' })
  @Expose()
  orderNumber: string;

  @Column({ type: 'int' })
  @Expose()
  userId: number;

  @Column({ type: 'int' })
  @Expose()
  productId: number;

  @Column({ type: 'int' })
  @IsNumber({}, { message: 'Quantity must be a valid number' })
  @Min(1, { message: 'Quantity must be at least 1' })
  @Expose()
  quantity: number;

  @Column({ type: 'decimal', precision: 10, scale: 2 })
  @IsNumber({}, { message: 'Unit price must be a valid number' })
  @Min(0, { message: 'Unit price must be positive' })
  @Expose()
  unitPrice: number;

  @Column({ type: 'decimal', precision: 10, scale: 2 })
  @IsNumber({}, { message: 'Total amount must be a valid number' })
  @Min(0, { message: 'Total amount must be positive' })
  @Expose()
  totalAmount: number;

  @Column({
    type: 'enum',
    enum: OrderStatus,
    default: OrderStatus.PENDING
  })
  @IsEnum(OrderStatus, { message: 'Status must be a valid order status' })
  @Expose()
  status: OrderStatus;

  @Column({
    type: 'enum',
    enum: PaymentStatus,
    default: PaymentStatus.PENDING
  })
  @IsEnum(PaymentStatus, { message: 'Payment status must be a valid payment status' })
  @Expose()
  paymentStatus: PaymentStatus;

  @Column({ type: 'text', nullable: true })
  @IsOptional()
  @Expose()
  shippingAddress?: string;

  @Column({ type: 'varchar', length: 100, nullable: true })
  @IsOptional()
  @Expose()
  trackingNumber?: string;

  @Column({ type: 'text', nullable: true })
  @IsOptional()
  @Expose()
  notes?: string;

  @Column({ type: 'timestamp', nullable: true })
  @IsOptional()
  @Expose()
  shippedAt?: Date;

  @Column({ type: 'timestamp', nullable: true })
  @IsOptional()
  @Expose()
  deliveredAt?: Date;

  @Column({ type: 'json', nullable: true })
  @Expose()
  metadata: Record<string, any> = {};

  @CreateDateColumn({ type: 'timestamp' })
  @Expose()
  createdAt: Date;

  @UpdateDateColumn({ type: 'timestamp' })
  @Expose()
  updatedAt: Date;

  // Relations
  @ManyToOne(() => User, user => user.orders, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'userId' })
  @Expose()
  user: User;

  @ManyToOne(() => Product, product => product.orders, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'productId' })
  @Expose()
  product: Product;

  // Virtual properties
  @Expose()
  get subtotal(): number {
    return this.unitPrice * this.quantity;
  }

  @Expose()
  get isCompleted(): boolean {
    return this.status === OrderStatus.DELIVERED;
  }

  @Expose()
  get isCancelled(): boolean {
    return this.status === OrderStatus.CANCELLED;
  }

  @Expose()
  get isPending(): boolean {
    return this.status === OrderStatus.PENDING;
  }

  @Expose()
  get canBeCancelled(): boolean {
    return [OrderStatus.PENDING, OrderStatus.CONFIRMED].includes(this.status);
  }

  @Expose()
  get canBeShipped(): boolean {
    return this.status === OrderStatus.PROCESSING && this.paymentStatus === PaymentStatus.COMPLETED;
  }

  @Expose()
  get daysSinceCreation(): number {
    return Math.floor((Date.now() - this.createdAt.getTime()) / (1000 * 60 * 60 * 24));
  }

  // Business logic methods
  public confirm(): void {
    if (this.status !== OrderStatus.PENDING) {
      throw new Error('Only pending orders can be confirmed');
    }
    this.status = OrderStatus.CONFIRMED;
  }

  public startProcessing(): void {
    if (this.status !== OrderStatus.CONFIRMED) {
      throw new Error('Only confirmed orders can be processed');
    }
    this.status = OrderStatus.PROCESSING;
  }

  public ship(trackingNumber?: string): void {
    if (!this.canBeShipped) {
      throw new Error('Order cannot be shipped in current status');
    }
    this.status = OrderStatus.SHIPPED;
    this.shippedAt = new Date();
    if (trackingNumber) {
      this.trackingNumber = trackingNumber;
    }
  }

  public deliver(): void {
    if (this.status !== OrderStatus.SHIPPED) {
      throw new Error('Only shipped orders can be delivered');
    }
    this.status = OrderStatus.DELIVERED;
    this.deliveredAt = new Date();
  }

  public cancel(reason?: string): void {
    if (!this.canBeCancelled) {
      throw new Error('Order cannot be cancelled in current status');
    }
    this.status = OrderStatus.CANCELLED;
    if (reason) {
      this.notes = reason;
    }
  }

  public refund(reason?: string): void {
    if (!this.isCompleted) {
      throw new Error('Only completed orders can be refunded');
    }
    this.status = OrderStatus.REFUNDED;
    this.paymentStatus = PaymentStatus.REFUNDED;
    if (reason) {
      this.notes = reason;
    }
  }

  public markPaymentCompleted(): void {
    this.paymentStatus = PaymentStatus.COMPLETED;
  }

  public markPaymentFailed(): void {
    this.paymentStatus = PaymentStatus.FAILED;
  }

  @BeforeInsert()
  private generateOrderNumber(): void {
    if (!this.orderNumber) {
      this.orderNumber = `ORD-${Date.now()}-${Math.random().toString(36).substr(2, 9).toUpperCase()}`;
    }
  }

  @BeforeInsert()
  @BeforeUpdate()
  private calculateTotal(): void {
    this.totalAmount = this.unitPrice * this.quantity;
  }

  // DTOs for different operations
  public toListItem(): Partial<Order> {
    return {
      id: this.id,
      orderNumber: this.orderNumber,
      quantity: this.quantity,
      totalAmount: this.totalAmount,
      status: this.status,
      paymentStatus: this.paymentStatus,
      createdAt: this.createdAt,
      shippedAt: this.shippedAt,
      deliveredAt: this.deliveredAt
    };
  }

  public toDetailView(): Record<string, any> {
    return {
      id: this.id,
      orderNumber: this.orderNumber,
      quantity: this.quantity,
      unitPrice: this.unitPrice,
      totalAmount: this.totalAmount,
      subtotal: this.subtotal,
      status: this.status,
      paymentStatus: this.paymentStatus,
      shippingAddress: this.shippingAddress,
      trackingNumber: this.trackingNumber,
      notes: this.notes,
      shippedAt: this.shippedAt,
      deliveredAt: this.deliveredAt,
      metadata: this.metadata,
      isCompleted: this.isCompleted,
      isCancelled: this.isCancelled,
      isPending: this.isPending,
      canBeCancelled: this.canBeCancelled,
      canBeShipped: this.canBeShipped,
      daysSinceCreation: this.daysSinceCreation,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt
    };
  }
}