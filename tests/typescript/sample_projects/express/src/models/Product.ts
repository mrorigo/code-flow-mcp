import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  OneToMany,
  Index,
  BeforeInsert,
  BeforeUpdate
} from 'typeorm';
import { Exclude, Expose } from 'class-transformer';
import { IsNotEmpty, IsNumber, IsEnum, IsOptional, Min, Max } from 'class-validator';
import { User } from './User';
import { Order } from './Order';

export enum ProductStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  OUT_OF_STOCK = 'out_of_stock',
  DISCONTINUED = 'discontinued'
}

export enum ProductCategory {
  ELECTRONICS = 'electronics',
  CLOTHING = 'clothing',
  BOOKS = 'books',
  HOME = 'home',
  SPORTS = 'sports',
  OTHER = 'other'
}

@Entity('products')
@Index(['name'])
@Index(['category'])
@Index(['status'])
@Index(['price'])
export class Product {
  @PrimaryGeneratedColumn()
  @Expose()
  id: number;

  @Column({ type: 'varchar', length: 255 })
  @IsNotEmpty({ message: 'Product name is required' })
  @Expose()
  name: string;

  @Column({ type: 'text', nullable: true })
  @IsOptional()
  @Expose()
  description?: string;

  @Column({ type: 'decimal', precision: 10, scale: 2 })
  @IsNumber({}, { message: 'Price must be a valid number' })
  @Min(0, { message: 'Price must be positive' })
  @Expose()
  price: number;

  @Column({ type: 'int', default: 0 })
  @IsNumber({}, { message: 'Stock quantity must be a valid number' })
  @Min(0, { message: 'Stock quantity cannot be negative' })
  @Expose()
  stockQuantity: number;

  @Column({
    type: 'enum',
    enum: ProductCategory,
    default: ProductCategory.OTHER
  })
  @IsEnum(ProductCategory, { message: 'Category must be a valid product category' })
  @Expose()
  category: ProductCategory;

  @Column({
    type: 'enum',
    enum: ProductStatus,
    default: ProductStatus.ACTIVE
  })
  @IsEnum(ProductStatus, { message: 'Status must be a valid product status' })
  @Expose()
  status: ProductStatus;

  @Column({ type: 'varchar', length: 500, nullable: true })
  @IsOptional()
  @Expose()
  imageUrl?: string;

  @Column({ type: 'json', nullable: true })
  @Expose()
  specifications: Record<string, any> = {};

  @Column({ type: 'json', nullable: true })
  @Expose()
  tags: string[] = [];

  @Column({ type: 'decimal', precision: 3, scale: 2, default: 0 })
  @IsNumber({}, { message: 'Discount must be a valid number' })
  @Min(0, { message: 'Discount cannot be negative' })
  @Max(1, { message: 'Discount cannot exceed 100%' })
  @Expose()
  discountPercentage: number;

  @Column({ type: 'int', default: 0 })
  @Expose()
  viewCount: number;

  @CreateDateColumn({ type: 'timestamp' })
  @Expose()
  createdAt: Date;

  @UpdateDateColumn({ type: 'timestamp' })
  @Expose()
  updatedAt: Date;

  // Relations
  @ManyToOne(() => User, { nullable: true })
  @Expose()
  createdBy?: User;

  @OneToMany(() => Order, order => order.product)
  @Exclude() // Don't expose orders in product serialization
  orders: Order[];

  // Virtual properties
  @Expose()
  get finalPrice(): number {
    return this.price * (1 - this.discountPercentage);
  }

  @Expose()
  get isOnSale(): boolean {
    return this.discountPercentage > 0;
  }

  @Expose()
  get isAvailable(): boolean {
    return this.status === ProductStatus.ACTIVE && this.stockQuantity > 0;
  }

  @Expose()
  get availabilityStatus(): string {
    if (this.status !== ProductStatus.ACTIVE) {
      return this.status.replace('_', ' ').toUpperCase();
    }
    if (this.stockQuantity === 0) {
      return 'OUT OF STOCK';
    }
    if (this.stockQuantity <= 5) {
      return 'LOW STOCK';
    }
    return 'IN STOCK';
  }

  // Business logic methods
  public canPurchase(quantity: number): boolean {
    return this.isAvailable && this.stockQuantity >= quantity;
  }

  public reduceStock(quantity: number): void {
    if (!this.canPurchase(quantity)) {
      throw new Error('Insufficient stock or product not available');
    }
    this.stockQuantity -= quantity;
  }

  public increaseStock(quantity: number): void {
    if (quantity <= 0) {
      throw new Error('Quantity must be positive');
    }
    this.stockQuantity += quantity;
  }

  public applyDiscount(percentage: number): void {
    if (percentage < 0 || percentage > 1) {
      throw new Error('Discount percentage must be between 0 and 1');
    }
    this.discountPercentage = percentage;
  }

  public removeDiscount(): void {
    this.discountPercentage = 0;
  }

  public updateStatus(newStatus: ProductStatus): void {
    this.status = newStatus;
  }

  @BeforeInsert()
  @BeforeUpdate()
  private validateProduct(): void {
    if (this.name) {
      this.name = this.name.trim();
    }
    if (this.description) {
      this.description = this.description.trim();
    }
  }

  // DTOs for different operations
  public toListItem(): Partial<Product> {
    return {
      id: this.id,
      name: this.name,
      price: this.price,
      finalPrice: this.finalPrice,
      category: this.category,
      status: this.status,
      imageUrl: this.imageUrl,
      isOnSale: this.isOnSale,
      availabilityStatus: this.availabilityStatus,
      createdAt: this.createdAt
    };
  }

  public toDetailView(): Record<string, any> {
    return {
      id: this.id,
      name: this.name,
      description: this.description,
      price: this.price,
      finalPrice: this.finalPrice,
      stockQuantity: this.stockQuantity,
      category: this.category,
      status: this.status,
      imageUrl: this.imageUrl,
      specifications: this.specifications,
      tags: this.tags,
      discountPercentage: this.discountPercentage,
      isOnSale: this.isOnSale,
      isAvailable: this.isAvailable,
      availabilityStatus: this.availabilityStatus,
      viewCount: this.viewCount,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt
    };
  }
}