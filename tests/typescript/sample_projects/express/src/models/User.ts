import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  OneToMany,
  Index,
  BeforeInsert,
  BeforeUpdate
} from 'typeorm';
import { Exclude, Expose } from 'class-transformer';
import { IsEmail, IsNotEmpty, MinLength, IsEnum, IsOptional } from 'class-validator';
import { UserRole, Permission } from '../types/api';
import { Order } from './Order';

@Entity('users')
@Index(['email'], { unique: true })
@Index(['role'])
export class User {
  @PrimaryGeneratedColumn()
  @Expose()
  id: number;

  @Column({ type: 'varchar', length: 100 })
  @IsNotEmpty({ message: 'First name is required' })
  @Expose()
  firstName: string;

  @Column({ type: 'varchar', length: 100 })
  @IsNotEmpty({ message: 'Last name is required' })
  @Expose()
  lastName: string;

  @Column({ type: 'varchar', length: 255, unique: true })
  @IsEmail({}, { message: 'Please provide a valid email address' })
  @Expose()
  email: string;

  @Column({ type: 'varchar', length: 255 })
  @Exclude({ toPlainOnly: true }) // Never expose password in JSON
  @IsNotEmpty({ message: 'Password is required' })
  @MinLength(8, { message: 'Password must be at least 8 characters long' })
  password: string;

  @Column({
    type: 'enum',
    enum: UserRole,
    default: UserRole.USER
  })
  @IsEnum(UserRole, { message: 'Role must be a valid user role' })
  @Expose()
  role: UserRole;

  @Column({ type: 'boolean', default: true })
  @IsOptional()
  @Expose()
  isActive: boolean;

  @Column({ type: 'text', nullable: true })
  @IsOptional()
  @Expose()
  avatar?: string;

  @Column({ type: 'varchar', length: 20, nullable: true })
  @IsOptional()
  @Expose()
  phoneNumber?: string;

  @Column({ type: 'json', nullable: true })
  @Expose()
  preferences: Record<string, any> = {};

  @CreateDateColumn({ type: 'timestamp' })
  @Expose()
  createdAt: Date;

  @UpdateDateColumn({ type: 'timestamp' })
  @Expose()
  updatedAt: Date;

  // Relations
  @OneToMany(() => Order, order => order.user, { cascade: true })
  @Expose()
  orders: Order[];

  // Virtual properties
  @Expose()
  get fullName(): string {
    return `${this.firstName} ${this.lastName}`;
  }

  @Expose()
  get permissions(): Permission[] {
    return this.getPermissionsForRole(this.role);
  }

  // Business logic methods
  public hasPermission(permission: Permission): boolean {
    return this.permissions.includes(permission);
  }

  public isAdmin(): boolean {
    return this.role === UserRole.ADMIN;
  }

  public isModerator(): boolean {
    return this.role === UserRole.MODERATOR || this.isAdmin();
  }

  public canAccessResource(resourceOwnerId: number): boolean {
    return this.isAdmin() || this.id === resourceOwnerId;
  }

  @BeforeInsert()
  @BeforeUpdate()
  private validateUser(): void {
    if (this.firstName) {
      this.firstName = this.firstName.trim();
    }
    if (this.lastName) {
      this.lastName = this.lastName.trim();
    }
    if (this.email) {
      this.email = this.email.toLowerCase().trim();
    }
  }

  private getPermissionsForRole(role: UserRole): Permission[] {
    const rolePermissions: Record<UserRole, Permission[]> = {
      [UserRole.USER]: [
        Permission.USER_READ,
        Permission.PRODUCT_READ,
        Permission.ORDER_READ,
        Permission.ORDER_CREATE,
        Permission.ORDER_UPDATE
      ],
      [UserRole.MODERATOR]: [
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.PRODUCT_READ,
        Permission.PRODUCT_CREATE,
        Permission.PRODUCT_UPDATE,
        Permission.ORDER_READ,
        Permission.ORDER_UPDATE
      ],
      [UserRole.ADMIN]: Object.values(Permission)
    };

    return rolePermissions[role] || [];
  }

  // DTOs for different operations
  public toPublicProfile(): Partial<User> {
    const { password, ...publicProfile } = this;
    return publicProfile;
  }

  public toSafeJSON(): Record<string, any> {
    return {
      id: this.id,
      fullName: this.fullName,
      email: this.email,
      role: this.role,
      isActive: this.isActive,
      avatar: this.avatar,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt
    };
  }
}