import { Component, OnInit, OnDestroy } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { User } from './user.model';
import { UserService } from './user.service';

@Component({
  selector: 'app-user-management',
  template: `
    <div class="user-management">
      <h2>User Management</h2>

      <!-- User List -->
      <div class="user-list" *ngIf="users.length > 0; else noUsers">
        <div *ngFor="let user of users" class="user-card">
          <h3>{{ user.name }}</h3>
          <p>{{ user.email }}</p>
          <span class="role" [class]="'role-' + user.role">{{ user.role }}</span>
          <div class="actions">
            <button (click)="editUser(user)" class="btn btn-secondary">Edit</button>
            <button (click)="deleteUser(user.id)" class="btn btn-danger">Delete</button>
          </div>
        </div>
      </div>

      <ng-template #noUsers>
        <p>No users found. Create the first user!</p>
      </ng-template>

      <!-- Create/Edit User Form -->
      <form [formGroup]="userForm" (ngSubmit)="onSubmit()" class="user-form">
        <h3>{{ isEditing ? 'Edit User' : 'Create New User' }}</h3>

        <div class="form-group">
          <label for="name">Name:</label>
          <input
            type="text"
            id="name"
            formControlName="name"
            class="form-control"
            [class.is-invalid]="isFieldInvalid('name')"
          >
          <div *ngIf="isFieldInvalid('name')" class="invalid-feedback">
            Name is required
          </div>
        </div>

        <div class="form-group">
          <label for="email">Email:</label>
          <input
            type="email"
            id="email"
            formControlName="email"
            class="form-control"
            [class.is-invalid]="isFieldInvalid('email')"
          >
          <div *ngIf="isFieldInvalid('email')" class="invalid-feedback">
            <div *ngIf="userForm.get('email')?.errors?.['required']">Email is required</div>
            <div *ngIf="userForm.get('email')?.errors?.['email']">Invalid email format</div>
          </div>
        </div>

        <div class="form-group">
          <label for="role">Role:</label>
          <select
            id="role"
            formControlName="role"
            class="form-control"
          >
            <option value="user">User</option>
            <option value="admin">Admin</option>
            <option value="guest">Guest</option>
          </select>
        </div>

        <div class="form-actions">
          <button type="submit" class="btn btn-primary" [disabled]="userForm.invalid">
            {{ isEditing ? 'Update' : 'Create' }} User
          </button>
          <button type="button" class="btn btn-secondary" (click)="resetForm()" *ngIf="isEditing">
            Cancel
          </button>
        </div>
      </form>

      <!-- Loading Indicator -->
      <div *ngIf="isLoading" class="loading">
        <p>Loading users...</p>
      </div>

      <!-- Error Message -->
      <div *ngIf="errorMessage" class="error-message">
        <p>{{ errorMessage }}</p>
        <button (click)="clearError()" class="btn btn-secondary">Dismiss</button>
      </div>
    </div>
  `,
  styles: [`
    .user-management {
      padding: 20px;
      max-width: 800px;
      margin: 0 auto;
    }

    .user-list {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }

    .user-card {
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 16px;
      background: #f9f9f9;
    }

    .role {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.8em;
      text-transform: uppercase;
    }

    .role-admin { background: #ff4444; color: white; }
    .role-user { background: #4444ff; color: white; }
    .role-guest { background: #44ff44; color: #333; }

    .user-form {
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 20px;
      margin-top: 20px;
    }

    .form-group {
      margin-bottom: 16px;
    }

    .form-control {
      width: 100%;
      padding: 8px 12px;
      border: 1px solid #ddd;
      border-radius: 4px;
    }

    .form-control.is-invalid {
      border-color: #dc3545;
    }

    .invalid-feedback {
      color: #dc3545;
      font-size: 0.9em;
      margin-top: 4px;
    }

    .form-actions {
      display: flex;
      gap: 8px;
    }

    .btn {
      padding: 8px 16px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }

    .btn-primary {
      background: #007bff;
      color: white;
    }

    .btn-secondary {
      background: #6c757d;
      color: white;
    }

    .btn-danger {
      background: #dc3545;
      color: white;
    }

    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .loading {
      text-align: center;
      padding: 20px;
      color: #666;
    }

    .error-message {
      background: #f8d7da;
      border: 1px solid #f5c6cb;
      color: #721c24;
      padding: 12px;
      border-radius: 4px;
      margin-top: 16px;
    }

    .actions {
      margin-top: 8px;
      display: flex;
      gap: 8px;
    }
  `]
})
export class UserManagementComponent implements OnInit, OnDestroy {
  public users: User[] = [];
  public userForm: FormGroup;
  public isEditing = false;
  public isLoading = false;
  public errorMessage = '';
  public currentUserId: number | null = null;

  private destroy$ = new Subject<void>();

  constructor(
    private userService: UserService,
    private formBuilder: FormBuilder
  ) {
    this.userForm = this.createForm();
  }

  public ngOnInit(): void {
    this.loadUsers();
    this.setupSubscriptions();
  }

  public ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private createForm(): FormGroup {
    return this.formBuilder.group({
      name: ['', [Validators.required, Validators.minLength(2)]],
      email: ['', [Validators.required, Validators.email]],
      role: ['user', Validators.required]
    });
  }

  private setupSubscriptions(): void {
    this.userService.users$.pipe(
      takeUntil(this.destroy$)
    ).subscribe({
      next: (users) => {
        this.users = users;
        this.isLoading = false;
      },
      error: (error) => {
        this.handleError('Failed to load users');
        this.isLoading = false;
      }
    });
  }

  private loadUsers(): void {
    this.isLoading = true;
    this.clearError();
    this.userService.getUsers().subscribe({
      error: (error) => {
        this.handleError('Failed to load users');
        this.isLoading = false;
      }
    });
  }

  public onSubmit(): void {
    if (this.userForm.valid) {
      const formValue = this.userForm.value;

      if (this.isEditing && this.currentUserId) {
        this.updateUser(this.currentUserId, formValue);
      } else {
        this.createUser(formValue);
      }
    } else {
      this.markFormGroupTouched();
    }
  }

  private createUser(userData: any): void {
    this.userService.createUser(userData).subscribe({
      next: () => {
        this.resetForm();
        this.loadUsers();
      },
      error: (error) => {
        this.handleError('Failed to create user');
      }
    });
  }

  private updateUser(id: number, userData: any): void {
    this.userService.updateUser(id, userData).subscribe({
      next: () => {
        this.resetForm();
        this.loadUsers();
      },
      error: (error) => {
        this.handleError('Failed to update user');
      }
    });
  }

  public editUser(user: User): void {
    this.isEditing = true;
    this.currentUserId = user.id;

    this.userForm.patchValue({
      name: user.name,
      email: user.email,
      role: user.role
    });
  }

  public deleteUser(id: number): void {
    if (confirm('Are you sure you want to delete this user?')) {
      this.userService.deleteUser(id).subscribe({
        next: () => {
          this.loadUsers();
        },
        error: (error) => {
          this.handleError('Failed to delete user');
        }
      });
    }
  }

  public resetForm(): void {
    this.userForm.reset({
      role: 'user'
    });
    this.isEditing = false;
    this.currentUserId = null;
  }

  private isFieldInvalid(fieldName: string): boolean {
    const field = this.userForm.get(fieldName);
    return !!(field && field.invalid && (field.dirty || field.touched));
  }

  private markFormGroupTouched(): void {
    Object.keys(this.userForm.controls).forEach(key => {
      const control = this.userForm.get(key);
      control?.markAsTouched();
    });
  }

  private handleError(message: string): void {
    this.errorMessage = message;
  }

  public clearError(): void {
    this.errorMessage = '';
  }
}