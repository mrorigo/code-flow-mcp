export interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
  createdAt: Date;
}

export interface CreateUserRequest {
  name: string;
  email: string;
  role?: 'admin' | 'user' | 'guest';
}