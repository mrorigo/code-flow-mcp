import { ref, Ref } from 'vue';
import type {
    ApiResponse,
    User,
    Product,
    Order,
    CreateUserRequest,
    UpdateUserRequest,
    LoginRequest,
    AuthResponse,
    ProductSearchRequest,
    CreateProductRequest,
    UpdateProductRequest,
    CreateOrderRequest,
    OrderSummary
} from '@/types/api';

// Generic API composable with TypeScript
export function useApi<T>() {
    const data = ref<T | null>(null) as Ref<T | null>;
    const loading = ref(false);
    const error = ref<string | null>(null);

    const execute = async <R = T>(
        apiCall: () => Promise<ApiResponse<R>>
    ): Promise<ApiResponse<R> | null> => {
        loading.value = true;
        error.value = null;

        try {
            const response = await apiCall();
            data.value = response.data as T;
            return response;
        } catch (err: any) {
            error.value = err.message || 'An error occurred';
            return null;
        } finally {
            loading.value = false;
        }
    };

    return {
        data: readonly(data),
        loading: readonly(loading),
        error: readonly(error),
        execute
    };
}

// Specific API composables for different resources
export function useUserApi() {
    const { data, loading, error, execute } = useApi<User>();

    const getUsers = () =>
        execute(() =>
            fetch('/api/users').then(res => res.json())
        );

    const getUser = (id: number) =>
        execute(() =>
            fetch(`/api/users/${id}`).then(res => res.json())
        );

    const createUser = (userData: CreateUserRequest) =>
        execute(() =>
            fetch('/api/users', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData)
            }).then(res => res.json())
        );

    const updateUser = (id: number, userData: UpdateUserRequest) =>
        execute(() =>
            fetch(`/api/users/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData)
            }).then(res => res.json())
        );

    const deleteUser = (id: number) =>
        execute(() =>
            fetch(`/api/users/${id}`, {
                method: 'DELETE'
            }).then(res => res.json())
        );

    return {
        data,
        loading,
        error,
        getUsers,
        getUser,
        createUser,
        updateUser,
        deleteUser
    };
}

export function useProductApi() {
    const { data, loading, error, execute } = useApi<Product[]>();

    const getProducts = (params?: ProductSearchRequest) => {
        const searchParams = new URLSearchParams();
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined) {
                    searchParams.append(key, value.toString());
                }
            });
        }

        const url = `/api/products${searchParams.toString() ? `?${searchParams.toString()}` : ''}`;
        return execute(() =>
            fetch(url).then(res => res.json())
        );
    };

    const getProduct = (id: number) =>
        execute(() =>
            fetch(`/api/products/${id}`).then(res => res.json())
        );

    const createProduct = (productData: CreateProductRequest) =>
        execute(() =>
            fetch('/api/products', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(productData)
            }).then(res => res.json())
        );

    const updateProduct = (id: number, productData: UpdateProductRequest) =>
        execute(() =>
            fetch(`/api/products/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(productData)
            }).then(res => res.json())
        );

    const deleteProduct = (id: number) =>
        execute(() =>
            fetch(`/api/products/${id}`, {
                method: 'DELETE'
            }).then(res => res.json())
        );

    const searchProducts = (query: string) =>
        execute(() =>
            fetch(`/api/products/search/suggestions?q=${encodeURIComponent(query)}`)
                .then(res => res.json())
        );

    return {
        data,
        loading,
        error,
        getProducts,
        getProduct,
        createProduct,
        updateProduct,
        deleteProduct,
        searchProducts
    };
}

export function useOrderApi() {
    const { data, loading, error, execute } = useApi<Order[]>();

    const getOrders = (status?: string) => {
        const url = status ? `/api/orders?status=${status}` : '/api/orders';
        return execute(() =>
            fetch(url).then(res => res.json())
        );
    };

    const getOrder = (id: number) =>
        execute(() =>
            fetch(`/api/orders/${id}`).then(res => res.json())
        );

    const createOrder = (orderData: CreateOrderRequest) =>
        execute(() =>
            fetch('/api/orders', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(orderData)
            }).then(res => res.json())
        );

    const cancelOrder = (id: number, reason?: string) =>
        execute(() =>
            fetch(`/api/orders/${id}/cancel`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason })
            }).then(res => res.json())
        );

    const getOrderStats = () =>
        execute(() =>
            fetch('/api/orders/stats/summary').then(res => res.json())
        );

    return {
        data,
        loading,
        error,
        getOrders,
        getOrder,
        createOrder,
        cancelOrder,
        getOrderStats
    };
}

export function useAuthApi() {
    const { data, loading, error, execute } = useApi<AuthResponse>();

    const login = (credentials: LoginRequest) =>
        execute(() =>
            fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(credentials)
            }).then(res => res.json())
        );

    const register = (userData: CreateUserRequest) =>
        execute(() =>
            fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData)
            }).then(res => res.json())
        );

    const refreshToken = (refreshToken: string) =>
        execute(() =>
            fetch('/api/auth/refresh', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refreshToken })
            }).then(res => res.json())
        );

    const logout = () =>
        execute(() =>
            fetch('/api/auth/logout', {
                method: 'POST'
            }).then(res => res.json())
        );

    const getCurrentUser = () =>
        execute(() =>
            fetch('/api/auth/me').then(res => res.json())
        );

    return {
        data,
        loading,
        error,
        login,
        register,
        refreshToken,
        logout,
        getCurrentUser
    };
}

// Generic CRUD composable factory
export function useCrudApi<T, CreateData = Partial<T>, UpdateData = Partial<T>>(
    resource: string
) {
    const { data, loading, error, execute } = useApi<T>();

    const getAll = (params?: Record<string, any>) => {
        const searchParams = new URLSearchParams();
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined) {
                    searchParams.append(key, value.toString());
                }
            });
        }

        const url = `/api/${resource}${searchParams.toString() ? `?${searchParams.toString()}` : ''}`;
        return execute(() =>
            fetch(url).then(res => res.json())
        );
    };

    const getById = (id: number) =>
        execute(() =>
            fetch(`/api/${resource}/${id}`).then(res => res.json())
        );

    const create = (createData: CreateData) =>
        execute(() =>
            fetch(`/api/${resource}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(createData)
            }).then(res => res.json())
        );

    const update = (id: number, updateData: UpdateData) =>
        execute(() =>
            fetch(`/api/${resource}/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updateData)
            }).then(res => res.json())
        );

    const remove = (id: number) =>
        execute(() =>
            fetch(`/api/${resource}/${id}`, {
                method: 'DELETE'
            }).then(res => res.json())
        );

    return {
        data,
        loading,
        error,
        getAll,
        getById,
        create,
        update,
        remove
    };
}