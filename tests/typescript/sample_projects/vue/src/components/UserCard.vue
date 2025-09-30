<template>
  <div class="user-card" :class="{ 'user-card--disabled': disabled }">
    <div class="user-card__avatar">
      <img
        v-if="user.avatar"
        :src="user.avatar"
        :alt="`${user.firstName} ${user.lastName}`"
        @error="handleImageError"
      />
      <div v-else class="user-card__avatar-placeholder">
        {{ getInitials(user.firstName, user.lastName) }}
      </div>
    </div>

    <div class="user-card__content">
      <h3 class="user-card__name">
        {{ user.firstName }} {{ user.lastName }}
        <span v-if="showRole" class="user-card__role" :class="`user-card__role--${user.role}`">
          {{ user.role }}
        </span>
      </h3>

      <p class="user-card__email">{{ user.email }}</p>

      <div v-if="user.phoneNumber" class="user-card__phone">
        <i class="icon-phone"></i>
        {{ user.phoneNumber }}
      </div>

      <div class="user-card__meta">
        <span class="user-card__status" :class="{ 'user-card__status--active': user.isActive }">
          {{ user.isActive ? 'Active' : 'Inactive' }}
        </span>
        <span class="user-card__created">
          Joined {{ formatDate(user.createdAt) }}
        </span>
      </div>
    </div>

    <div class="user-card__actions">
      <button
        v-if="showEdit"
        class="btn btn--secondary btn--small"
        @click="handleEdit"
        :disabled="disabled"
      >
        Edit
      </button>
      <button
        v-if="showDelete"
        class="btn btn--danger btn--small"
        @click="handleDelete"
        :disabled="disabled"
      >
        Delete
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, PropType } from 'vue';
import type { User, UserRole } from '@/types/api';

// Component props with detailed type definitions
interface Props {
  user: User;
  disabled?: boolean;
  showRole?: boolean;
  showEdit?: boolean;
  showDelete?: boolean;
  size?: 'small' | 'medium' | 'large';
  variant?: 'default' | 'compact' | 'detailed';
}

interface Emits {
  edit: [user: User];
  delete: [user: User];
  'image-error': [event: Event];
}

// Define props with validation and defaults
const props = withDefaults(defineProps<Props>(), {
  disabled: false,
  showRole: true,
  showEdit: true,
  showDelete: false,
  size: 'medium',
  variant: 'default'
});

// Define emits with type safety
const emit = defineEmits<Emits>();

// Computed properties with type safety
const cardClasses = computed(() => [
  `user-card--${props.size}`,
  `user-card--${props.variant}`,
  {
    'user-card--disabled': props.disabled
  }
]);

const roleDisplayName = computed((): string => {
  const roleNames: Record<UserRole, string> = {
    admin: 'Administrator',
    moderator: 'Moderator',
    user: 'User'
  };
  return roleNames[props.user.role] || props.user.role;
});

// Methods with proper typing
const getInitials = (firstName: string, lastName: string): string => {
  return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase();
};

const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
};

const handleImageError = (event: Event): void => {
  emit('image-error', event);
};

const handleEdit = (): void => {
  if (!props.disabled) {
    emit('edit', props.user);
  }
};

const handleDelete = (): void => {
  if (!props.disabled) {
    emit('delete', props.user);
  }
};

// Type-safe event handlers
const onUserClick = (event: MouseEvent): void => {
  // Custom click handling logic
  console.log('User card clicked:', props.user.id, event);
};

// Expose methods for parent components (template refs)
defineExpose({
  getInitials,
  formatDate,
  focus: () => {
    // Focus logic for accessibility
  }
});
</script>

<style scoped>
.user-card {
  display: flex;
  align-items: center;
  padding: 1rem;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  background: white;
  transition: all 0.2s ease;
}

.user-card:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.user-card--disabled {
  opacity: 0.6;
  pointer-events: none;
}

.user-card__avatar {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  overflow: hidden;
  margin-right: 1rem;
  flex-shrink: 0;
}

.user-card__avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.user-card__avatar-placeholder {
  width: 100%;
  height: 100%;
  background: #f0f2f5;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  color: #666;
}

.user-card__content {
  flex: 1;
  min-width: 0;
}

.user-card__name {
  margin: 0 0 0.5rem 0;
  font-size: 1.1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.user-card__role {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
  font-weight: 500;
  text-transform: uppercase;
}

.user-card__role--admin {
  background: #f0f9ff;
  color: #0369a1;
}

.user-card__role--moderator {
  background: #fef3c7;
  color: #d97706;
}

.user-card__role--user {
  background: #f0fdf4;
  color: #166534;
}

.user-card__email {
  margin: 0 0 0.25rem 0;
  color: #666;
  font-size: 0.9rem;
}

.user-card__phone {
  margin: 0 0 0.5rem 0;
  color: #666;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.user-card__meta {
  display: flex;
  gap: 1rem;
  font-size: 0.8rem;
  color: #888;
}

.user-card__status {
  color: #ef4444;
}

.user-card__status--active {
  color: #22c55e;
}

.user-card__actions {
  display: flex;
  gap: 0.5rem;
  margin-left: 1rem;
}

.btn {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s ease;
}

.btn--secondary {
  background: #f8fafc;
  color: #475569;
  border: 1px solid #e2e8f0;
}

.btn--danger {
  background: #fee2e2;
  color: #dc2626;
  border: 1px solid #fecaca;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Responsive design */
@media (max-width: 768px) {
  .user-card {
    flex-direction: column;
    text-align: center;
  }

  .user-card__actions {
    margin-left: 0;
    margin-top: 1rem;
  }
}
</style>