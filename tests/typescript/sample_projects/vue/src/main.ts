import { createApp } from 'vue';
import { createPinia } from 'pinia';
import ElementPlus from 'element-plus';
import * as ElementPlusIconsVue from '@element-plus/icons-vue';

import App from './App.vue';
import router from './router';
import { setupInterceptors } from './api/http';

import 'element-plus/dist/index.css';
import './assets/styles/global.css';

// Create Vue application instance
const app = createApp(App);

// Create and use Pinia store
const pinia = createPinia();
app.use(pinia);

// Use Vue Router
app.use(router);

// Use Element Plus UI library
app.use(ElementPlus);

// Register all Element Plus icons globally
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component);
}

// Setup HTTP interceptors for API calls
setupInterceptors();

// Global error handler
app.config.errorHandler = (err: Error, instance, info) => {
  console.error('Global error handler:', err, info);

  // You can integrate with error tracking services here
  // Example: Sentry, LogRocket, etc.
};

// Global warning handler for Vue warnings
app.config.warnHandler = (msg, instance, trace) => {
  console.warn('Vue warning:', msg, trace);
};

// Performance monitoring
if (process.env.NODE_ENV === 'development') {
  app.config.performance = true;
}

// Mount the application
app.mount('#app');

// Export app instance for testing
export default app;