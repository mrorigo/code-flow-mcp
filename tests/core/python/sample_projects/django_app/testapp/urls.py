"""testapp URL Configuration"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('testapp.api.urls')),
    path('', include('testapp.core.urls')),
]