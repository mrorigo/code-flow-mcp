
"""Models for the core Django app."""

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User as AuthUser


class Category(models.Model):
    """Category model for organizing content."""

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name_plural = "categories"
        ordering = ['name']

    def __str__(self):
        return self.name


