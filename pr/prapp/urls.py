# myapp/urls.py

from django.urls import path
from .views import upload_image, success

urlpatterns = [
    path('', upload_image, name='upload_image'),
    path('success/', success, name='success'),
]

