from django.urls import path,include
from .views import home, display_comments


urlpatterns = [
    path('', home, name='home' ),
    
    path('comments/', display_comments, name='display_comments'),
    path('comments/<str:video_id>/', display_comments, name='display_comments_with_id'),
]

# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.home, name='home'),
#     path('analyze/', views.analyze_comments, name='analyze_comments'),
# ]
