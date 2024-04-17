from django.urls import path,include
from .views import home, display_comments, classify_comments, display_evaluation, fetch_comments, svm


urlpatterns = [
    path('', home, name='home' ),
    
    path('comments/', display_comments, name='display_comments'),
    path('comments/<str:video_id>/', display_comments, name='display_comments_with_id'),
    path('classify-comments/', classify_comments, name='classify_comments'),
    path('classify-comments/<str:video_id>/', classify_comments, name='display_classification_with_id'),
    path('evaluate/', display_evaluation, name='evaluate'),
    path('fetch_comment/', fetch_comments, name='fetch_comment'),
    path('svm', svm, name='svm'),
    
]
