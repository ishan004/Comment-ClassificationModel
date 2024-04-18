from django.urls import path,include
from .views import  display_comments, classify_comments, display_evaluation, fetch_comments, svm, process_video, svm_evaluation


urlpatterns = [
    path('', process_video, name='home' ), 
    path('comments/', display_comments, name='display_comments'),
    path('comments/<str:video_id>/', display_comments, name='display_comments_with_id'),
    path('classify-comments/', classify_comments, name='classify_comments'),
    path('classify-comments/<str:video_id>/', classify_comments, name='display_classification_with_id'),
    path('evaluate/', display_evaluation, name='evaluate'),
    path('fetch_comments/', fetch_comments, name='fetch_comments'),
    path('svm', svm, name='svm'),
    path('process_video/', process_video, name='process_video'),
    path('svm_evaluation', svm_evaluation, name='svm_evaluation'),
    
]
