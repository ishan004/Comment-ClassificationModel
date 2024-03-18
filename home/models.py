from django.db import models

class Comment(models.Model):
    video_id = models.CharField(max_length=255)
    text = models.TextField()
    is_toxic = models.BooleanField(default=False)
