from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # This is the root URL ('/')
    path('speech-to-text/', views.speech_to_text, name='speech_to_text'),
    path('text-to-speech/', views.text_to_speech, name='text_to_speech'),
]
