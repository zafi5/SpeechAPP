import os
import deepspeech
import wave
import numpy as np
from pydub import AudioSegment
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render  
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import io
from django.views.decorators.csrf import csrf_exempt

# Index view for testing purposes
def index(request):
    return render(request, 'index.html')

# Load DeepSpeech model for STT
# model_file_path = os.path.join(os.path.dirname(__file__), 'deepspeech-0.9.3-models.pbmm')
model_file_path = '/Users/tappware/PycharmProjects/API/deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_file_path)

# Load Text-to-Speech pipeline for TTS
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Audio conversion for DeepSpeech STT
def convert_audio(input_path, output_path):
    audio = AudioSegment.from_wav(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # Convert to 16kHz mono
    audio.export(output_path, format="wav")

# Speech-to-Text (STT) API endpoint
@csrf_exempt
def speech_to_text(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        if not audio_file:
            return JsonResponse({'error': 'Audio file is required.'}, status=400)

        # Save and convert the audio file
        converted_audio_file_path = 'converted_audio.wav'
        with open(converted_audio_file_path, 'wb') as f:
            f.write(audio_file.read())
        convert_audio(converted_audio_file_path, converted_audio_file_path)

        # Process the audio with DeepSpeech
        with wave.open(converted_audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.readframes(audio_file.getnframes())
            audio_data_np = np.frombuffer(audio_data, dtype=np.int16)

        # Perform speech-to-text
        text = model.stt(audio_data_np)
        return JsonResponse({'recognized_text': text})

    return JsonResponse({'error': 'POST method required.'}, status=405)

# Text-to-Speech (TTS) API endpoint
@csrf_exempt
def text_to_speech(request):
    if request.method == 'POST':
        input_text = request.POST.get('text', '')
        if not input_text:
            return JsonResponse({'error': 'Text is required.'}, status=400)

        # Generate speech from text
        speech = synthesiser(input_text, forward_params={"speaker_embeddings": speaker_embedding})

        # Create a BytesIO buffer to stream the audio
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, speech["audio"], samplerate=speech["sampling_rate"], format='WAV')
        audio_buffer.seek(0)

        # Return the audio as a streaming response
        response = HttpResponse(audio_buffer, content_type='audio/wav')
        response['Content-Disposition'] = 'inline; filename="speech.wav"'
        return response

    return JsonResponse({'error': 'POST method required.'}, status=405)
