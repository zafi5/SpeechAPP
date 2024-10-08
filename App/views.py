# import os
# import deepspeech
# import wave
# import numpy as np
# from pydub import AudioSegment
# from django.http import JsonResponse, HttpResponse
# from django.shortcuts import render  # Ensure you have this import
# from transformers import pipeline
# from datasets import load_dataset
# import soundfile as sf
# import torch
#
# # Index view
# def index(request):
#     print("Index view accessed")  # Debug message
#     return render(request, 'index.html')  # Adjust the template path as needed
#
# # Load DeepSpeech model (for STT)
# model_file_path = os.path.join(os.path.dirname(__file__), 'deepspeech-0.9.3-models.pbmm')
# model = deepspeech.Model(model_file_path)
# print("DeepSpeech model loaded")  # Debug message
#
# # Initialize the text-to-speech pipeline (for TTS)
# synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# print("Text-to-speech pipeline initialized")  # Debug message
#
# # Audio conversion for STT
# def convert_audio(input_path, output_path):
#     print(f"Converting audio: {input_path} to {output_path}")  # Debug message
#     audio = AudioSegment.from_wav(input_path)
#     audio = audio.set_frame_rate(16000).set_channels(1)  # 16 kHz and mono
#     audio.export(output_path, format="wav")
#     print("Audio conversion complete")  # Debug message
#
# # Speech-to-Text (STT) View
# def speech_to_text(request):
#     if request.method == 'POST':
#         print("Speech-to-text request received")  # Debug message
#         audio_file = request.FILES.get('audio')
#         if not audio_file:
#             return JsonResponse({'error': 'Audio file is required.'}, status=400)
#
#         # Convert audio file to the correct format
#         converted_audio_file_path = 'converted_audio.wav'  # Ensure path is writable
#         with open(converted_audio_file_path, 'wb') as f:
#             f.write(audio_file.read())
#         print("Audio file saved")  # Debug message
#
#         # Convert the audio for DeepSpeech processing
#         convert_audio(converted_audio_file_path, converted_audio_file_path)
#
#         # Open and process the audio file with DeepSpeech
#         with wave.open(converted_audio_file_path, 'rb') as audio_file:
#             audio_data = audio_file.readframes(audio_file.getnframes())
#             audio_data_np = np.frombuffer(audio_data, dtype=np.int16)
#
#         # Perform speech-to-text using DeepSpeech
#         print("Performing speech-to-text")  # Debug message
#         text = model.stt(audio_data_np)
#         print(f"Recognized text: {text}")  # Debug message
#
#         # Return the recognized text as JSON response
#         return JsonResponse({'recognized_text': text})
#     else:
#         return JsonResponse({'error': 'POST method required.'}, status=405)
#
#
# def text_to_speech(request):
#     if request.method == 'POST':
#         input_text = request.POST.get('text', '')
#         if not input_text:
#             return JsonResponse({'error': 'Text is required.'}, status=400)
#
#         print(f"Converting text to speech for text: {input_text}")  # Debug message
#
#         # Generate speech from the provided text
#         speech = synthesiser(input_text, forward_params={"speaker_embeddings": speaker_embedding})
#
#         # Instead of saving the audio, stream the audio directly in the response
#         audio_data = speech["audio"]  # Get the generated audio data
#         sample_rate = speech["sampling_rate"]
#
#         # Create a BytesIO stream to hold the audio data
#         import io
#         audio_buffer = io.BytesIO()
#         sf.write(audio_buffer, audio_data, samplerate=sample_rate, format='WAV')
#         audio_buffer.seek(0)  # Rewind to the start of the BytesIO buffer
#
#         # Return the audio data as a streaming response
#         response = HttpResponse(audio_buffer, content_type='audio/wav')
#         response['Content-Disposition'] = 'inline; filename="speech.wav"'
#
#         print("Text-to-speech streaming complete")  # Debug message
#         return response
#
#     return JsonResponse({'error': 'POST method required.'}, status=405)



import os
import deepspeech
import wave
import numpy as np
from pydub import AudioSegment
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render  # Ensure you have this import
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
model_file_path = os.path.join(os.path.dirname(__file__), 'deepspeech-0.9.3-models.pbmm')
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
