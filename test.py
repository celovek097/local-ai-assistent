from response import local_response, clear_history
import sys
import json
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
import torch
import sounddevice as sd
from os import path

vosk_path = path.dirname(path.abspath(__file__))+"/models/vosk-model-small-ru-0.22" 
vosk_model = Model(vosk_path) #stt путь до модели обработчика голоса Vosk

recognizer = KaldiRecognizer(vosk_model, 16000)

print("LOG load listening setting")
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,input=True, frames_per_buffer=4000)

print("LOG load silero, tts module")

device = torch.device('cpu')  #'cpu' можно и 'cuda' если есть GPU
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='ru',
                                     speaker='v3_1_ru')  # или 'v4_ru' для новой версии

model.to(device)

#Доступные голоса
available_speakers = model.speakers
print(f"Доступные голоса: {available_speakers}")

clear_history()
while True:
	try:
		data = stream.read(8000, exception_on_overflow=False)
		print("Listening...")
		if recognizer.AcceptWaveform(data): #Обработка звука блоками по 4000 байт (или бит не помню)
			result_dict = json.loads(recognizer.Result())
			text = result_dict.get("text", "")
			text = text.lower()
			print("You said:", text)

			if "выход" in text:
				print("Exiting program...")
				stream.stop_stream()
				stream.close()
				break
			if text != "" and text != " ":
				text = local_response(text)
				print(text)
				#speak(text)
				# Генерируем аудио и воспроизводим аудио Доступные голоса: ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
				audio = model.apply_tts(text=text,speaker='xenia', sample_rate=48000)
				sd.play(audio, samplerate=48000, latency='low', blocksize=256)
				sd.wait()
	except Exception as e:
		print("err", e)
