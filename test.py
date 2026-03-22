from ollama import chat
from ollama import ChatResponse
import sys
import json
import pyaudio
from vosk import Model, KaldiRecognizer
import torch
import sounddevice as sd
from os import path
#import soundfile as sf
#from features.speak import speak
#from pathlib import Path

path = path.dirname(path.abspath(__file__))+"/models/vosk-model-small-ru-0.22" 
model = Model(path) #путь до модели обработчика голоса Vosk

print("LOG Load system_instruction")
system_instruction = "Ты Стелла, голосовой ассистент, общайся с пользователем как с другом, всегда отвечай на русском и не используй смайлики или иные символы которые не способен обработать синтезатор речи, всегда укладывай ответ в 1000 символов."
response: ChatResponse = chat(model='gemma3', messages=[{'role': 'system', 'content': system_instruction,}])
print(response.message.content)

recognizer = KaldiRecognizer(model, 16000)

print("LOG load listening setting")
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,input=True, frames_per_buffer=4000)

#Инициализация silero
print("LOG load silero, tts module")
device = torch.device('cpu')  #'cpu' можно и 'cuda' если есть GPU
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='ru',
                                     speaker='v3_1_ru')  # или 'v4_ru' для новой версии

model.to(device)

# Доступные голоса
available_speakers = model.speakers
print(f"Доступные голоса: {available_speakers}")


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
				break
			elif text != "" or text != " ":
				response: ChatResponse = chat(model='gemma3', messages=[{'role': 'user', 'content': text,}]) #Запрос

				text = response.message.content #Результат
				print(text)
			
			#speak(text)
			# Генерируем аудио и воспроизводим аудио Доступные голоса: ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
				audio = model.apply_tts(text=text,speaker='kseniya', sample_rate=48000)
				sd.play(audio, 48000)
				sd.wait()

	#except sr.RequestError as e:
		#print("Could not request results; {0}".format(e))

	except sr.UnknownValueError:
		print("Could not understand audio")
