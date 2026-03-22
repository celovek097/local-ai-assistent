from ollama import chat, ChatResponse
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
from openai import OpenAI

client = OpenAI(
    api_key="sk-a03cec19ea88493c85fb3a3e0c8592e0",
    base_url="https://api.deepseek.com"
)

def ethernet_response(text):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Ты "},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    return response.choices[0].message.content

def local_response(text):
    response: ChatResponse = chat(
        model='gemma3',#Запрос
        messages=[
            {'role': 'system', 'content': text}])
    return response.message.content #Результат


path = path.dirname(path.abspath(__file__))+"/models/vosk-model-small-ru-0.22" 
model = Model(path) #путь до модели обработчика голоса Vosk

print("LOG Load system_instruction")
system_instruction = "Ты Стелла, голосовой ассистент, общайся с пользователем как с другом, всегда отвечай на русском и только буквами, всегда укладывай ответ в 1000 символов."
print(local_response(system_instruction))

recognizer = KaldiRecognizer(model, 16000)

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
            text = local_response(text)
            print(text)
            #speak(text)
			# Генерируем аудио и воспроизводим аудио Доступные голоса: ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
            audio = model.apply_tts(text=text,speaker='kseniya', sample_rate=48000)
            sd.play(audio, 48000)
            sd.wait()
    except:
        print("err")