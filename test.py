from ollama import chat, ChatResponse
from cerebras.cloud.sdk import Cerebras
from vosk import Model, KaldiRecognizer
import os
import dotenv
import json
import pyaudio
import torch
import sounddevice as sd
import socket

dotenv.load_dotenv()  # Загружаем переменные из .env
api_key = os.getenv("API_KEY")
client = Cerebras(api_key=api_key)  # Клиент сам найдёт ключ
local_ai_model = "gemma3:latest"
ethernet_ai_model = "llama3.1-8b"
system_instruction = "Ты Стелла, голосовой ассистент, общайся с пользователем как с другом, поддерживай простой диалог, всегда отвечай на русском и только буквами, всегда укладывай ответ в 1000 символов."

sys_instr = [{'role': 'system', 'content': system_instruction}, {'role': 'assistant', 'content': 'Здравствуйте!'}]

def clear_history(filename="history.txt"):
	try:
		abs_path = os.path.dirname(os.path.abspath(__file__))+'/'+filename
		with open(abs_path, 'w', encoding='utf-8'):
			print(' -----------------------\n',' clear history success\n','-----------------------')
	except:
		print("error clear history")

def save_history(history, filename="history.txt"):
	abs_path = os.path.dirname(os.path.abspath(__file__))+'/'+filename
	with open(abs_path, 'a', encoding='utf-8') as f:
		json.dump(history, f, ensure_ascii=False)
		f.write('\n')

def load_history(filename="history.txt"):
	abs_path = os.path.dirname(os.path.abspath(__file__))+'/'+filename
	try:
		with open(abs_path, 'r', encoding='utf-8') as  f:
			content = []
			for js_object in f:
				print(json.loads(js_object))
				content.append(json.loads(js_object))
			return content
	except FileNotFoundError:
		print("LOG file not found")
	except Exception as e:
		print('LOG Error load history with error ', e)
	return None

def local_response(text, role="user"):
	global local_ai_model, sys_instr
	history = load_history()
	message ={'role': role, 'content': text}
	content = [*sys_instr, *history, message] if history != None else [sys_instr, message]
	response: ChatResponse = chat(model=local_ai_model, messages=content)
	save_history(message)
	print(content, '\n')
	result = response.message.content
	save_history({'role':'assistant', 'content':result})
	return result #Результат

def ethernet_response(text, role="user"):
	global ethernet_ai_model, sys_instr
	history = load_history()
	message ={'role': role, 'content': text}
	content = [*sys_instr, *history, message] if history != None else [sys_instr, message]
	completion = client.chat.completions.create(messages=content,model=ethernet_ai_model)
	save_history(message)
	print(content, '\n')
	result = completion.choices[0].message.content
	save_history({'role':'assistant', 'content':result})
	return result #Результат

def check_internet(host="8.8.8.8", port=53, timeout=3):
	print("checking internet connection")
	"""
	Проверка доступа в интернет через создание сокета.
	По умолчанию использует DNS-сервер Google (8.8.8.8) и порт 53.
	"""
	try:
		# Устанавливаем соединение, таймаут - 3 секунды
		socket.setdefaulttimeout(timeout)
		socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
		print("True")
		return True
	except socket.error:
		print("False")
		return False

vosk_path = os.path.dirname(os.path.abspath(__file__))+"/models/vosk-model-small-ru-0.22"
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
				if check_internet: text = ethernet_response(text)
				else:text = local_response(text)
				print(text)
				# Генерируем аудио и воспроизводим аудио Доступные голоса: ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
				audio = model.apply_tts(text=text,speaker='xenia', sample_rate=48000)
				sd.play(audio, samplerate=48000, latency='low', blocksize=256)
				sd.wait()
	except Exception as e:
		print("err", e)
