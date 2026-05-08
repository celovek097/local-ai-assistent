from ollama import chat, ChatResponse
from cerebras.cloud.sdk import Cerebras, APIError
from vosk import Model, KaldiRecognizer
import os
import dotenv
import json
import pyaudio
import torch
import sounddevice as sd

print_api_error = True # False to off print "API error, check ethernet connection and API key"
local_ai_model = "gemma3:latest"
ethernet_ai_model = "llama3.1-8b"
system_instruction = "Ты Стелла, голосовой ассистент, общайся с пользователем как с другом, поддерживай простой диалог, всегда отвечай на русском и только буквами, всегда укладывай ответ в 1000 символов."

sys_instr = [{'role': 'system', 'content': system_instruction}, {'role': 'assistant', 'content': 'Здравствуйте!'}]

dotenv.load_dotenv()
api_key = os.getenv("API_KEY")
client = Cerebras(api_key=api_key)

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

def response(text, role="user"):
	global sys_instr, print_api_error
	history = load_history()
	message = {'role': role, 'content': text}
	content = [*sys_instr, *history, message] if history != None else [*sys_instr, message]
	try:
		result = ethernet_response(content)
	except APIError:
		print("API error, check ethernet connection and API key") if print_api_error else None
		result = local_response(content)
	save_history(message)
	print(content, '\n')
	save_history({'role': 'assistant', 'content': result})
	return result  # Результат

def local_response(content):
	global local_ai_model
	response: ChatResponse = chat(model=local_ai_model, messages=content)
	return response.message.content

def ethernet_response(content):
	global ethernet_ai_model
	completion = client.chat.completions.create(messages=content,model=ethernet_ai_model)
	return completion.choices[0].message.content #Результат

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
				text = response(text)
				print(text)
				# Генерируем аудио и воспроизводим аудио Доступные голоса: ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
				audio = model.apply_tts(text=text,speaker='xenia', sample_rate=48000)
				sd.play(audio, samplerate=48000, latency='low', blocksize=256)
				sd.wait()
	except Exception as e:
		print("err", e)
