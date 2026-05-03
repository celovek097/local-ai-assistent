# Локальный голосовой ИИ-ассистент

Голосовой ассистент с полностью локальной обработкой:

- STT — Vosk (распознавание речи)
- LLM — Ollama (генерация ответа)
- TTS — Piper (синтез речи)

Целевая ОС: Linux. Основной язык — русский.

## Структура

.
├── main.py             # точка входа
├── config.yaml         # вся конфигурация
├── requirements.txt
├── assistant/
│   ├── config.py       # загрузка YAML-конфига
│   ├── history.py      # JSONL-история диалога
│   ├── llm.py          # обёртка над Ollama
│   ├── stt.py          # Vosk + PyAudio
│   ├── tts.py          # Piper + sounddevice
│   └── assistant.py    # цикл «слушаю → отвечаю → озвучиваю»
└── models/             # сюда кладутся модели Vosk и Piper

## Установка (Linux)

### 1. Системные зависимости

Bash

sudo apt update
sudo apt install -y python3 python3-venv python3-pip \
    portaudio19-dev libportaudio2 ffmpeg

portaudio19-dev нужен для сборки PyAudio, libportaudio2 — для sounddevice.

### 2. Ollama

Bash

curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3

Любую другую модель можно прописать в config.yaml → llm.model.

### 3. Python-окружение

Bash

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 4. Модель Vosk (STT)

Скачайте с https://alphacephei.com/vosk/models нужную русскую модель,
например vosk-model-small-ru-0.22, распакуйте и положите в models/:

models/vosk-model-small-ru-0.22/

Путь меняется в config.yaml → stt.model_path.

### 5. Голос Piper (TTS)

Скачайте .onnx и .onnx.json нужного русского голоса с
https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU и положите
оба файла в models/. Доступные русские голоса:

- ru_RU-irina-medium
- ru_RU-denis-medium
- ru_RU-dmitri-medium
- ru_RU-ruslan-medium

Путь к голосу — config.yaml → tts.voice.

## Запуск

Bash

source .venv/bin/activate
python main.py

Флаги:

-c PATH, --config PATH    путь к альтернативному YAML-конфигу
-v, --verbose             подробный лог

Завершить работу — произнесите фразу из stt.exit_phrase (по-умолчанию
«выход») или нажмите Ctrl+C.

## Настройки TTS

В config.yaml, секция tts:

| Параметр        | Описание                                                     |
|-----------------|--------------------------------------------------------------|
| voice         | путь к .onnx файлу голоса Piper                            |
| length_scale  | темп речи (<1 быстрее, >1 медленнее)                     |
| noise_scale   | вариативность интонации (0.0–1.0)                            |
| noise_w       | вариативность длительностей фонем (0.0–1.0)                  |
| volume        | масштаб громкости (0.0–2.0)                                  |
| output_device | ID/имя аудио-устройства sounddevice (null = по-умолчанию)  |

Сменить голос — поменяйте путь в tts.voice (модель и её .onnx.json
должны лежать рядом). Список устройств:

Bash

python -c "import sounddevice as sd; print(sd.query_devices())"

## Настройки LLM

В config.yaml, секция llm:

- model — имя модели Ollama (ollama list покажет установленные).
- system_prompt — системный промпт.
- history_window — сколько последних пар сообщений подавать в контекст.
- options — параметры генерации Ollama (temperature, top_p,
  num_ctx, и т.д.).

История сохраняется построчно в JSONL (history.file); очистку при
запуске можно выключить (history.clear_on_start: false)
