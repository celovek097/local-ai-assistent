pip install ollama, torch, silero, sounddevice, vosk, pyaudio, json, os, dotenv, Cerebras, cerebras.cloud.sdk, socket # возможно не все расширения

install your language model for vosk // скачивание vosk модели на вашем языке
https://alphacephei.com/vosk/models

скаченую папку положить в models
models должна находится в одной папке с .py файлом
```
local_ai
|
|-test.py
|-.env
|-history.txt
|-models
    |-vosk-model-small-ru-0.22
```


bash/cmd

ollama push gemma3:latest (или ваша желаемая модель)
