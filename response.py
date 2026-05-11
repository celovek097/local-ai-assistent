from ollama import chat, ChatResponse
import json
from os import path

local_ai_model = "gemma3:latest"
system_instruction = "Ты Стелла, голосовой ассистент, общайся с пользователем как с другом, поддерживай простой диалог, всегда отвечай в женском роде на русском и только буквами(никаких спец символов), всегда укладывай ответ в 1000 символов."

sys_instr = [{'role': 'system', 'content': system_instruction}, {'role': 'assistant', 'content': 'Здравствуйте!'}]

def clear_history(filename="history.txt"):
    try:
        abs_path = path.dirname(path.abspath(__file__))+'/'+filename
        with open(abs_path, 'w', encoding='utf-8'):
            print(' -------------\n',' clear history success\n','-------------')
    except:
        print("error clear history")

def save_history(history, filename="history.txt"):
    abs_path = path.dirname(path.abspath(__file__))+'/'+filename
    with open(abs_path, 'a', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False)
        f.write('\n')

def load_history(filename="history.txt"):
    abs_path = path.dirname(path.abspath(__file__))+'/'+filename
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
