from dotenv import load_dotenv
from openai import OpenAI
import os
import base64

load_dotenv()

client = OpenAI()


def stt(audio):
    # 파일로 변환
    file_name = 'prompt.mp3'
    audio.export(file_name, format='mp3')

    # whisper-1 모델로 stt
    with open(file_name, 'rb') as f:
        transcription = client.audio.transcriptions.create(
            model='whisper-1',
            file=f
        )
    
    # 음원파일 삭제 
    os.remove(file_name)
    
    return transcription.text

def ask_gpt(messages, model):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        top_p=1,
        max_tokens=4096
    )
    
    return response.choices[0].message.content


def tts(response):
    file_name = 'voice.mp3'
    with client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='ash',
        input=response
    ) as stream:
        stream.stream_to_file(file_name)
        
    # 음원을 base64로 인코딩처리
    with open(file_name, 'rb') as f:
        data = f.read()
        base64_encoded = base64.b64encode(data).decode()
        
    # 음원파일 삭제
    os.remove(file_name)
    return base64_encoded
