import whisper
import os
# pip install openai-whisper
# ffmpeg : https://www.cyrilvincent.com/download/ffmpeg-2023-11-20-git-e56d91f8a8-essentials_build.7z

os.environ["path"] += ";c:\\ffmpeg\\bin"

path = "data/chatgpt/python.mp3"
model = whisper.load_model("base")
result = model.transcribe(path)
print(result["text"])
with open("data/chatgpt/python.txt", "w") as f:
    f.write(result["text"])