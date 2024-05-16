from openai import OpenAI 
import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

MODEL="gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

1 - Basic Test

completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": "You are a helpful coding assistant"},
    {"role": "user", "content": "Hello! Could you provide solution for 2 sum problem"}
  ]
)

print("Assistant: " + completion.choices[0].message.content)

# 2 : Image Encoding: Convert Image to Base64 String for Image conversations 
import base64

IMAGE_PATH = "math.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)


response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
        {"role": "user", "content": [
            {"type": "text", "text": "can you please solve this math problem"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)



import cv2
from moviepy.editor import VideoFileClip
import time
import base64

VIDEO_PATH = "mkbsdIpad.mp4"

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path

def transcribe_audio(audio_path):
    transcription_response = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb"),
    )
    return transcription_response.text


base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)
transcription_text = transcribe_audio(audio_path)  # Transcribe the extracted audio

response = client.chat.completions.create(
    model=MODEL,
    messages=[
    {"role": "system", "content":"""You are generating a video summary. Create a summary of the provided video and its transcript. Also provide the details about the dress and color they are wearing"""},
    {"role": "user", "content": [
        "These are the frames from the video.",
        *map(lambda x: {"type": "image_url", 
                        "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
                        {"type": "text", "text": f"The audio transcription is: {transcription_text}"}
        ],
    }
],
    temperature=0,
)

print(response.choices[0].message.content)