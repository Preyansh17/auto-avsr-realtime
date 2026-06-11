
from moviepy import VideoFileClip, ColorClip
print(f"Has resized: {hasattr(ColorClip(size=(100,100), color=(0,0,0)), 'resized')}")
print(f"Has resize: {hasattr(ColorClip(size=(100,100), color=(0,0,0)), 'resize')}")
print(f"Has with_start: {hasattr(ColorClip(size=(100,100), color=(0,0,0)), 'with_start')}")


