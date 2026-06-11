
from moviepy import VideoFileClip, ColorClip
import inspect

clip = ColorClip(size=(100,100), color=(0,0,0))
print(inspect.signature(clip.resized))
print(clip.resized.__doc__)


