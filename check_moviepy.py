
import moviepy
print(f"Moviepy version: {moviepy.__version__}")
try:
    from moviepy import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, ImageClip
    print("Import from moviepy successful")
except ImportError:
    print("Import from moviepy failed")

try:
    import moviepy.editor as mp
    print("Import moviepy.editor successful")
except ImportError:
    print("Import moviepy.editor failed")


