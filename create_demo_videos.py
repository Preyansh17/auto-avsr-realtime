
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, ColorClip, CompositeAudioClip

# Paths
# Video path option: 'a' or 'b'
VIDEO_PATH_OPTION = 'a'  # 'a' for bbox_seg24s, 'b' for video_seg24s

if VIDEO_PATH_OPTION == 'a':
    VIDEO_DIR = "/scratch/th3482/LipVideoData/video_300_25p_crops_and_bbox5/video300_mediapipe/video300_mediapipe_video_bbox_seg24s/val"
elif VIDEO_PATH_OPTION == 'b':
    VIDEO_DIR = "/scratch/th3482/LipVideoData/video_300_25p_crops_and_bbox5/video300_mediapipe/video300_mediapipe_video_seg24s/val"
else:
    raise ValueError("VIDEO_PATH_OPTION must be 'a' or 'b'")

AUDIO_DIR = "/home/th3482/auto-avsr/xtts_project/legal_outputs_terry_v2_1.3x"
OUTPUT_DIR = "/home/th3482/auto-avsr/tianyu_videos_bbx_decodespeech"

# Settings
W, H = 1920, 1080
SPLIT_X = 750 # Split between left and right columns (Left width = 750, Right width = 1170)
LEFT_W = SPLIT_X
RIGHT_W = W - SPLIT_X

BG_COLOR = (0, 0, 0) # Black
TITLE_COLOR = (255, 255, 255) # White
GT_TEXT_COLOR = (0, 255, 0) # Green
DECODED_TEXT_COLOR = (0, 255, 255) # Cyan
TITLE_FONT_SIZE = 55
CONTENT_FONT_SIZE = 60

# Options
INCLUDE_ORIGINAL_AUDIO = False
SHOW_DECODED_TEXT = False  # Set to True to show Decoded Text section (like demo2.py)

def extract_number_and_text(filename):
    """Extract number and text from filename like '12.They will have it for you..wav' or '12.They will have it for you.mp4'"""
    # Remove extension
    basename = os.path.splitext(filename)[0]
    # Handle double extension for audio files (..wav)
    if basename.endswith('.'):
        basename = basename[:-1]
    
    # Match pattern: number.text
    match = re.match(r'^(\d+)\.(.+)$', basename)
    if match:
        number = int(match.group(1))
        text = match.group(2)
        return number, text
    return None, None

def create_text_image(text, width, height, font_size=40, align='center', color=TITLE_COLOR):
    img = Image.new('RGB', (width, height), color=BG_COLOR)
    d = ImageDraw.Draw(img)
    
    # Try to find a font
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf"
    ]
    font = None
    try:
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
        if font is None:
             # Fallback to PIL default (very small)
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Draw text
    # We use basic positioning
    bbox = d.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    if align == 'center':
        x = (width - text_w) // 2
        y = (height - text_h) // 2
    else:
        x = 10
        y = (height - text_h) // 2
        
    d.text((x, y), text, fill=color, font=font)
    return np.array(img)

def create_waveform_image(audio_path, output_path, width_px, height_px):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(width_px/100, height_px/100), dpi=100)
    plt.axis('off')
    # Plot with cyan color on black background
    librosa.display.waveshow(y, sr=sr, color='cyan', axis='off')
    
    # Remove margins
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_facecolor('black')
    
    # Save
    plt.savefig(output_path, facecolor='black', bbox_inches='tight', pad_inches=0)
    plt.close()

def process_pair(video_path, audio_path, video_num, video_text, audio_num, audio_text):
    # Output filename: gd_视频序号后句子_pred_音频序号后句子.mp4
    output_filename = os.path.join(OUTPUT_DIR, f"gd_{video_text}_pred_{audio_text}.mp4")
    print(f"Processing video #{video_num} and audio #{audio_num}...")
    print(f"Video: {video_path}")
    print(f"Audio: {audio_path}")
    print(f"Video text: {video_text}")
    print(f"Audio text: {audio_text}")
    
    # Load clips
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
    video_duration = video_clip.duration
    audio_duration = audio_clip.duration
    
    # Timeline - Different based on INCLUDE_ORIGINAL_AUDIO
    if not INCLUDE_ORIGINAL_AUDIO:
        # New timeline when original audio is off:
        # 0.0: GT Text appears (stays visible throughout)
        # 1.0: Video starts playing
        # 1.0 + video_duration: Video ends
        # 1.0 + video_duration + 1.0: Waveform appears (and Decoded Text if enabled)
        # 1.0 + video_duration + 1.0 + 1.0: Decoded Audio plays
        # End: 1.0 + video_duration + 1.0 + 1.0 + audio_duration + 0.5
        
        video_start_time = 1.0
        video_end_time = video_start_time + video_duration
        waveform_start_time = video_end_time + 1.0
        decoded_audio_start_time = waveform_start_time + 1.0
        total_duration = decoded_audio_start_time + audio_duration + 0.5
        gt_text_start_time = 0.0
        gt_text_duration = total_duration  # Show throughout the entire video
    else:
        # Original timeline when original audio is on:
        # 0.0: Video plays (with original audio)
        # video_duration: Video ends (freeze)
        # video_duration: GT Text appears
        # video_duration + 1.0: Waveform appears (and Decoded Text if enabled)
        # video_duration + 2.0: Decoded Audio plays (1.0s delay after waveform)
        # End: video_duration + 2.0 + audio_duration + 0.5
        
        video_start_time = 0.0
        video_end_time = video_duration
        waveform_start_time = video_duration + 1.0
        decoded_audio_start_time = waveform_start_time + 1.0
        total_duration = decoded_audio_start_time + audio_duration + 0.5
        gt_text_start_time = video_duration
        gt_text_duration = total_duration - gt_text_start_time  # Show until end
    
    # --- Layout ---
    # W=1920, H=1080
    # Left Column: x=0, w=LEFT_W (750). Title "Video" at top. Video below.
    # Right Column: x=LEFT_W, w=RIGHT_W (1170). 
    #   If SHOW_DECODED_TEXT:
    #     Top Module: "Groundtruth Text"
    #     Middle Module: "Decoded Text"
    #     Bottom Module: "Decoded Speech"
    #   Else:
    #     Top Module: "Groundtruth Text"
    #     Bottom Module: "Decoded Speech"
    
    # Background
    bg_clip = ColorClip(size=(W, H), color=BG_COLOR).with_duration(total_duration)
    
    # Titles
    title_h = 120 # Increased height for larger title font
    
    # Left Title (Centered in 0 to LEFT_W)
    t_video_img = create_text_image("Video", LEFT_W, title_h, TITLE_FONT_SIZE, align='center', color=TITLE_COLOR)
    t_video_clip = ImageClip(t_video_img).with_duration(total_duration).with_position((0, 40))
    
    # Right Top Title (Groundtruth Text)
    t_gt_img = create_text_image("Groundtruth Text", RIGHT_W, title_h, TITLE_FONT_SIZE, align='center', color=TITLE_COLOR)
    if SHOW_DECODED_TEXT:
        t_gt_clip = ImageClip(t_gt_img).with_duration(total_duration).with_position((LEFT_W, 30))
    else:
        t_gt_clip = ImageClip(t_gt_img).with_duration(total_duration).with_position((LEFT_W, 40))
    
    # Right Middle Title (Decoded Text) - Only if SHOW_DECODED_TEXT
    t_dt_clip = None
    if SHOW_DECODED_TEXT:
        t_dt_img = create_text_image("Decoded Text", RIGHT_W, title_h, TITLE_FONT_SIZE, align='center', color=TITLE_COLOR)
        t_dt_clip = ImageClip(t_dt_img).with_duration(total_duration).with_position((LEFT_W, 320))
    
    # Right Bottom Title (Decoded Speech)
    t_ds_img = create_text_image("Decoded Speech", RIGHT_W, title_h, TITLE_FONT_SIZE, align='center', color=TITLE_COLOR)
    if SHOW_DECODED_TEXT:
        t_ds_clip = ImageClip(t_ds_img).with_duration(total_duration).with_position((LEFT_W, 620))
    else:
        t_ds_clip = ImageClip(t_ds_img).with_duration(total_duration).with_position((LEFT_W, 450))
    
    # Video Content
    # Max width and height - larger for option 'a'
    # Center in left column (0-LEFT_W), below title (y > 160)
    if VIDEO_PATH_OPTION == 'a':
        max_vw, max_vh = 700, 700  # Larger size for option 'a'
    else:
        max_vw, max_vh = 500, 500  # Moderate size for option 'b'
    
    # Resize video
    # Always fit to box (upscale if needed as per user request to handle small videos)
    scale = min(max_vw / video_clip.w, max_vh / video_clip.h)
    video_disp = video_clip.resized(new_size=scale)
    
    # Center position in left column
    vx = (LEFT_W - video_disp.w) // 2
    vy = 160 + (920 - 160 - video_disp.h) // 2 # Center in remaining space
    
    # Video Playing
    vid_playing = video_disp.with_start(video_start_time).with_position((vx, vy))
    
    # Video Freeze (Last Frame) - only if video ends before total duration
    vid_freeze = None
    if video_end_time < total_duration:
        last_frame_img = video_disp.get_frame(video_duration)
        vid_freeze = ImageClip(last_frame_img).with_start(video_end_time).with_duration(total_duration - video_end_time).with_position((vx, vy))
    
    # Groundtruth Text Content - Display video text
    gt_text_img = create_text_image(video_text, RIGHT_W, 200, CONTENT_FONT_SIZE, align='center', color=GT_TEXT_COLOR)
    if SHOW_DECODED_TEXT:
        gt_clip = ImageClip(gt_text_img).with_start(gt_text_start_time).with_duration(gt_text_duration).with_position((LEFT_W, 140))
    else:
        gt_clip = ImageClip(gt_text_img).with_start(gt_text_start_time).with_duration(gt_text_duration).with_position((LEFT_W, 180))
    
    # Decoded Text Content - Display audio text - Only if SHOW_DECODED_TEXT
    dt_clip = None
    if SHOW_DECODED_TEXT:
        dt_text_img = create_text_image(audio_text, RIGHT_W, 200, CONTENT_FONT_SIZE, align='center', color=DECODED_TEXT_COLOR)
        dt_clip = ImageClip(dt_text_img).with_start(waveform_start_time).with_duration(total_duration - waveform_start_time).with_position((LEFT_W, 430))
    
    # Waveform Content
    # Appears at waveform_start_time
    wave_img_path = f"temp_wave_{audio_num}_{audio_text.replace(' ', '_')}.png"
    # Increase waveform width since right column is wider
    wave_w = 1000
    wave_h = 300
    create_waveform_image(audio_path, wave_img_path, wave_w, wave_h)
    
    wave_clip = ImageClip(wave_img_path).with_start(waveform_start_time).with_duration(total_duration - waveform_start_time)
    # Center waveform below title in right column
    wx = LEFT_W + (RIGHT_W - wave_w) // 2
    if SHOW_DECODED_TEXT:
        wy = 740 # Explicit position
    else:
        wy = 600 # Explicit position
    wave_clip = wave_clip.with_position((wx, wy))
    
    # Audio Mix
    # 1. Original video audio (from video_start_time) - Optional
    # 2. Decoded speech audio (from decoded_audio_start_time)
    
    audio_list = []
    if INCLUDE_ORIGINAL_AUDIO and video_clip.audio is not None:
        audio_list.append(video_clip.audio.with_start(video_start_time))
    
    decoded_audio_track = audio_clip.with_start(decoded_audio_start_time)
    audio_list.append(decoded_audio_track)
    
    mixed_audio = CompositeAudioClip(audio_list)
    
    # Composite - Build clip list conditionally
    clip_list = [
        bg_clip,
        t_video_clip, t_gt_clip, t_ds_clip,
        vid_playing,
        gt_clip,
        wave_clip
    ]
    
    # Add video freeze frame if exists
    if vid_freeze is not None:
        clip_list.insert(4, vid_freeze)  # Insert after vid_playing
    
    # Add decoded text elements if enabled
    if SHOW_DECODED_TEXT:
        clip_list.insert(3, t_dt_clip)  # Insert after t_gt_clip
        clip_list.append(dt_clip)  # Add decoded text clip
    
    final = CompositeVideoClip(clip_list)
    
    final = final.with_audio(mixed_audio)
    
    final.write_videofile(output_filename, fps=24, codec='libx264', audio_codec='aac')
    
    # Cleanup
    if os.path.exists(wave_img_path):
        os.remove(wave_img_path)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"Video path option: {VIDEO_PATH_OPTION}")
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"SHOW_DECODED_TEXT: {SHOW_DECODED_TEXT}")
    
    # Load all audio files and extract numbers
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    audio_dict = {}  # {number: (path, text)}
    
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        num, text = extract_number_and_text(filename)
        if num is not None:
            audio_dict[num] = (audio_path, text)
    
    print(f"Found {len(audio_dict)} audio files.")
    
    # Load all video files and extract numbers
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    video_dict = {}  # {number: (path, text)}
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        num, text = extract_number_and_text(filename)
        if num is not None:
            video_dict[num] = (video_path, text)
    
    print(f"Found {len(video_dict)} video files.")
    
    # Match by number
    processed_count = 0
    for num in sorted(audio_dict.keys()):
        # Skip number 9
        if num == 9:
            print(f"Skipping number {num}")
            continue
        
        if num not in video_dict:
            print(f"No video match for audio number {num}")
            continue
        
        audio_path, audio_text = audio_dict[num]
        video_path, video_text = video_dict[num]
        
        process_pair(video_path, audio_path, num, video_text, num, audio_text)
        
        processed_count += 1
        if processed_count >= 10:
            print(f"Reached limit of 10 videos. Stopping.")
            break
    
    print(f"Processed {processed_count} pairs.")

if __name__ == "__main__":
    main()
