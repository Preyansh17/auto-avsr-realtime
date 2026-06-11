
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, ColorClip, CompositeAudioClip

# Paths
VIDEO_DIR = "/scratch/th3482/LipVideoData/patient_25p_crops_and_bbox4/patient_retinaface/patient_retinaface_video_seg24s/test"
AUDIO_DIR = "/home/th3482/auto-avsr/xtts_project/patient"
OUTPUT_DIR = "/home/th3482/auto-avsr/demo_videos"

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
INCLUDE_ORIGINAL_AUDIO = True
SHOW_DECODED_TEXT = False  # Set to True to show Decoded Text section (like demo2.py)

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

def get_decoded_text(sentence):
    """Convert groundtruth text to decoded text format"""
    if sentence.strip() == "Do you feel comfortable":
        return "You feel comfortable"
    elif sentence.strip() == "Faith is good":
        return "My faith is good"
    else:
        return sentence

def process_pair(video_path, audio_path, sentence, index):
    output_filename = os.path.join(OUTPUT_DIR, f"{index}.mp4")
    print(f"Processing {sentence} (index {index})...")
    print(f"Video: {video_path}")
    print(f"Audio: {audio_path}")
    
    # Load clips
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
    video_duration = video_clip.duration
    audio_duration = audio_clip.duration
    
    # Timeline
    # 0.0: Start
    # 0.0 -> video_duration: Video plays (with original audio optional)
    # video_duration: Video ends (freeze)
    # video_duration: GT Text appears
    # video_duration + 1.0: Waveform appears (and Decoded Text if enabled)
    # video_duration + 2.0: Decoded Audio plays (1.0s delay after waveform)
    # End: video_duration + 2.0 + audio_duration + 0.5 (buffer)
    
    waveform_start_time = video_duration + 1.0
    decoded_audio_start_time = waveform_start_time + 1.0
    
    total_duration = decoded_audio_start_time + audio_duration + 0.5
    
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
    # Max width 500, max height 500 (moderate size)
    # Center in left column (0-LEFT_W), below title (y > 160)
    max_vw, max_vh = 500, 500
    
    # Resize video
    # Always fit to box (upscale if needed as per user request to handle small videos)
    scale = min(max_vw / video_clip.w, max_vh / video_clip.h)
    video_disp = video_clip.resized(new_size=scale)
    
    # Center position in left column
    vx = (LEFT_W - video_disp.w) // 2
    vy = 160 + (920 - 160 - video_disp.h) // 2 # Center in remaining space
    
    # Video Playing
    vid_playing = video_disp.with_start(0).with_position((vx, vy))
    
    # Video Freeze (Last Frame)
    last_frame_img = video_disp.get_frame(video_duration)
    vid_freeze = ImageClip(last_frame_img).with_start(video_duration).with_duration(total_duration - video_duration).with_position((vx, vy))
    
    # Groundtruth Text Content
    # Appears at video_duration
    gt_text_img = create_text_image(sentence, RIGHT_W, 200, CONTENT_FONT_SIZE, align='center', color=GT_TEXT_COLOR)
    if SHOW_DECODED_TEXT:
        gt_clip = ImageClip(gt_text_img).with_start(video_duration).with_duration(total_duration - video_duration).with_position((LEFT_W, 140))
    else:
        gt_clip = ImageClip(gt_text_img).with_start(video_duration).with_duration(total_duration - video_duration).with_position((LEFT_W, 180))
    
    # Decoded Text Content - Only if SHOW_DECODED_TEXT
    dt_clip = None
    if SHOW_DECODED_TEXT:
        decoded_sentence = get_decoded_text(sentence)
        dt_text_img = create_text_image(decoded_sentence, RIGHT_W, 200, CONTENT_FONT_SIZE, align='center', color=DECODED_TEXT_COLOR)
        dt_clip = ImageClip(dt_text_img).with_start(waveform_start_time).with_duration(total_duration - waveform_start_time).with_position((LEFT_W, 430))
    
    # Waveform Content
    # Appears at waveform_start_time
    wave_img_path = f"temp_wave_{index}.png"
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
    # 1. Original video audio (from 0 to video_duration) - Optional
    # 2. Decoded speech audio (from decoded_audio_start_time)
    
    audio_list = []
    if INCLUDE_ORIGINAL_AUDIO and video_clip.audio is not None:
        audio_list.append(video_clip.audio.with_start(0))
    
    decoded_audio_track = audio_clip.with_start(decoded_audio_start_time)
    audio_list.append(decoded_audio_track)
    
    mixed_audio = CompositeAudioClip(audio_list)
    
    # Composite - Build clip list conditionally
    clip_list = [
        bg_clip,
        t_video_clip, t_gt_clip, t_ds_clip,
        vid_playing, vid_freeze,
        gt_clip,
        wave_clip
    ]
    
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
        
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    print(f"Found {len(audio_files)} audio files.")
    print(f"SHOW_DECODED_TEXT: {SHOW_DECODED_TEXT}")
    
    processed_count = 0
    index = 1
    for audio_path in sorted(audio_files):
        filename = os.path.basename(audio_path)
        sentence = os.path.splitext(filename)[0]
        
        # Find video
        pattern = os.path.join(VIDEO_DIR, f"{sentence}_*.mp4")
        candidates = glob.glob(pattern)
        
        if not candidates:
            print(f"No video match for: {sentence}")
            continue
            
        video_path = candidates[0]
        process_pair(video_path, audio_path, sentence, index)
        
        processed_count += 1
        index += 1
        # if processed_count >= 1: # Limit removed
        #    print("Created 1 demo video. Stopping.")
        #    break

if __name__ == "__main__":
    main()
