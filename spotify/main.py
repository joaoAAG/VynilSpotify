import yt_dlp as youtube_dl
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TCON, TRCK
import os
import random
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageFilter, ImageOps
from colorthief import ColorThief  # Install colorthief using: pip install colorthief
import pygame
import requests
import customtkinter as ctk  # Ensure you have customtkinter installed
import cv2
import mediapipe as mp
from threading import Thread
import numpy as np
from ultralytics import YOLO  # Import YOLO from ultralytics

# Initialize pygame mixer
pygame.mixer.init()

last_downloaded_song = None
first_play_click = True
current_song_index = -1
song_list = []
paused = False
current_song_duration = 0
gesture_detection_active = True  # Variable to control gesture detection

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Gesture state
fist_start_time = None
fist_duration = 1.0  # Minimum time to hold fist to select button
hovered_button = None

# Load the YOLO model
model_path = 'C:\\Users\\joaoa\\PycharmProjects\\spotify\\runs\\train-ALL\\weights\\best.pt'
print(f"Loading model from {model_path}")
model = YOLO(model_path)
print("Model loaded successfully")

# Initialize album recognition state
recognized_album = None
album_recognition_start_time = None


def is_fist_gesture(landmarks):
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    # Check if all fingers are curled (forming a fist)
    if (index_tip.y > index_mcp.y and middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y and pinky_tip.y > pinky_mcp.y):
        return True
    return False


def download_song(url, output_path):
    global last_downloaded_song
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_id = info_dict.get("id", None)
        mp3_path = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.mp3'
        last_downloaded_song = mp3_path
        thumbnail_path = mp3_path.replace('.mp3', '.jpg')
        download_thumbnail(video_id, thumbnail_path)
        update_song_list()
        return video_id, mp3_path


def download_thumbnail(video_id, output_image_path):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    response = requests.get(thumbnail_url)
    if response.status_code == 200:
        with open(output_image_path, 'wb') as file:
            file.write(response.content)


def tag_song(file_path, title, artist, album, genre, track_number):
    audio = MP3(file_path, ID3=ID3)
    audio['TIT2'] = TIT2(encoding=3, text=title)
    audio['TPE1'] = TPE1(encoding=3, text=artist)
    audio['TALB'] = TALB(encoding=3, text=album)
    audio['TCON'] = TCON(encoding=3, text=genre)
    audio['TRCK'] = TRCK(encoding=3, text=track_number)
    audio.save()


def play_music(file_path):
    global paused, current_song_duration
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    play_pause_btn.configure(image=pause_img)
    thumbnail_path = file_path.replace('.mp3', '.jpg')
    update_album_cover(thumbnail_path)
    print(f"Playing: {file_path}")
    paused = False

    audio = MP3(file_path)
    current_song_duration = audio.info.length
    update_time_visualizer()
    root.after(1000, check_music_status)  # Check the music status every second


def check_music_status():
    if not pygame.mixer.music.get_busy() and not paused:
        play_next_song()
    else:
        root.after(1000, check_music_status)  # Continue checking every second


def toggle_play_pause():
    global first_play_click, paused
    print("Toggle play/pause")
    if first_play_click:
        first_play_click = False
        play_last_or_random()
    else:
        if pygame.mixer.music.get_busy() and not paused:
            pygame.mixer.music.pause()
            play_pause_btn.configure(image=play_img)
            print("Music paused")
            paused = True
        elif paused:
            pygame.mixer.music.unpause()
            play_pause_btn.configure(image=pause_img)
            print("Music resumed")
            paused = False
        else:
            print("No music is playing")


def play_last_or_random():
    global last_downloaded_song, song_list, current_song_index
    update_song_list()
    if last_downloaded_song and os.path.exists(last_downloaded_song):
        play_music(last_downloaded_song)
    elif song_list:
        current_song_index = random.randint(0, len(song_list) - 1)
        play_music(song_list[current_song_index])
    else:
        print("No music files found in the Music directory.")


def play_next_song():
    global current_song_index, song_list, paused
    if song_list:
        current_song_index = (current_song_index + 1) % len(song_list)
        play_music(song_list[current_song_index])
        print(f"Next song: {song_list[current_song_index]}")
        paused = False


def play_previous_song():
    global current_song_index, song_list, paused
    if song_list:
        current_song_index = (current_song_index - 1) % len(song_list)
        play_music(song_list[current_song_index])
        print(f"Previous song: {song_list[current_song_index]}")
        paused = False


def stop_music():
    global paused
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        play_pause_btn.configure(image=play_img)
        print("Music stopped")
        paused = False


def download_and_tag():
    url = url_entry.get()
    title = 'Unknown Title'
    artist = 'Unknown Artist'
    album = 'Unknown Album'
    genre = 'Unknown Genre'
    track_number = '0'

    output_path = filedialog.askdirectory()
    if not output_path:
        return

    video_id, mp3_path = download_song(url, output_path)
    tag_song(mp3_path, title, artist, album, genre, track_number)


def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        play_music(file_path)


def update_album_cover(image_path):
    if not os.path.exists(image_path):
        image_path = 'default_image.png'  # Correct path to your default image
    img = Image.open(image_path)
    img = img.resize((400, 400), Image.LANCZOS).filter(ImageFilter.GaussianBlur(5))
    ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(400, 400))
    album_cover_label.configure(image=ctk_image)
    album_cover_label.image = ctk_image
    print(f"Album cover updated: {image_path}")

    # Extract dominant color from the album image and change the frame color
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    darker_color = tuple(max(0, c - 20) for c in dominant_color)  # Reduce each RGB component by 20
    hex_color = f'#{darker_color[0]:02x}{darker_color[1]:02x}{darker_color[2]:02x}'

    # Update the main frame color
    frame.configure(fg_color=hex_color)

    # Update UI buttons color (background and border)
    play_pause_frame.configure(fg_color=hex_color)
    volume_frame.configure(fg_color=hex_color)
    play_pause_btn.configure(fg_color=hex_color, border_color=hex_color)
    previous_btn.configure(fg_color=hex_color, border_color=hex_color)
    next_btn.configure(fg_color=hex_color, border_color=hex_color)
    toggle_download_btn.configure(fg_color=hex_color, border_color=hex_color)

    # Update the volume icon color
    update_icon_color("volume_icon.png", "volume_icon_colored.png", darker_color)
    volume_icon = ctk.CTkImage(light_image=Image.open("volume_icon_colored.png"), size=(30, 30))
    volume_icon_label.configure(image=volume_icon)
    volume_icon_label.image = volume_icon


def update_icon_color(input_path, output_path, color):
    with Image.open(input_path) as img:
        img = img.convert("RGBA")
        data = np.array(img)
        r, g, b, a = data.T
        # Identify black areas: pixels with all RGB components less than a threshold (e.g., 50)
        black_areas = (r < 50) & (g < 50) & (b < 50)
        data[..., :-1][black_areas.T] = color
        img = Image.fromarray(data)
        img.save(output_path)


def update_song_list():
    global song_list
    music_folder = os.path.expanduser("~/Music")
    song_list = [os.path.join(music_folder, f) for f in os.listdir(music_folder) if f.endswith('.mp3')]
    print(f"Song list updated: {song_list}")


def toggle_download_widgets():
    if download_widgets_frame.winfo_ismapped():
        download_widgets_frame.pack_forget()
    else:
        download_widgets_frame.pack(padx=10, pady=10)


def update_time_visualizer():
    if pygame.mixer.music.get_busy() or paused:
        current_time = pygame.mixer.music.get_pos() / 1000
        total_time = current_song_duration
        time_label.configure(text=f"{format_time(current_time)} / {format_time(total_time)}")
        time_label.after(1000, update_time_visualizer)


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}"


def set_volume(val):
    volume = float(val) / 100
    pygame.mixer.music.set_volume(volume)


def get_hovered_button(index_finger_tip_x, index_finger_tip_y, frame_width, frame_height):
    # Calculate the relative position of the index finger tip
    rel_x = (index_finger_tip_x - frame_width / 2) / (frame_width / 2)
    rel_y = (index_finger_tip_y - frame_height / 2) / (frame_height / 2)

    if rel_x < -0.5 and abs(rel_y) < 0.5:
        return previous_btn
    elif rel_x > 0.5 and abs(rel_y) < 0.5:
        return next_btn
    elif rel_y > 0.5 and abs(rel_x) < 0.5:
        return scan_btn
    elif rel_y < -0.5 and abs(rel_x) < 0.5:
        return toggle_download_btn
    elif abs(rel_x) < 0.5 and abs(rel_y) < 0.5:
        return play_pause_btn
    return None


def detect_gestures():
    cap = cv2.VideoCapture(0)
    global fist_start_time, hovered_button

    while gesture_detection_active:  # Check the control variable
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_height, frame_width, _ = frame.shape
        # Draw grid lines
        cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (0, 255, 0), 2)
        cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2), (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                if is_fist_gesture(landmarks):
                    if fist_start_time is None:
                        fist_start_time = time.time()
                    elif time.time() - fist_start_time >= fist_duration:
                        if hovered_button:
                            hovered_button.invoke()  # Simulate button click
                            fist_start_time = None
                            hovered_button = None
                else:
                    fist_start_time = None

                index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_tip_x = int(index_finger_tip.x * frame_width)
                index_finger_tip_y = int(index_finger_tip.y * frame_height)

                hovered_button = get_hovered_button(index_finger_tip_x, index_finger_tip_y, frame_width, frame_height)

                # Reset all button colors
                play_pause_btn.configure(fg_color="gray")
                previous_btn.configure(fg_color="gray")
                next_btn.configure(fg_color="gray")
                scan_btn.configure(fg_color="gray")
                toggle_download_btn.configure(fg_color="gray")

                # Highlight the hovered button with a darker color
                if hovered_button:
                    hovered_button.configure(fg_color="darkgreen")

        # Display the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Thread to run gesture detection in parallel
gesture_thread = Thread(target=detect_gestures)
gesture_thread.daemon = True
gesture_thread.start()


def start_yolo_detection():
    global gesture_detection_active
    gesture_detection_active = False  # Stop gesture detection
    print("Gesture detection turned off")

    yolo_thread = Thread(target=perform_yolo_detection)
    yolo_thread.daemon = True
    yolo_thread.start()


def perform_yolo_detection():
    global recognized_album, album_recognition_start_time, gesture_detection_active
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame, conf=0.25)  # Lower the confidence threshold if needed
        print("Inference results:", results)  # Print results to debug

        current_album = None
        # Check if any album is detected
        for result in results:
            boxes = result.boxes
            for box in boxes:
                label = model.names[int(box.cls[0])]
                current_album = label  # Assuming the label is the album name
                break

        if current_album:
            if current_album == recognized_album:
                if time.time() - album_recognition_start_time >= 5:
                    download_and_play_album_music(current_album)
                    recognized_album = None  # Reset to avoid repeated downloads
                    gesture_detection_active = True  # Turn on gesture detection after music starts playing
                    gesture_thread = Thread(target=detect_gestures)
                    gesture_thread.daemon = True
                    gesture_thread.start()
            else:
                recognized_album = current_album
                album_recognition_start_time = time.time()
        else:
            recognized_album = None

        # Display the frame
        cv2.imshow('YOLOv8 Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def download_and_play_album_music(album_name):
    search_query = album_name
    output_path = os.path.expanduser("~/Music")

    # Use YouTube search to find the album music
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            results = ydl.extract_info(f"ytsearch:{search_query}", download=False)['entries']
            for result in results:
                if result['duration'] < 600:  # Check if the song is less than 10 minutes
                    ydl.download([result['webpage_url']])
                    mp3_path = ydl.prepare_filename(result).rsplit('.', 1)[0] + '.mp3'
                    thumbnail_path = mp3_path.replace('.mp3', '.jpg')
                    download_thumbnail(result['id'], thumbnail_path)
                    play_music(mp3_path)
                    break
        except Exception as e:
            print(f"Error downloading album music: {e}")


root = ctk.CTk()
root.title("Minimalist Music Player")
root.geometry("500x700")

frame = ctk.CTkFrame(root, fg_color='black')
frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Frame for download widgets
download_widgets_frame = ctk.CTkFrame(root, fg_color='black')
download_widgets_frame.pack(padx=10, pady=10)
download_widgets_frame.pack_forget()  # Initially hidden

ctk.CTkLabel(download_widgets_frame, text="YouTube URL:", text_color='white').pack(pady=10)
url_entry = ctk.CTkEntry(download_widgets_frame, width=500)
url_entry.pack(pady=10)

download_button = ctk.CTkButton(download_widgets_frame, text="Download", corner_radius=32, width=300, height=60,
                                fg_color="gray",
                                hover_color="green", command=download_and_tag)
download_button.pack(pady=10)

# Toggle button for download widgets
download_icon = ctk.CTkImage(light_image=Image.open("download_icon.png"), size=(30, 30))
toggle_download_btn = ctk.CTkButton(frame, image=download_icon, text="", width=40, height=40, corner_radius=30,
                                    fg_color="gray", hover_color="green", command=toggle_download_widgets)
toggle_download_btn.pack(pady=10)

album_cover_label = ctk.CTkLabel(frame, text="", fg_color='black')
album_cover_label.pack(pady=10)

# Time visualizer
time_label = ctk.CTkLabel(frame, text="00:00 / 00:00", fg_color='black', text_color='white')
time_label.pack(pady=10)

# Volume control
volume_frame = ctk.CTkFrame(frame, fg_color='black')
volume_frame.pack(pady=10)

volume_icon_label = ctk.CTkLabel(volume_frame, text="", fg_color='black')
volume_icon_label.pack(side=tk.LEFT, padx=10)

volume_slider = ctk.CTkSlider(volume_frame, from_=0, to=100, orientation=tk.HORIZONTAL, command=set_volume,
                              fg_color='gray', button_color='green')
volume_slider.set(50)  # Set initial volume to 50%
volume_slider.pack(side=tk.LEFT, padx=10)

# Ensure you have the actual images in your working directory
play_img = ctk.CTkImage(light_image=Image.open("play_icon.png"), size=(30, 30))
pause_img = ctk.CTkImage(light_image=Image.open("pause_icon.png"), size=(30, 30))
next_img = ctk.CTkImage(light_image=Image.open("next_icon.png"), size=(30, 30))
previous_img = ctk.CTkImage(light_image=Image.open("previous_icon.png"), size=(30, 30))

play_pause_frame = ctk.CTkFrame(frame, fg_color='black')
play_pause_frame.pack(pady=10)

previous_btn = ctk.CTkButton(play_pause_frame, image=previous_img, text="", width=50, height=50, corner_radius=30,
                             fg_color="gray", hover_color="green", command=play_previous_song)
previous_btn.pack(side=tk.LEFT, padx=10)

play_pause_btn = ctk.CTkButton(play_pause_frame, image=play_img, text="", width=50, height=50, corner_radius=30,
                               fg_color="gray", hover_color="green", command=toggle_play_pause)
play_pause_btn.pack(side=tk.LEFT, padx=10)

next_btn = ctk.CTkButton(play_pause_frame, image=next_img, text="", width=50, height=50, corner_radius=30,
                         fg_color="gray", hover_color="green", command=play_next_song)
next_btn.pack(side=tk.LEFT, padx=10)

# Add scan button
scan_icon = ctk.CTkImage(light_image=Image.open("scan_icon.png"), size=(30, 30))  # Ensure you have a scan icon image
scan_btn = ctk.CTkButton(frame, image=scan_icon, text="", width=40, height=40, corner_radius=30,
                         fg_color="gray", hover_color="green", command=start_yolo_detection)
scan_btn.pack(pady=10)

root.mainloop()
