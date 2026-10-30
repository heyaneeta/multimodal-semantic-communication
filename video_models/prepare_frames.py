import os
import cv2
from tqdm import tqdm

VIDEO_DIR = "datasets/action_youtube_naudio"
OUTPUT_DIR = "datasets/ucf11_frames"

FRAMES_PER_VIDEO = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_extensions = (".avi", ".mp4", ".mov", ".mpeg", ".mpg")

for class_name in os.listdir(VIDEO_DIR):
    class_path = os.path.join(VIDEO_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    all_video_files = []

    for root, dirs, files in os.walk(class_path):
        if "Annotation" in root:
            continue

        for file in files:
            if file.lower().endswith(video_extensions):
                all_video_files.append(os.path.join(root, file))

    print(f"{class_name}: found {len(all_video_files)} videos")

    for video_path in tqdm(all_video_files, desc=f"Processing {class_name}"):
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            continue

        step = max(total_frames // FRAMES_PER_VIDEO, 1)

        frame_index = 0
        saved_count = 0

        video_base_name = os.path.splitext(os.path.basename(video_path))[0]

        while cap.isOpened() and saved_count < FRAMES_PER_VIDEO:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % step == 0:
                frame_name = f"{video_base_name}_frame_{saved_count}.jpg"
                frame_path = os.path.join(output_class_path, frame_name)

                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_index += 1

        cap.release()

print("Frame extraction completed.")
print("Frames saved in:", OUTPUT_DIR)