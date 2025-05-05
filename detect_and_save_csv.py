import torch
import pandas as pd
import cv2
from PIL import Image
import numpy as np
import os

import sys
if len(sys.argv) != 2:
    print("❌ Usage: python detect_and_save_csv.py <video_filename>")
    exit()

video_filename = sys.argv[1]
video_path = f"databrary/{video_filename}"
video_name = os.path.splitext(video_filename)[0]
output_csv_path = f"runs/databrary_outputs/{video_name}.csv"
output_video_path = f"runs/databrary_outputs/annotated_{video_name}.mp4"



# Create output folder if it doesn't exist
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Load YOLOv5 model (small model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Cannot open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up video writer for output
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

frame_number = 0
all_detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV) to RGB (YOLO expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(rgb_frame)

    # Get detection dataframe
    detections = results.pandas().xyxy[0]
    time_in_video = frame_number / fps
    detections['time'] = time_in_video
    all_detections.append(detections)

    # Draw detections on the frame
    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out.write(frame)
    frame_number += 1

# Save all detections to CSV
final_detections = pd.concat(all_detections, ignore_index=True)
final_detections.to_csv(output_csv_path, index=False)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ YOLO detections saved to {output_csv_path}")
print(f"✅ Annotated video saved to {output_video_path}")
