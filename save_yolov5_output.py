import torch
import pandas as pd
import cv2
from PIL import Image
import numpy as np
import os

# --- Configuration ---
video_path = "/Users/aviyamegiddoshaked/Desktop/image-and-video-processing-/runs/databrary-asspart4/gaussian_blur_databraryoutputs/blurred_databarry11.mp4"

# Derive file name without extension
video_name = os.path.splitext(os.path.basename(video_path))[0]

# ✅ Correct output paths
output_csv_path = f"/Users/aviyamegiddoshaked/Desktop/image-and-video-processing-/runs/databrary-asspart4/csvfiles_databrary/{video_name}.csv"
output_video_path = f"/Users/aviyamegiddoshaked/Desktop/image-and-video-processing-/runs/databrary-asspart4/yoloblurred_databrary/{video_name}_output.mp4"

# Create output directories if they don't exist
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# Load YOLOv5 model (small model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Error: Cannot open video: {video_path}")
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

    # Convert frame from BGR to RGB (YOLO expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(rgb_frame)

    # Get detection dataframe
    detections = results.pandas().xyxy[0]
    time_in_video = frame_number / fps
    detections['time'] = time_in_video
    all_detections.append(detections)

    # Draw detections on the frame
    for _, row in detections.iterrows():
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
