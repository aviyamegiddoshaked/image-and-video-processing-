import torch
import pandas as pd
import cv2  # OpenCV for video processing

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5s model

# Path to your video
video_path = '/Users/aviyamegiddoshaked/Desktop/IMG_1799.MOV'
#video_path = '/path/to/your/video.mp4'
output_csv_path = 'detections_with_time.csv'


# Open the video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
frame_number = 0  # Track the current frame number

all_detections = []  # List to store detections for all frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate time in seconds for the current frame
    time_in_video = frame_number / fps

    # Run inference on the current frame
    results = model(frame)

    # Convert detections to a DataFrame
    detections = results.pandas().xyxy[0]

    # Add the time column
    detections['time'] = time_in_video

    # Append the frame's detections to the list
    all_detections.append(detections)

    # Increment the frame counter
    frame_number += 1

# Combine all detections into a single DataFrame
final_detections = pd.concat(all_detections, ignore_index=True)

# Save to CSV
final_detections.to_csv(output_csv_path, index=False)

print(f"Detections with timestamps saved to {output_csv_path}")

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
