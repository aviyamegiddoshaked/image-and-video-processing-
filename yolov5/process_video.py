import torch
import cv2

# Path to your video and output file
input_video_path = '/Users/aviyamegiddoshaked/Downloads/RPReplay_Final1707744312.mp4'
output_video_path = '/Users/aviyamegiddoshaked/Desktop/RPReplay_Final1707744312_output.mp4'

# Load YOLOv5 model (you can change to 'yolov5m', 'yolov5l', etc., for larger models)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 inference on the frame
    results = model(frame)
    processed_frame = results.render()[0]  # Draw boxes and labels

    # Write the processed frame to the output video
    out.write(processed_frame)

# Release resources
cap.release()
out.release()
print(f"Processed video saved at: {output_video_path}")
