import cv2
from PIL import Image, ImageFilter
import numpy as np
import os

# Path to your original video file
video_path = "/Users/aviyamegiddoshaked/Desktop/image-and-video-processing-/databrary/db11.mp4"

# Define the output path
output_folder = "/Users/aviyamegiddoshaked/Desktop/image-and-video-processing-/runs/gaussian_blur_databraryoutputs"
output_file_name = "blurred_databarry11.mp4"
output_path = os.path.join(output_folder, output_file_name)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV frame to PIL
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply Gaussian blur
    blurred_pil_frame = pil_frame.filter(ImageFilter.GaussianBlur(radius=10))

    # Convert back to OpenCV
    blurred_frame = cv2.cvtColor(np.array(blurred_pil_frame), cv2.COLOR_RGB2BGR)

    # Write to output video
    out.write(blurred_frame)

# Release everything automatically after done
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Blurred video saved to: {output_path}")
