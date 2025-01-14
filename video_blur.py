import cv2
from PIL import Image, ImageFilter
import numpy as np
import os

# Path to your original video file
video_path = "/Users/aviyamegiddoshaked/Desktop/IMG_1799.MOV"

# Define the output path in the current script folder
output_folder = "/Users/aviyamegiddoshaked/Desktop/image-and-video-processing--1"
output_file_name = "blurred_IMG_1799.MOV"
output_path = os.path.join(output_folder, output_file_name)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was loaded successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
else:
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object for the blurred video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break  # Exit the loop if no more frames are available

        # Convert the frame (OpenCV format) to a PIL Image
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Apply GaussianBlur using Pillow
        blurred_pil_frame = pil_frame.filter(ImageFilter.GaussianBlur(radius=10))  # Adjust radius as needed

        # Convert the blurred PIL Image back to OpenCV format
        blurred_frame = cv2.cvtColor(np.array(blurred_pil_frame), cv2.COLOR_RGB2BGR)

        # Display the blurred frame (optional)
        cv2.imshow('Blurred Video', blurred_frame)

        # Write the blurred frame to the output video file
        out.write(blurred_frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Blurred video saved to: {output_path}")
