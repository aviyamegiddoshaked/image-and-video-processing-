from PIL import Image, ImageFilter

# Path to your image
image_path = r"/Users/aviyamegiddoshaked/Desktop/image-and-video-processing-/databrary/db1.mp4"

# Open the image
image = Image.open(image_path)

# Apply Gaussian Blur with a specific radius
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))  # Adjust radius for desired effect

# Display the blurred image
blurred_image.show()

# Optionally, save the blurred image to a new file
output_path = r"/Users/aviyamegiddoshaked/Desktop/image-and-video-processing-/runs/gaussian_blur_databraryoutputs"
blurred_image.save(output_path)
