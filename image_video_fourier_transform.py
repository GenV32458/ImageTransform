# Image and Video Processing Using Fourier Transform
# Google Colab version

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# 1. Upload an image
uploaded_image = files.upload()
photo_name = list(uploaded_image.keys())[0]
photo = cv2.imread(photo_name)

if photo is None:
    raise ValueError("The uploaded image could not be read.")

print("Your photograph has been uploaded successfully!")
print("Photo dimensions:", photo.shape)

# 2. Display the original image
photo_rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(9, 6))
plt.imshow(photo_rgb)
plt.title("Original Photograph")
plt.axis("off")
plt.show()

# 3. Convert the image into grayscale
photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(9, 6))
plt.imshow(photo_gray, cmap="gray")
plt.title("Black and White Photograph")
plt.axis("off")
plt.show()

# 4. Apply Fourier Transform to the image
image_frequency = np.fft.fft2(photo_gray)
centered_frequency = np.fft.fftshift(image_frequency)
image_spectrum = 20 * np.log(np.abs(centered_frequency) + 1)

# 5. Display the image magnitude spectrum
plt.figure(figsize=(9, 6))
plt.imshow(image_spectrum, cmap="gray")
plt.title("Image Frequency Spectrum")
plt.axis("off")
plt.show()

# 6. Upload a video
uploaded_movie = files.upload()
movie_name = list(uploaded_movie.keys())[0]

print("Your video has been uploaded successfully!")
print("Video file:", movie_name)

# 7. Open the uploaded video
video_capture = cv2.VideoCapture(movie_name)

if video_capture.isOpened():
    print("Video is ready for processing!")
else:
    raise ValueError("Unable to open the uploaded video.")

# 8. Extract one frame from the video
success, video_image = video_capture.read()

if not success:
    video_capture.release()
    raise ValueError("Failed to extract a frame from the video.")

print("A video frame has been extracted successfully!")

# 9. Display the extracted video frame
video_rgb = cv2.cvtColor(video_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(9, 6))
plt.imshow(video_rgb)
plt.title("Selected Video Frame")
plt.axis("off")
plt.show()

# 10. Convert the video frame into grayscale
video_gray = cv2.cvtColor(video_image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(9, 6))
plt.imshow(video_gray, cmap="gray")
plt.title("Grayscale Video Frame")
plt.axis("off")
plt.show()

# 11. Apply Fourier Transform to the video frame
frequency_frame = np.fft.fft2(video_gray)
centered_frame = np.fft.fftshift(frequency_frame)
frame_spectrum = 20 * np.log(np.abs(centered_frame) + 1)

# 12. Display the video frame magnitude spectrum
plt.figure(figsize=(9, 6))
plt.imshow(frame_spectrum, cmap="gray")
plt.title("Video Frame Frequency Spectrum")
plt.axis("off")
plt.show()

# 13. Release the video resource
video_capture.release()
print("Video processing completed successfully!")
