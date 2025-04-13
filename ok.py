from ultralytics import YOLO
import cv2

# Load your custom model
model = YOLO("Chicken_Cory.pt")

# Run inference on an image
results = model("e4e47c2878b72d4ace48bd4f89c06ad1.webp")

# Plot the detection results on the image
plotted_img = results[0].plot()

# Save the plotted image to disk
cv2.imwrite("test_1.png", plotted_img)
