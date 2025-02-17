import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/coins.jpg')
original = image.copy()
half = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
# Detect edges using Canny
edges = cv2.Canny(blurred, 50, 150)

# Close gaps between edges using dilation
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_contours = []
for contour in contours:
    if cv2.contourArea(contour) < 100:  # Ignore small contours
        continue
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)
    contour_area = cv2.contourArea(contour)
    ratio = contour_area / circle_area if circle_area != 0 else 0
    if ratio > 0.7:  # Threshold for circularity
        valid_contours.append(contour)

# Draw contours on the original image
cv2.drawContours(half, valid_contours, -1, (0, 255, 0), 2)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(half, cv2.COLOR_BGR2RGB))
plt.title('Detected Coins')
plt.axis('off')
plt.show()

# Create directory to save segmented coins (run once)
import os
if not os.path.exists('segmented_coins'):
    os.makedirs('segmented_coins')

# Segment and save each coin
for idx, contour in enumerate(valid_contours):
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    segmented = cv2.bitwise_and(half, half, mask=mask)

    # Crop to the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    cropped = segmented[y:y+h, x:x+w]

    cv2.imwrite(f'segmented_coins/coin_{idx}.png', cropped)


# Count coins
coin_count = len(valid_contours)

# Display count on the image
result_image = half.copy()
cv2.putText(result_image, f'Total Coins: {coin_count}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()