# assignment_VR

## Counting coins
The counting of coins is done in 6 simple steps

### 1. Loading and preprosessing
the image chosen was extremly large, which needed to be reduced in size.
I use resize from open cv to reduce the size of image and blur it using gaussian blur
``` python
half = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)  # Resize image to 50% of original size
gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
```

### 2. Edge detection
simle canny edge detector with 50 and 150 thrasholds which work pretty well
``` python
edges = cv2.Canny(blurred, 50, 150) # Canny
```

### 3. Edge improvement
To connect the broken edges i use a simple dilation filter using a 3x3 kernel. Dilation expands the edges, which makes contours easy to detect
``` python
kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel
edges = cv2.dilate(edges, kernel, iterations=1)  # dilation
```

### 4. contours
again simple openCV function of findContours
After finding contors, I filter out objects with area < 100 to remove noies, and then check circulatity ratio to determine if the detected contour actually represents a coin (is it circular enough)
``` python
valid_contours = []
for contour in contours:
    if cv2.contourArea(contour) < 100:  # small contours
        continue
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)  # compute circle area
    contour_area = cv2.contourArea(contour)  # compute actual contour area
    ratio = contour_area / circle_area if circle_area != 0 else 0  # compute ratio
    if ratio > 0.7:  # circular thrashold to check if contour is vaolid coin
        valid_contours.append(contour)
```

### 5. Disply
Simple openCV function mapped on the image
``` python
cv2.drawContours(half, valid_contours, -1, (0, 255, 0), 2) # green
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(half, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct visualization
plt.title('Detected Coins')
plt.axis('off')
plt.show()
```
I also save all coins using os in a segmented_coins directory 

### 6. Count and display Number of coins
due to storing contours, this is an easy step of just seeing size of the contour
``` python
coin_count = len(valid_contours)
```


## Panorama

### 1. Determine keypoints and descriptors using SIFT
we use sift to determine the keypoints and descriptors of images. Our functions can only take two images at a time so for 3 images we need to do it iteratively
``` python
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
```

### 2. Match features
to match features we could have done simpel similarity scores but that was taking too much time and memory. So I resorted to using FLANN in CV. this calcultes and finds matches in a resonable times.
``` python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)
```

### 3. Filter good matches
simple function with a 0.7 threshold, to remove other matches. 

``` python
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```
### 4. COmpute Homography
if there are enough good matches, we compute the homography matrix to transform and align the two images. WE also use RANSAC to remvoe the outliers
``` python
if len(good_matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

### 5. Create Panorama
after computing homography matrix we use that to warp the prespective of the image and create panorama
``` python
    height, width, channels = img2.shape
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
```


There is further error handling if not enough matches are found.
Moreover since the function only works on 2 images we have to iterate on all three images to create the panorama. i.e
```python
panorama = create_panorama(img3,img2)
final_panorama = create_panorama(panorama, img1)
```

## Requirements
Both scripts require 
- cv2
- numpy
- matplotlib
In addition to that, to save the detected segmented coins we require
- os

