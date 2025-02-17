import cv2
import numpy as np
import matplotlib.pyplot as plt


#function to stitch two images into a panorama
def create_panorama(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Use FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # match
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Compute homography if enough matches are found
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    #wrap
        height, width, channels = img2.shape
        result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        result[0:img2.shape[0], 0:img2.shape[1]] = img2

        return result
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 10))
        return None


img1 = cv2.imread('images/IMG_3134.jpeg')  # Replace with your image paths
img2 = cv2.imread('images/IMG_3135.jpeg')
img3 = cv2.imread('images/IMG_3136.jpeg')


#reduce sizes
img1 = cv2.resize(img1, (0, 0), fx = 0.5, fy = 0.5)
img2 = cv2.resize(img2, (0, 0), fx = 0.5, fy = 0.5)
img3 = cv2.resize(img3, (0, 0), fx = 0.5, fy = 0.5)




# Stitch the first two images
panorama = create_panorama(img3, img2)

#panaroma of this aand third image
if panorama is not None:
    final_panorama = create_panorama(panorama, img1)

    #display the final panorama
    if final_panorama is not None:
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
        plt.title("Final Panorama")
        plt.axis('off')
        plt.show()
