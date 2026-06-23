from skimage.feature import local_binary_pattern
import cv2
import numpy as np

#image = cv2.imread('131.jpg', cv2.IMREAD_GRAYSCALE)
#edges = cv2.Canny(gray, threshold1=100, threshold2=200)
#cv2.imwrite("result.jpg", edges)
#

image = cv2.imread('131.jpg', cv2.IMREAD_GRAYSCALE)
mid=np.mean(image)

_, binary = cv2.threshold(image, mid, 255, cv2.THRESH_BINARY)

cv2.imwrite('binary_result.jpg', binary)


#lbp = local_binary_pattern(image, 8, 1, method='default')
#lbp_normalized = (lbp/lbp.max()*(255).astype(np.uint8)
#cv2.imwrite("result.jpg", lbp_normalized)
