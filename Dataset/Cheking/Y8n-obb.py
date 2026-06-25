import numpy as np
import cv2

image=cv2.imread("DS_Y8n-obb/images/train/124.jpg")
h, w=image.shape[:2]
with open("DS_Y8n-obb/labels/train/124.txt","r",encoding="UTF-8") as file:
    data=file.read()
A=np.array([float(x) for x in data.split(" ")][1:])
A=A.reshape(4, 2)
A[:, 0]*=w
A[:, 1]*=h
A=A.astype(np.int32)

cv2.polylines(image, [A], isClosed=True, color=(0, 255, 0), thickness=3)
cv2.imwrite("result.jpg", image)