from pylibdmtx.pylibdmtx import decode
import cv2

img=cv2.imread('image.png')
decoded_objects=decode(img)

for obj in decoded_objects:
    rect=obj.rect
    cv2.rectangle(img, (rect.left, rect.top), (rect.left+rect.width, rect.top+rect.height), (0, 255, 0), 2)

cv2.imwrite('result.png', img)