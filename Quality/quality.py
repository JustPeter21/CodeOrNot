from skimage.feature import local_binary_pattern
import cv2
import numpy as np

image = cv2.imread('11.jpg', cv2.IMREAD_GRAYSCALE)
height, width=image.shape[:2]

k=height*width//3000
if k%2==0:
    k=k-1


binary = cv2.adaptiveThreshold(
    image,           # исходное изображение
    255,             # максимальное значение
    cv2.ADAPTIVE_THRESH_MEAN_C,  # метод: среднее по окрестности
    cv2.THRESH_BINARY,           # тип порога
    k,              # размер окрестности (нечетное число)
    10               # константа вычитания
)

cv2.imwrite('binary_result.jpg', binary)

edges = cv2.Canny(binary, 50, 150, apertureSize=3)  # Поиск границ
cv2.imwrite('canny.jpg', edges)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=150, maxLineGap=30)

# Отрисовка найденных отрезков
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite('Hough.jpg', image)
