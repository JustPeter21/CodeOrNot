import cv2
import numpy as np
import math

image=cv2.imread("binary_result.py")

height, width=image.shape[:2]

k=(height+width)//2







def rotate_image(image, angle):
    """
    Поворачивает изображение относительно центра без обрезания углов.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def find_rotated_template(image_path, template_path, angle_step=30, threshold=0.7):
    """
    Ищет шаблон на изображении независимо от его поворота.

    Аргументы:
        image_path: путь к большому изображению (сцена).
        template_path: путь к шаблону (объект, который ищем).
        angle_step: шаг угла поворота (градусы). Чем меньше, тем точнее, но медленнее.
        threshold: порог уверенности (0..1). Рекомендуется 0.7-0.8.

    Возвращает:
        Координаты (x, y) центра найденного объекта, угол поворота и максимальную уверенность.
    """
    # 1. Загружаем изображения в оттенках серого (для скорости)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or template is None:
        raise FileNotFoundError("Проверьте пути к файлам.")

    h, w = template.shape
    best_match = None
    best_val = -1
    best_angle = 0
    best_location = (0, 0)

    # 2. Перебираем углы от 0 до 360 с заданным шагом
    for angle in range(0, 360, angle_step):
        # Поворачиваем шаблон
        rotated_template = rotate_image(template, angle)
        
        # Метод сопоставления - коэффициент корреляции (лучше всего для поворотов)
        # cv2.TM_CCOEFF_NORMED дает результат от -1 до 1. 1 = идеальное совпадение.
        result = cv2.matchTemplate(img, rotated_template, cv2.TM_CCOEFF_NORMED)
        
        # Находим максимум на карте совпадений
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Сохраняем лучший результат
        if max_val > best_val:
            best_val = max_val
            best_angle = angle
            best_location = max_loc

    # 3. Проверяем, превышает ли найденное значение порог
    if best_val < threshold:
        print(f"Объект не найден. Лучшее совпадение: {best_val:.2f}")
        return None

    # Координаты центра найденного объекта
    x_center = best_location[0] + w // 2
    y_center = best_location[1] + h // 2

    print(f"Объект найден! Позиция: ({x_center}, {y_center}), Угол: {best_angle}°, Уверенность: {best_val:.2f}")

    return {
        "x": x_center,
        "y": y_center,
        "angle": best_angle,
        "confidence": best_val,
        "top_left": best_location
    }