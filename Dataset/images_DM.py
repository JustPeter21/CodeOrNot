import numpy as np
from pathlib import Path
import cv2

for directory in ["train", "validation"]:
    (Path("dataset_DM") / "images" / directory).mkdir(parents=True, exist_ok=True)

for directory in ["train", "validation"]:
    i=0
    while True:
        try:
            img_path = Path("dataset") / "images" / directory / f"{i}.jpg"
            txt_path = Path("dataset") / "labels" / directory / f"{i}.txt"

            with open(txt_path, "r", encoding="UTF-8") as file:
                data=file.read()
            A=[float(x) for x in data.split(" ")[1:]]

            image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            height, width=image.shape[:2]   

            x_max=int((A[0]+A[2]/2)*width*1.01)
            x_min=int((A[0]-A[2]/2)*width*0.99)
            y_max=int((A[1]+A[3]/2)*height*1.01)
            y_min=int((A[1]-A[3]/2)*height*0.99)
            image=image[y_min:y_max, x_min:x_max]

            save_path = Path("dataset_DM") / "images" / directory / f"{i}.jpg"
            cv2.imwrite(save_path, image)
            i=i+1
        except FileNotFoundError:
            break