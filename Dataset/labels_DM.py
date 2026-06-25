import numpy as np
from pathlib import Path
import cv2

for directory in ["train", "validation"]:
    (Path("dataset_DM") / "labels" / directory).mkdir(parents=True, exist_ok=True)

for directory in ["train", "validation"]:
    i=0
    while True:
        try:
            img_path = Path("dataset_DM") / "images" / directory / f"{i}.jpg"
            txt_path = Path("dataset_DM") / "labels" / directory / f"{i}.txt"

            image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            height, width=image.shape[:2]


            save_path = Path("dataset_DM") / "images" / directory / f"{i}.jpg"
            cv2.imwrite(save_path, image)
            i=i+1
        except FileNotFoundError:
            break