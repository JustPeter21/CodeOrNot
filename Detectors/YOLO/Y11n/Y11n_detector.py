from ultralytics import YOLO
import cv2

model=YOLO('runs/detect/dm_detector/yolo11n_dm/weights/best.pt')
PATH='11.jpg'

results=model.predict(PATH, conf=0.5)
result=results[0]
img=result.orig_img.copy()
boxes=result.boxes
height, width=img.shape[:2]

if boxes is not None:
    for box in boxes:
        x1, y1, x2, y2=box.xyxy[0].cpu().numpy().astype(int)
        conf=box.conf[0].cpu().numpy()
        label=f"dm_code {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1/1750*height, (0, 255, 0), 2)

cv2.imwrite('output_image.png', img)

