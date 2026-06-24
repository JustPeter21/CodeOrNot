import cv2
import zxingcpp
import time
from pathlib import Path



project_dir = Path(__file__).parent.parent
PATH = project_dir / "YOLO" / "dataset"
i=0
A=[]
while True:
    try:
        with open(PATH/"labels"/"validation"/f"{i}.txt", "r", encoding='utf-8') as file:
            data=file.read()
        i=i+1
    except FileNotFoundError:
        break
    A.append([float(x) for x in data.split(" ")][1:])
print(len(A))

start=time.time()
B=[]
for i in range(len(A)):
    image=cv2.imread(PATH/"images"/"validation"/f"{i}.jpg")
    codes=zxingcpp.read_barcodes(image)
    DM=[b for b in codes if b.format == zxingcpp.BarcodeFormat.DataMatrix]
    B.append(len(DM))
    print(i)

end=time.time()
total=end-start

print(sum(B)/len(B))
print(total)