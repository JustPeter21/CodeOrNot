from pathlib import Path
import yaml
import datasets
import numpy as np

# Скачиваем датасет
dataset=datasets.load_dataset("shortery/dm-codes")

# Изначальное соотнешение validation и test очень плохое, поэтому переделаем это разделение                                       
mega_dataset=datasets.concatenate_datasets([dataset['validation'], dataset['test']])
size=len(mega_dataset)
train_size=int(size*0.8)
mega_dataset=mega_dataset.shuffle(seed=67)

dataset=datasets.DatasetDict({
    'train': mega_dataset.select(range(train_size)),
    'validation': mega_dataset.select(range(train_size, size))
})

# Преобразуем формат аннотаций для YOLO-OBB
def converter(example):
    width, height=example["image"].size

    X=np.array([example["tl"][0], example["tr"][0], example["br"][0], example["bl"][0]])/width
    Y=np.array([example["tl"][1], example["tr"][1], example["br"][1], example["bl"][1]])/height

    example["yolo_annotation"]=(f"0 {X[0]} {Y[0]} {X[1]} {Y[1]} {X[2]} {Y[2]} {X[3]} {Y[3]}")
    return example

dataset=dataset.map(converter)

# Сохраняем аннотации
PATH=Path("./DS_Y8n-obb")
PATH.mkdir(exist_ok=True)
(PATH/"images"/"train").mkdir(parents=True, exist_ok=True)
(PATH/"images"/"validation").mkdir(parents=True, exist_ok=True)
(PATH/"labels"/"train").mkdir(parents=True, exist_ok=True)
(PATH/"labels"/"validation").mkdir(parents=True, exist_ok=True)

for i in ["train", "validation"]:
    for id, example in enumerate(dataset[i]):
        example["image"].save(PATH/"images"/i/f"{id}.jpg")     
        with open(PATH/ "labels"/i/f"{id}.txt", "w") as f:
            f.write(example["yolo_annotation"] + "\n")

data_yaml={
    "path": str(PATH.absolute()),
    "train": "images/train",
    "val": "images/validation",
    "nc": 1
}

yaml_path=PATH/"data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)
