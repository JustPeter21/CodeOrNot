from pathlib import Path
import yaml
import datasets

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

# Преобразуем формат аннотаций для YOLO
def converter(example):
    width, height=example["image"].size

    xs=[example["tl"][0], example["tr"][0], example["br"][0], example["bl"][0]]
    ys=[example["tl"][1], example["tr"][1], example["br"][1], example["bl"][1]]

    x_min, x_max=min(xs), max(xs)
    y_min, y_max=min(ys), max(ys)

    center_x=(x_min+x_max)/2/width
    center_y=(y_min+y_max)/2/height
    bbox_width=(x_max-x_min)/width
    bbox_height=(y_max-y_min)/height

    example["yolo_annotation"]=(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
    return example
dataset=dataset.map(converter)


# Сохраняем аннотации
PATH=Path("./dataset")
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
