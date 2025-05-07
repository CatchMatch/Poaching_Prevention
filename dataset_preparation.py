import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import re

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data" / "animals"
YOLO_DIR = BASE_DIR / "yolo_dataset"
TEST_SPLIT = 0.2

CLASSES = {
    "antelope": 0, "badger": 1, "bear": 2, "bison": 3, "boar": 4, "chimpanzee": 5, "coyote": 6, "deer": 7,
    "elephant": 8, "flamingo": 9, "fox": 10, "goose": 11, "gorilla": 12, "gun": 13, "hedgehog": 14,
    "hippopotamus": 15, "hornbill": 16, "human": 17, "human_with_gun": 18, "hyena": 19, "kangaroo": 20,
    "koala": 21, "leopard": 22, "lion": 23, "okapi": 24, "orangutan": 25, "otter": 26, "ox": 27, "panda": 28,
    "peacock": 29, "penguin": 30, "porcupine": 31, "raccoon": 32, "reindeer": 33, "rhinoceros": 34,
    "sandpiper": 35, "seal": 36, "swan": 37, "tiger": 38, "turtle": 39, "wolf": 40, "wombat": 41, "zebra": 42
}

for split in ["train", "val"]:
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def normalize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9]', '', filename).lower()

def find_annotation(img_path):
    for ext in [".xml", ".txt"]:
        ann_path = img_path.with_suffix(ext)
        if ann_path.exists():
            return ann_path

    img_stem = normalize_filename(img_path.stem)
    for ann_path in img_path.parent.glob("*"):
        if normalize_filename(ann_path.stem) == img_stem and ann_path.suffix.lower() in [".xml", ".txt"]:
            return ann_path

    possible_prefixes = [img_path.stem.split('_')[0], img_path.stem[:8]]
    for prefix in possible_prefixes:
        for ext in [".xml", ".txt"]:
            ann_path = img_path.parent / f"{prefix}{ext}"
            if ann_path.exists():
                return ann_path

    return None

def create_empty_annotation(img_path, dest_label):
    print(f"Creating empty annotation for {img_path.name}")
    with open(dest_label, 'w') as f:
        f.write("")

all_images = []
for class_name, class_id in CLASSES.items():
    class_dir = DATA_DIR / class_name
    if not class_dir.exists():
        print(f"Warning: Missing directory {class_dir}")
        continue
    for img_path in class_dir.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            all_images.append((class_id, img_path))

train_set, val_set = train_test_split(all_images, test_size=TEST_SPLIT, random_state=42)

def process_dataset(files, split):
    processed = 0
    missing = 0

    for class_id, img_path in files:
        try:
            dest_img = YOLO_DIR / "images" / split / img_path.name
            shutil.copy(img_path, dest_img)

            ann_path = find_annotation(img_path)
            dest_label = YOLO_DIR / "labels" / split / f"{img_path.stem}.txt"

            if ann_path and ann_path.suffix.lower() == ".xml":
                try:
                    tree = ET.parse(ann_path)
                    root = tree.getroot()

                    with open(dest_label, 'w') as f:
                        for obj in root.findall("object"):
                            bbox = obj.find("bndbox")
                            coords = [
                                float(bbox.find("xmin").text),
                                float(bbox.find("ymin").text),
                                float(bbox.find("xmax").text),
                                float(bbox.find("ymax").text)
                            ]
                            img_w = float(root.find("size/width").text)
                            img_h = float(root.find("size/height").text)

                            x_center = (coords[0] + coords[2]) / 2 / img_w
                            y_center = (coords[1] + coords[3]) / 2 / img_h
                            width = (coords[2] - coords[0]) / img_w
                            height = (coords[3] - coords[1]) / img_h

                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    processed += 1
                except ET.ParseError:
                    print(f"Corrupted XML: {ann_path}")
                    create_empty_annotation(img_path, dest_label)
                    missing += 1

            elif ann_path and ann_path.suffix.lower() == ".txt":
                shutil.copy(ann_path, dest_label)
                processed += 1
            else:
                create_empty_annotation(img_path, dest_label)
                missing += 1

        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue

    print(f"\n{split.upper()} SET:")
    print(f"- Processed: {processed} images with annotations")
    print(f"- Missing: {missing} annotations (created empty files)")
    print(f"- Total: {processed + missing} images processed\n")

process_dataset(train_set, "train")
process_dataset(val_set, "val")

with open(YOLO_DIR / "dataset.yaml", 'w') as f:
    f.write(f"path: {YOLO_DIR.as_posix()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write("names:\n")
    for name, id in sorted(CLASSES.items(), key=lambda x: x[1]):
        f.write(f"  {id}: {name}\n")

print("✅ Dataset preparation complete!")
print(f"📂 YOLO dataset created at: {YOLO_DIR}")