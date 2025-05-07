from pathlib import Path
import shutil
import os

project_root = Path(__file__).resolve().parent.parent
animals_path = project_root / "data" / "animals"
yolo_path = project_root / "yolo_dataset"

if not animals_path.exists():
    raise FileNotFoundError(f"❌ 'animals' folder not found at {animals_path}")

print(f"✅ Found animals folder at: {animals_path}")

try:
    class_folders = sorted([f for f in animals_path.iterdir() if f.is_dir()])
    if not class_folders:
        raise FileNotFoundError(f"No class folders found in {animals_path}")
    
    CLASSES = {folder.name: idx for idx, folder in enumerate(class_folders)}
    
    print("\n🔢 Discovered Classes:")
    for idx, name in enumerate(CLASSES):
        print(f"{idx}: {name}")

except Exception as e:
    print(f"❌ Error discovering classes: {str(e)}")
    raise

for split in ["train", "val"]:
    (yolo_path / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (yolo_path / f"labels/{split}").mkdir(parents=True, exist_ok=True)

total_images = 0
for class_id, class_name in enumerate(CLASSES):
    class_path = animals_path / class_name
    try:
        images = list(class_path.glob("*.[jJpP]*")) + list(class_path.glob("*.[pP][nN][gG]"))
        if not images:
            print(f"⚠️ No images found in {class_name}")
            continue
        
        split_idx = int(len(images) * 0.8)
        for i, img in enumerate(images):
            dest = "train" if i < split_idx else "val"
            
            shutil.copy(img, yolo_path / f"images/{dest}/{img.name}")
            
            txt_file = img.with_suffix('.txt')
            label_path = yolo_path / f"labels/{dest}/{txt_file.name}"
            if txt_file.exists():
                shutil.copy(txt_file, label_path)
            else:
                label_path.touch()
        
        total_images += len(images)
        print(f"✅ {class_name.ljust(15)}: {len(images)} images processed")
    
    except Exception as e:
        print(f"❌ Error processing {class_name}: {str(e)}")
        continue

yaml_content = f"""path: {yolo_path.as_posix()}
train: images/train
val: images/val
test: images/val

names:
"""
for class_id, class_name in enumerate(CLASSES):
    yaml_content += f"  {class_id}: {class_name}\n"

with open(yolo_path / "dataset.yaml", "w") as f:
    f.write(yaml_content)

print(f"\n📊 Dataset Summary:")
print(f"Total classes: {len(CLASSES)}")
print(f"Total images: {total_images}")
print(f"Train images: {len(list((yolo_path/'images/train').glob('*')))}")
print(f"Val images: {len(list((yolo_path/'images/val').glob('*')))}")

print("\n🎉 Dataset preparation complete!")
print(f"YOLO dataset saved at: {yolo_path}")