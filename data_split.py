import random
import shutil
from pathlib import Path

# 你的原始数据目录（相对当前 ML-CNN 目录）
src_root = Path("dataset")
# 输出目录（会自动创建）
dst_root = Path("dataset_split")

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

random.seed(42)

# 识别的图片后缀（如有 .webp 可以加进去）
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

classes = [d.name for d in src_root.iterdir() if d.is_dir()]
print("发现类别：", classes)

# 创建输出目录结构
for split in ["train", "val", "test"]:
    for cls in classes:
        (dst_root / split / cls).mkdir(parents=True, exist_ok=True)

def list_images(folder: Path):
    return [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in img_exts]

def copy_files(file_list, out_dir: Path):
    for f in file_list:
        shutil.copy2(f, out_dir / f.name)

for cls in classes:
    cls_dir = src_root / cls
    images = list_images(cls_dir)

    if len(images) == 0:
        print(f"[警告] 类别 {cls} 没找到图片。当前支持后缀: {img_exts}")
        continue

    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val  # 剩下的全给 test，避免四舍五入误差

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    print(f"\n类别: {cls}")
    print(f"总数: {n} -> train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

    copy_files(train_files, dst_root / "train" / cls)
    copy_files(val_files, dst_root / "val" / cls)
    copy_files(test_files, dst_root / "test" / cls)

print("\n✅ 完成！输出目录：", dst_root.resolve())
