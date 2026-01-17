import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


label_folders = [p for p in ROOT.iterdir() if p.is_dir()]


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

rows = []
for folder in sorted(label_folders):
    label = folder.name
    for img_path in folder.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
            rows.append([str(img_path), label])

out_csv = ROOT / "labels.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label"])
    writer.writerows(rows)

print("Folders used as labels:", [p.name for p in sorted(label_folders)])
print(f"Done! {len(rows)} images written to {out_csv}")
