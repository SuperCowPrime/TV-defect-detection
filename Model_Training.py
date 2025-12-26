import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil
from ultralytics import YOLO
import multiprocessing


def main():

    input_folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset\defected_tvs"
    output_folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset\split_data"


    train_ratio = 0.8

    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        files = sorted([f for f in os.listdir(class_path) if f.lower().endswith('.png')])
        total_files = len(files)

        if total_files == 0:
            continue

        split_idx = int(total_files * train_ratio)


        if split_idx == 0 and total_files > 0:
            split_idx = 1
        elif split_idx == total_files and total_files > 1:
            split_idx = total_files - 1

        train_files = files[:split_idx]
        val_files = files[split_idx:]

        print(f"Class '{class_name}': Total={total_files} -> Train={len(train_files)}, Val={len(val_files)}")


        for f in train_files:
            dest = os.path.join(output_folder, "train", class_name)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(os.path.join(class_path, f), os.path.join(dest, f))


        for f in val_files:
            dest = os.path.join(output_folder, "val", class_name)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(os.path.join(class_path, f), os.path.join(dest, f))

    model = YOLO('yolo11n-cls.pt')

    results = model.train(
        data=output_folder,
        epochs=10,
        imgsz=640,
        batch=8,
        project="tv_defect_project",
        name="run_final_windows",
        workers=0
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()