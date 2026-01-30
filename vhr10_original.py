import os
import random
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG
# =====================================================

RAW_LABEL_DIR = "./vhr10/gt"
POS_IMAGE_DIR = "./vhr10/positive_image_set"
NEG_IMAGE_DIR = "./vhr10/negative_image_set"

YOLO_LABEL_DIR = "./vhr10/yolo_gt"
OUTPUT_DIR = "./vhr10_original"

IMAGE_EXT = ".jpg"
SEED = 10

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

NEGATIVE_SAMPLES_TRAIN = 20

# =====================================================
# UTILS
# =====================================================

def clean_bbox_line(line: str):
    """Remove parentheses and split bbox line."""
    return line.replace("(", "").replace(")", "").split(",")

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# =====================================================
# STEP 1: CONVERT GT → YOLO FORMAT
# =====================================================

def convert_gt_to_yolo():
    ensure_dirs(YOLO_LABEL_DIR)

    for label_file in os.listdir(RAW_LABEL_DIR):
        if not label_file.endswith(".txt"):
            continue

        image_file = os.path.splitext(label_file)[0] + IMAGE_EXT
        image_path = os.path.join(POS_IMAGE_DIR, image_file)

        if not os.path.exists(image_path):
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        yolo_lines = []

        with open(os.path.join(RAW_LABEL_DIR, label_file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                x1, y1, x2, y2, class_id = clean_bbox_line(line)
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                yolo_lines.append(
                    f"{int(class_id) - 1} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                )

        with open(os.path.join(YOLO_LABEL_DIR, label_file), "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

# =====================================================
# STEP 2: TRAIN / VAL / TEST SPLIT
# =====================================================

def split_dataset():
    random.seed(SEED)

    img_dirs = {
        "train": os.path.join(OUTPUT_DIR, "images/train"),
        "val": os.path.join(OUTPUT_DIR, "images/val"),
        "test": os.path.join(OUTPUT_DIR, "images/test"),
    }

    lbl_dirs = {
        "train": os.path.join(OUTPUT_DIR, "labels/train"),
        "val": os.path.join(OUTPUT_DIR, "labels/val"),
        "test": os.path.join(OUTPUT_DIR, "labels/test"),
    }

    ensure_dirs(*img_dirs.values(), *lbl_dirs.values())

    images = [f for f in os.listdir(POS_IMAGE_DIR) if f.endswith(IMAGE_EXT)]
    labels = [f.replace(IMAGE_EXT, ".txt") for f in images]

    train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
        images, labels, test_size=TEST_RATIO, random_state=SEED
    )

    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        train_imgs, train_lbls, test_size=val_size, random_state=SEED
    )

    def copy_pairs(img_list, lbl_list, img_dst, lbl_dst):
        for img, lbl in zip(img_list, lbl_list):
            shutil.copy(os.path.join(POS_IMAGE_DIR, img), img_dst)
            shutil.copy(os.path.join(YOLO_LABEL_DIR, lbl), lbl_dst)

    copy_pairs(train_imgs, train_lbls, img_dirs["train"], lbl_dirs["train"])
    copy_pairs(val_imgs, val_lbls, img_dirs["val"], lbl_dirs["val"])
    copy_pairs(test_imgs, test_lbls, img_dirs["test"], lbl_dirs["test"])

    return img_dirs["train"], lbl_dirs["train"]

# =====================================================
# STEP 3: ADD NEGATIVE SAMPLES TO TRAIN
# =====================================================

def add_negative_samples(train_img_dir, train_lbl_dir):
    negatives = [f for f in os.listdir(NEG_IMAGE_DIR) if f.endswith(IMAGE_EXT)]
    random.shuffle(negatives)

    for img in negatives[:NEGATIVE_SAMPLES_TRAIN]:
        shutil.copy(os.path.join(NEG_IMAGE_DIR, img), train_img_dir)
        open(os.path.join(train_lbl_dir, img.replace(IMAGE_EXT, ".txt")), "w").close()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("Converting GT to YOLO format...")
    convert_gt_to_yolo()

    print("Splitting dataset...")
    train_img_dir, train_lbl_dir = split_dataset()

    print("Adding negative samples to training set...")
    add_negative_samples(train_img_dir, train_lbl_dir)

    print("Dataset preparation completed.")
