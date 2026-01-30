import os
import shutil
import random
from PIL import Image
from collections import defaultdict, Counter

# =====================================================
# CONFIG
# =====================================================

VERSIONS = [
    {"name": "VHR10_10", "train_count": 10, "val_count": 20, "seed": 7521},
    {"name": "VHR10_15", "train_count": 15, "val_count": 30, "seed": 61},
    {"name": "VHR10_20", "train_count": 20, "val_count": 40, "seed": 10},
    {"name": "VHR10_25", "train_count": 25, "val_count": 40, "seed": 568},
    {"name": "VHR10_30", "train_count": 30, "val_count": 40, "seed": 8559},
]

RAW_GT_DIR = "./vhr10/gt"
POS_IMAGE_DIR = "./vhr10/positive_image_set"
NEG_IMAGE_DIR = "./vhr10/negative_image_set"
YOLO_GT_DIR = "./vhr10/yolo_gt"

IMAGE_EXT = ".jpg"
NEGATIVE_TRAIN_SAMPLES = 10

# =====================================================
# UTILS
# =====================================================

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def parse_bbox(line):
    return line.replace("(", "").replace(")", "").split(",")

# =====================================================
# STEP 1: CONVERT GT → YOLO (RUN ONCE)
# =====================================================

def convert_gt_to_yolo():
    ensure_dirs(YOLO_GT_DIR)

    for label_file in os.listdir(RAW_GT_DIR):
        if not label_file.endswith(".txt"):
            continue

        img_file = label_file.replace(".txt", IMAGE_EXT)
        img_path = os.path.join(POS_IMAGE_DIR, img_file)

        if not os.path.exists(img_path):
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        yolo_lines = []

        with open(os.path.join(RAW_GT_DIR, label_file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                x1, y1, x2, y2, cls = parse_bbox(line)
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                yolo_lines.append(
                    f"{int(cls) - 1} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )

        with open(os.path.join(YOLO_GT_DIR, label_file), "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

# =====================================================
# STEP 2: CLASS-BALANCED SPLIT (PER VERSION)
# =====================================================

def run_split(version):
    name = version["name"]
    train_limit = version["train_count"]
    val_limit = version["val_count"]
    random.seed(version["seed"])

    paths = {
        "train_img": f"{name}/images/train",
        "val_img": f"{name}/images/val",
        "test_img": f"{name}/images/test",
        "train_lbl": f"{name}/labels/train",
        "val_lbl": f"{name}/labels/val",
        "test_lbl": f"{name}/labels/test",
    }

    ensure_dirs(*paths.values())

    train_cnt = defaultdict(int)
    val_cnt = defaultdict(int)

    train_files = []
    val_files = []
    test_files = []

    label_files = os.listdir(YOLO_GT_DIR)
    random.shuffle(label_files)

    for lbl in label_files:
        if not lbl.endswith(".txt"):
            continue

        img = lbl.replace(".txt", IMAGE_EXT)

        with open(os.path.join(YOLO_GT_DIR, lbl)) as f:
            classes = [l.split()[0] for l in f.readlines()]

        class_counts = Counter(classes)

        can_train = all(train_cnt[c] + n <= train_limit for c, n in class_counts.items())
        can_val = all(val_cnt[c] + n <= val_limit for c, n in class_counts.items())

        if can_train:
            train_files.append((img, lbl))
            for c, n in class_counts.items():
                train_cnt[c] += n

        elif can_val:
            val_files.append((img, lbl))
            for c, n in class_counts.items():
                val_cnt[c] += n

        else:
            test_files.append((img, lbl))

    def copy(files, img_dst, lbl_dst):
        for img, lbl in files:
            shutil.copy(os.path.join(POS_IMAGE_DIR, img), img_dst)
            shutil.copy(os.path.join(YOLO_GT_DIR, lbl), lbl_dst)

    copy(train_files, paths["train_img"], paths["train_lbl"])
    copy(val_files, paths["val_img"], paths["val_lbl"])
    copy(test_files, paths["test_img"], paths["test_lbl"])

    # Add negative samples to TRAIN
    negs = [f for f in os.listdir(NEG_IMAGE_DIR) if f.endswith(IMAGE_EXT)]
    random.shuffle(negs)

    for img in negs[:NEGATIVE_TRAIN_SAMPLES]:
        shutil.copy(os.path.join(NEG_IMAGE_DIR, img), paths["train_img"])
        open(os.path.join(paths["train_lbl"], img.replace(IMAGE_EXT, ".txt")), "w").close()

# =====================================================
# STEP 3: FORCE COMMON TEST SET
# =====================================================

def unify_test_sets():
    label_dirs = [f"{v['name']}/labels/test" for v in VERSIONS]
    image_dirs = [f"{v['name']}/images/test" for v in VERSIONS]

    common_labels = set(os.listdir(label_dirs[0]))
    for d in label_dirs[1:]:
        common_labels &= set(os.listdir(d))

    common_images = {f.replace(".txt", IMAGE_EXT) for f in common_labels}

    def cleanup(dirs, keep):
        for d in dirs:
            for f in os.listdir(d):
                if f not in keep:
                    os.remove(os.path.join(d, f))

    cleanup(label_dirs, common_labels)
    cleanup(image_dirs, common_images)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("Converting GT → YOLO...")
    convert_gt_to_yolo()

    for v in VERSIONS:
        print(f"Running split for {v['name']}")
        run_split(v)

    print("Forcing common test set...")
    unify_test_sets()

    print("Done ✔")
