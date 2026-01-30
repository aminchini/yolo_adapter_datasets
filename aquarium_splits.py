import os
import shutil
import random
from collections import defaultdict, Counter

# =====================================================
# CONFIG
# =====================================================

VERSIONS = [
    {"name": "Aquarium_10", "train_count": 10, "val_count": 20, "seed": 2},
    {"name": "Aquarium_15", "train_count": 15, "val_count": 30, "seed": 1},
    {"name": "Aquarium_20", "train_count": 20, "val_count": 40, "seed": 2586},
    {"name": "Aquarium_25", "train_count": 25, "val_count": 35, "seed": 1977},
    {"name": "Aquarium_30", "train_count": 30, "val_count": 30, "seed": 531},
]

RAW_DATASET_DIR = "./aquarium_dataset"
FLAT_DATASET_DIR = "./aquarium_dataset_organized"

IMAGE_DIR = os.path.join(FLAT_DATASET_DIR, "images")
LABEL_DIR = os.path.join(FLAT_DATASET_DIR, "labels")

IMAGE_EXT = ".jpg"   # change if needed

# =====================================================
# STEP 1: FLATTEN DATASET (RUN ONCE)
# =====================================================

def organize_dataset():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    for split in ["train", "val", "valid", "test"]:
        img_dir = os.path.join(RAW_DATASET_DIR, split, "images")
        lbl_dir = os.path.join(RAW_DATASET_DIR, split, "labels")

        if os.path.isdir(img_dir):
            for f in os.listdir(img_dir):
                if f.endswith((".jpg", ".jpeg", ".png")):
                    shutil.move(
                        os.path.join(img_dir, f),
                        os.path.join(IMAGE_DIR, f)
                    )

        if os.path.isdir(lbl_dir):
            for f in os.listdir(lbl_dir):
                if f.endswith(".txt"):
                    shutil.move(
                        os.path.join(lbl_dir, f),
                        os.path.join(LABEL_DIR, f)
                    )

# =====================================================
# STEP 2: SPLIT LOGIC
# =====================================================

def run_split(version):
    name = version["name"]
    train_count = version["train_count"]
    val_count = version["val_count"]
    seed = version["seed"]

    random.seed(seed)

    paths = {
        "train_img": f"{name}/images/train",
        "val_img": f"{name}/images/val",
        "test_img": f"{name}/images/test",
        "train_lbl": f"{name}/labels/train",
        "val_lbl": f"{name}/labels/val",
        "test_lbl": f"{name}/labels/test",
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    train_cls_cnt = defaultdict(int)
    val_cls_cnt = defaultdict(int)

    train_files = defaultdict(list)
    val_files = defaultdict(list)
    test_files = defaultdict(list)

    label_files = os.listdir(LABEL_DIR)
    random.shuffle(label_files)

    for lbl in label_files:
        if not lbl.endswith(".txt"):
            continue

        img = lbl.replace(".txt", IMAGE_EXT)

        with open(os.path.join(LABEL_DIR, lbl)) as f:
            class_ids = [line.split()[0] for line in f.readlines()]

        class_counts = Counter(class_ids)

        can_train = all(
            train_cls_cnt[c] + cnt <= train_count
            for c, cnt in class_counts.items()
        )

        can_val = all(
            val_cls_cnt[c] + cnt <= val_count
            for c, cnt in class_counts.items()
        )

        if can_train:
            for c in class_ids:
                train_files[c].append((img, lbl))
                train_cls_cnt[c] += 1

        elif can_val:
            for c in class_ids:
                val_files[c].append((img, lbl))
                val_cls_cnt[c] += 1

        else:
            for c in set(class_ids):
                test_files[c].append((img, lbl))

    def move(files, img_dst, lbl_dst):
        for img, lbl in files:
            shutil.copy(os.path.join(IMAGE_DIR, img), os.path.join(img_dst, img))
            shutil.copy(os.path.join(LABEL_DIR, lbl), os.path.join(lbl_dst, lbl))

    for files in train_files.values():
        move(files, paths["train_img"], paths["train_lbl"])

    for files in val_files.values():
        move(files, paths["val_img"], paths["val_lbl"])

    for files in test_files.values():
        move(files, paths["test_img"], paths["test_lbl"])

# =====================================================
# STEP 3: FORCE COMMON TEST SET
# =====================================================

def find_common_files(directories):
    common = set(os.listdir(directories[0]))
    for d in directories[1:]:
        common.intersection_update(os.listdir(d))
    return common

def remove_uncommon_files(directories, common_files):
    for d in directories:
        for f in os.listdir(d):
            if f not in common_files:
                path = os.path.join(d, f)
                if os.path.isfile(path):
                    os.remove(path)

def unify_test_sets():
    label_test_dirs = [f"{v['name']}/labels/test" for v in VERSIONS]
    image_test_dirs = [f"{v['name']}/images/test" for v in VERSIONS]

    common_labels = find_common_files(label_test_dirs)
    common_images = {f.replace(".txt", IMAGE_EXT) for f in common_labels}

    remove_uncommon_files(label_test_dirs, common_labels)
    remove_uncommon_files(image_test_dirs, common_images)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("Flattening dataset...")
    organize_dataset()

    for v in VERSIONS:
        print(f"Running split for {v['name']}")
        run_split(v)

    print("Unifying test sets across all versions...")
    unify_test_sets()

    print("Done.")
