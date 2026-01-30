# YOLO-Adapter Datasets

This repository contains dataset preparation and splitting scripts used for experiments with **YOLO-based object detection models**, particularly in the context of **data-limited training** and **parameter-efficient fine-tuning (PEFT)** such as **YOLO-Adapter**.

The scripts focus on:

* Converting datasets into **YOLO format**
* Creating **class-balanced train/val splits**
* Generating **multiple dataset versions with different training sizes**
* Enforcing a **shared (frozen) test set** across all experimental settings

---

## Datasets

### 1. Aquarium Dataset

* **Source:** Roboflow Public Datasets
* **Link:** [Roboflow Dataset link](https://public.roboflow.com/object-detection/aquarium)
* **Task:** Underwater object detection
* **Annotations:** Bounding boxes

The Aquarium dataset is used to evaluate model behavior under **controlled reductions in training data**, while keeping validation and test sets consistent.

---

### 2. VHR10 Dataset

* **Source:** Very High Resolution Remote Sensing Dataset
* **Link:** [VHR10 Google Drive link](https://drive.google.com/file/d/1--foZ3dV5OCsqXQXT84UeKtrAqc5CkAE/view)
* **Task:** Remote sensing object detection
* **Characteristics:**

  * Positive and negative image sets
  * Non-YOLO ground-truth format (converted by this repo)

The VHR10 dataset is used to test robustness in **aerial imagery** and **low-data regimes**, including the controlled injection of negative samples.

---

## Scripts Overview

---

## `aquarium_splits.py`

### Purpose

Creates **multiple versions of the Aquarium dataset** with different training sizes while ensuring:

* Class-balanced train and validation sets
* A **shared (identical) test set** across all versions

### Key Features

* Flattens the original Roboflow dataset structure
* Generates multiple dataset versions (e.g., 10, 15, 20, 25, 30 instances per class)
* Uses deterministic random seeds
* Enforces a **frozen test set** for fair comparison
* Outputs datasets in **YOLO-ready format**

### Typical Output Structure

```text
Aquarium_20/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

## `vhr10_original.py`

### Purpose

Prepares a **baseline VHR10 dataset split** using a standard random strategy.

### Key Features

* Converts VHR10 ground truth to **YOLO format**
* Performs a **random 70/20/10 split** (train/val/test)
* Adds a fixed number of **negative samples** to the training set
* Produces a standard YOLO directory layout

### Output

```text
vhr10_original/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

## `vhr10_split.py`

### Purpose

Creates **multiple class-balanced versions of the VHR10 dataset**, similar in spirit to `aquarium_splits.py`.

### Key Features

* Converts raw VHR10 annotations to YOLO format (once)
* Generates **multiple dataset versions** with different per-class training limits
* Maintains class balance in train and validation sets
* Injects **negative samples into training**
* Enforces a **shared test set across all versions**

### Example Versions

```text
VHR10_10
VHR10_15
VHR10_20
VHR10_25
VHR10_30
```

Each version differs **only in training data size**, enabling controlled experiments.

---

## Prepared Dataset Availability

All prepared dataset versions (Aquarium and VHR10), including:

* images
* labels
* train / val / test splits
* corresponding `.yaml` files for YOLO training

are available at the following **[Google Drive link](https://drive.google.com/drive/folders/1M4xL1qXdYsf2zb23SqTOgjLzlUW957Fk?usp=sharing)**

---

## Reproducibility Notes

* All splits are **seed-controlled**
* Test sets are **identical across versions**
* No image appears in more than one split
* Negative samples contain empty label files (YOLO-compatible)

These guarantees ensure **fair comparison** across all experimental settings.

---

## Intended Use

This repository is intended for:

* Research in **object detection under limited data**
* Evaluation of **parameter-efficient fine-tuning methods**
* Controlled dataset scaling experiments
* YOLO-based training pipelines
