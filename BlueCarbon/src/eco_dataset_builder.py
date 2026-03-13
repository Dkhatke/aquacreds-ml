# src/eco_dataset_builder.py
"""
eco_dataset_builder.py
----------------------
Utility script for building image datasets for ecosystem classification.

Outputs:
- metadata_eco.csv
- label_encoder_eco.pkl
- classes.json
- resized_images.npy
- encoded_labels.npy
- eco_dataset.tfrecord  (optional)

Also generates visual EDA plots for dataset inspection.
"""

import os
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import tensorflow as tf


# ================================================================
# CONFIG
# ================================================================
DATA_DIR = "Dataset2"          # Folder containing class subfolders
TARGET_SIZE = (224, 224)

METADATA_CSV = "metadata_eco.csv"
LABEL_ENCODER_PKL = "label_encoder_eco.pkl"
CLASSES_JSON = "classes.json"

RESIZED_IMAGES_NPY = "resized_images.npy"
ENCODED_LABELS_NPY = "encoded_labels.npy"

TFRECORD_PATH = "eco_dataset.tfrecord"


# ================================================================
# DISCOVER CLASSES
# ================================================================
def discover_classes():
    class_folders = sorted([f.name for f in os.scandir(DATA_DIR) if f.is_dir()])
    if not class_folders:
        raise RuntimeError(f"No class folders found in {DATA_DIR}. "
                           f"Expected: mangrove, seagrass, saltmarsh, etc.")

    print("✓ Classes discovered:", class_folders)
    return class_folders


# ================================================================
# BUILD METADATA
# ================================================================
def build_metadata(class_folders):
    image_paths = []

    for cls in class_folders:
        pattern = os.path.join(DATA_DIR, cls, "*")
        valid_ext = (".jpg", ".jpeg", ".png")
        class_imgs = [p for p in glob(pattern) if p.lower().endswith(valid_ext)]
        image_paths.extend(class_imgs)

    print(f"✓ Total images found: {len(image_paths)}")

    labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    df = pd.DataFrame({"file_path": image_paths, "label": labels})
    df.to_csv(METADATA_CSV, index=False)

    print(f"✓ Saved metadata → {METADATA_CSV}")
    return df


# ================================================================
# PLOT CLASS DISTRIBUTION
# ================================================================
def plot_class_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="label")
    plt.title("Class Distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ================================================================
# ENCODE LABELS AND SAVE
# ================================================================
def encode_labels(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    encoded = le.fit_transform(df["label"].values)

    with open(LABEL_ENCODER_PKL, "wb") as f:
        pickle.dump(le, f)

    classes = list(le.classes_)
    with open(CLASSES_JSON, "w") as f:
        json.dump(classes, f, indent=2)

    print(f"✓ Saved label encoder → {LABEL_ENCODER_PKL}")
    print(f"✓ Saved class list → {CLASSES_JSON}: {classes}")

    return encoded, classes


# ================================================================
# RESIZE IMAGES + SAVE TO NUMPY
# ================================================================
def resize_images(df, encoded_labels):
    resized_images = []
    encoded_list = []
    bad_files = []

    for idx, row in df.iterrows():
        path = row["file_path"]
        try:
            img = Image.open(path).convert("RGB")
            img_resized = img.resize(TARGET_SIZE)
            resized_images.append(np.array(img_resized))
            encoded_list.append(int(encoded_labels[idx]))
        except Exception as e:
            bad_files.append((path, str(e)))

    if bad_files:
        print("⚠ Some images failed to load. Showing first 5:")
        print(bad_files[:5])

    resized_images = np.array(resized_images)
    encoded_list = np.array(encoded_list)

    np.save(RESIZED_IMAGES_NPY, resized_images)
    np.save(ENCODED_LABELS_NPY, encoded_list)

    print(f"✓ Saved resized images → {RESIZED_IMAGES_NPY}")
    print(f"✓ Saved encoded labels → {ENCODED_LABELS_NPY}")

    return resized_images, encoded_list


# ================================================================
# TFRECORD CREATION (Optional)
# ================================================================
def create_tfrecord(images, labels):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    with tf.io.TFRecordWriter(TFRECORD_PATH) as writer:
        for img_arr, lbl in zip(images, labels):
            encoded_jpeg = tf.io.encode_jpeg(img_arr).numpy()
            features = {
                "image": _bytes_feature(encoded_jpeg),
                "label": _int64_feature(int(lbl))
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=features)
            )
            writer.write(example.SerializeToString())

    print(f"✓ Saved TFRecord dataset → {TFRECORD_PATH}")


# ================================================================
# PREVIEW SAMPLE IMAGES
# ================================================================
def preview_samples(images, labels, class_names):
    plt.figure(figsize=(8, 8))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ================================================================
# MAIN PIPELINE
# ================================================================
if __name__ == "__main__":
    print("🚀 Starting Eco Dataset Builder...")

    # 1. scan folders
    class_folders = discover_classes()

    # 2. metadata
    df = build_metadata(class_folders)

    # 3. EDA plot
    plot_class_distribution(df)

    # 4. encode labels
    encoded_labels, class_names = encode_labels(df)

    # 5. resize images and save
    images, enc = resize_images(df, encoded_labels)

    # 6. create TFRecord
    create_tfrecord(images, enc)

    # 7. preview sample outputs
    preview_samples(images, enc, class_names)

    print("🎉 Dataset build complete!")
