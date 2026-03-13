# src/cnn_model.py
"""
CNN Training Script for Ecosystem Image Classification
Model: MobileNetV2 Transfer Learning
Classes: Mangrove, Seagrass, Saltmarsh, etc.

Outputs:
- models/eco_model.h5
- models/classes.json
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===========================================================
# CONFIG
# ===========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

DATA_DIR = "Dataset2"   # Folder containing class folders
MODEL_OUT = "models/eco_model.h5"
CLASSES_OUT = "models/classes.json"

os.makedirs("models", exist_ok=True)


# ===========================================================
# DATA PIPELINE (With Augmentation)
# ===========================================================
train_aug = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = train_aug.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    shuffle=True
)

val_gen = train_aug.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    shuffle=False
)


# ===========================================================
# MODEL: MobileNetV2 Transfer Learning
# ===========================================================
base = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(224, 224, 3),
    weights="imagenet"
)

# Freeze base model for faster training
base.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.25)(x)
out = tf.keras.layers.Dense(train_gen.num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ===========================================================
# CALLBACKS
# ===========================================================
checkpoint = ModelCheckpoint(
    MODEL_OUT,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# ===========================================================
# TRAINING
# ===========================================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# Save final best version manually (backup)
model.save(MODEL_OUT)


# ===========================================================
# SAVE CLASS LABELS
# ===========================================================
classes = list(train_gen.class_indices.keys())
with open(CLASSES_OUT, "w") as f:
    json.dump(classes, f, indent=2)

print("✓ CNN training complete")
print(f"Saved model to {MODEL_OUT}")
print(f"Saved classes to {CLASSES_OUT}: {classes}")
