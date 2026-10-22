"""
Alzheimer's Disease Detection — End-to-End Training Pipeline
Uses MobileNetV2 transfer learning, which is the correct approach for
a small dataset (80 train / 20 test images).

Pipeline:
  1. Data loading with aggressive augmentation
  2. MobileNetV2 feature extractor (ImageNet weights, frozen)  →  head training
  3. Fine-tuning: unfreeze top blocks
  4. Evaluation on clean test set
  5. model.h5 saved
  6. Per-image inference demo
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from sklearn.metrics import classification_report, confusion_matrix

# ── reproducibility ──────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 65)
print("  Alzheimer Detection — Training Pipeline  (Transfer Learning)")
print(f"  TF {tf.__version__}  |  Python {sys.version.split()[0]}")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_DIR  = os.path.join("dataset", "train")
TEST_DIR   = os.path.join("dataset", "test")
CLASSES    = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
IMG_SIZE   = 160      # MobileNetV2 minimum; use 160 to keep speed reasonable
BATCH_SIZE = 8        # tiny dataset: smaller batches → more weight updates/epoch
EPOCHS_HEAD = 30      # Phase 1: train head only
EPOCHS_FINE = 30      # Phase 2: fine-tune top layers
MODEL_PATH  = "model.h5"

print(f"\n  Train dir : {TRAIN_DIR}")
print(f"  Test  dir : {TEST_DIR}")
print(f"  Classes   : {CLASSES}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Building data generators …")

train_aug = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
    validation_split=0.2,        # 16 images → val, 64 → train
)

test_aug = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_aug.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, classes=CLASSES,
    class_mode="categorical", subset="training",
    shuffle=True, seed=42,
)
val_gen = train_aug.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, classes=CLASSES,
    class_mode="categorical", subset="validation",
    shuffle=False, seed=42,
)
test_gen = test_aug.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, classes=CLASSES,
    class_mode="categorical", shuffle=False,
)

print(f"      Train : {train_gen.samples} images  "
      f"| Val : {val_gen.samples}  | Test : {test_gen.samples}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD MODEL  (MobileNetV2 backbone + custom head)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Building MobileNetV2 transfer-learning model …")

base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False          # Phase 1: freeze backbone

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(4, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=[
        keras.metrics.CategoricalAccuracy(name="acc"),
        keras.metrics.AUC(name="auc"),
    ],
)
print(f"      Trainable params : {model.count_params():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PHASE 1 — train head only
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3/6] Phase 1 — Head training (up to {EPOCHS_HEAD} epochs) …")

ckpt1 = ModelCheckpoint(MODEL_PATH, monitor="val_acc",
                        save_best_only=True, verbose=0)
early1 = EarlyStopping(monitor="val_loss", patience=12,
                       restore_best_weights=True, verbose=1)
lr1 = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                        patience=5, min_lr=1e-6, verbose=1)

history1 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS_HEAD, callbacks=[ckpt1, early1, lr1], verbose=1,
)

best_h1_acc = max(history1.history["val_acc"])
print(f"\n  Phase 1 best val_acc : {best_h1_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PHASE 2 — fine-tune top layers of MobileNetV2
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/6] Phase 2 — Fine-tuning (up to {EPOCHS_FINE} more epochs) …")

base.trainable = True
# Freeze all layers except the last 30 (top blocks)
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),   # lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=[
        keras.metrics.CategoricalAccuracy(name="acc"),
        keras.metrics.AUC(name="auc"),
    ],
)

ckpt2  = ModelCheckpoint(MODEL_PATH, monitor="val_acc",
                         save_best_only=True, verbose=0)
early2 = EarlyStopping(monitor="val_loss", patience=12,
                       restore_best_weights=True, verbose=1)
lr2 = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                        patience=5, min_lr=1e-7, verbose=1)

history2 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS_FINE, callbacks=[ckpt2, early2, lr2], verbose=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Loading best checkpoint → evaluating on test set …")

best_model = tf.keras.models.load_model(MODEL_PATH)

test_gen.reset()
loss, accuracy, auc = best_model.evaluate(test_gen, verbose=0)

print("\n" + "=" * 65)
print("  EVALUATION METRICS  (best val_acc checkpoint)")
print("=" * 65)
print(f"  Test Loss     : {loss:.4f}")
print(f"  Test Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"  Test AUC      : {auc:.4f}")
print("=" * 65)

# Per-class report
test_gen.reset()
y_pred_probs = best_model.predict(test_gen, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes[: len(y_pred)]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES,
                            zero_division=0))

print("Confusion Matrix  (rows=Actual, cols=Predicted):")
cm = confusion_matrix(y_true, y_pred)
header = "            " + "  ".join(f"{c[:10]:>10}" for c in CLASSES)
print(header)
for i, row in enumerate(cm):
    print(f"{CLASSES[i][:12]:>12}  " + "  ".join(f"{v:10d}" for v in row))

# ── Combined training curves ─────────────────────────────────────────────────
h1_loss = history1.history["loss"]
h1_vloss= history1.history["val_loss"]
h1_acc  = history1.history["acc"]
h1_vacc = history1.history["val_acc"]

h2_loss = history2.history["loss"]
h2_vloss= history2.history["val_loss"]
h2_acc  = history2.history["acc"]
h2_vacc = history2.history["val_acc"]

all_loss = h1_loss + h2_loss
all_vloss= h1_vloss+ h2_vloss
all_acc  = h1_acc  + h2_acc
all_vacc = h1_vacc + h2_vacc
ep = range(len(all_loss))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(ep, all_loss,  label="Train Loss")
axes[0].plot(ep, all_vloss, label="Val Loss")
axes[0].axvline(len(h1_loss)-1, color="gray", linestyle="--",
                label="Fine-tune start")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(ep, all_acc,  label="Train Acc")
axes[1].plot(ep, all_vacc, label="Val Acc")
axes[1].axvline(len(h1_acc)-1, color="gray", linestyle="--",
                label="Fine-tune start")
axes[1].set_title("Accuracy"); axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png")
print("\n  ✓ training_curves.png saved")

# ─────────────────────────────────────────────────────────────────────────────
# 6. INFERENCE DEMO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Per-image inference …")
print("\n" + "=" * 65)
print("  INFERENCE DEMO — Test Images")
print("=" * 65)

results = []
for cls_folder in sorted(CLASSES):
    cls_path = os.path.join(TEST_DIR, cls_folder)
    if not os.path.isdir(cls_path):
        continue
    for img_file in sorted(os.listdir(cls_path)):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        full_path = os.path.join(cls_path, img_file)
        img  = load_img(full_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr  = img_to_array(img) / 255.0
        arr  = np.expand_dims(arr, axis=0)
        pred = best_model.predict(arr, verbose=0)[0]
        idx  = int(np.argmax(pred))
        pred_cls  = CLASSES[idx]
        confidence = float(pred[idx])
        match = "✓" if pred_cls == cls_folder else "✗"
        results.append({
            "file":       img_file,
            "actual":     cls_folder,
            "predicted":  pred_cls,
            "confidence": f"{confidence:.2%}",
            "match":      match,
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))

correct = (df["match"] == "✓").sum()
total   = len(df)
print(f"\n  Per-image accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"\n  ✓ Model saved → {MODEL_PATH}")
print("\n  Pipeline complete.\n")
