import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import random

# =========================================================
# 1. SETTINGS
# =========================================================

DATA_DIR = Path(
    "dataset/PlantVillage-Dataset-master/"
    "PlantVillage-Dataset-master/raw/color"
)

# Use the CORRECT CNN model
MODEL_PATH = "model/plant_disease_cnn_correct_best.keras"

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 2. LOAD CLASS NAMES
# =========================================================

class_names = sorted([
    folder.name
    for folder in DATA_DIR.iterdir()
    if folder.is_dir()
])

num_classes = len(class_names)

print("\n========================================")
print("PLANT DISEASE CNN EVALUATION")
print("========================================")

print("Number of classes:", num_classes)

# =========================================================
# 3. RECREATE THE SAME VALIDATION SPLIT
# =========================================================

print("\nCreating validation dataset...")
print("Using the same 80/20 split as training...\n")

val_paths = []
val_labels = []

for class_index, class_name in enumerate(class_names):

    class_dir = DATA_DIR / class_name

    image_files = sorted([
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]
    ])

    # IMPORTANT:
    # Same random shuffle used during training
    random.shuffle(image_files)

    total_images = len(image_files)

    split_index = int(
        total_images * 0.8
    )

    class_val = image_files[
        split_index:
    ]

    val_paths.extend(
        [str(path) for path in class_val]
    )

    val_labels.extend(
        [class_index] * len(class_val)
    )

    print(
        f"{class_index:2d} | "
        f"{class_name:<55} | "
        f"Validation: {len(class_val):5d}"
    )

print("\n========================================")
print("VALIDATION DATASET CREATED")
print("========================================")

print(
    "Total validation images:",
    len(val_paths)
)

# =========================================================
# 4. IMAGE LOADING FUNCTION
# =========================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# =========================================================
# 5. CREATE VALIDATION DATASET
# =========================================================

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

# =========================================================
# 6. LOAD CORRECT CNN MODEL
# =========================================================

print("\n========================================")
print("LOADING CORRECT CNN MODEL")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully!"
)

# =========================================================
# 7. EVALUATE MODEL
# =========================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

loss, accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\n========================================")
print("FINAL EVALUATION RESULTS")
print("========================================")

print(
    f"Validation Loss: {loss:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)

# =========================================================
# 8. GENERATE PREDICTIONS
# =========================================================

print("\nGenerating predictions...")
print("Please wait...\n")

predictions = model.predict(
    val_ds,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = np.array(
    val_labels
)

print(
    "\nPredictions generated successfully!"
)

# =========================================================
# 9. CLASSIFICATION REPORT
# =========================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================\n")

all_labels = np.arange(
    num_classes
)

report = classification_report(
    y_true,
    y_pred,
    labels=all_labels,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# Save report

Path("model").mkdir(
    exist_ok=True
)

with open(
    "model/correct_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)

# =========================================================
# 10. CONFUSION MATRIX
# =========================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=all_labels
)

fig, ax = plt.subplots(
    figsize=(20, 20)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    colorbar=False
)

plt.title(
    "Plant Disease CNN - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "model/correct_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# 11. FINAL RESULTS
# =========================================================

print("\n========================================")
print("EVALUATION COMPLETED SUCCESSFULLY! 🌱")
print("========================================")

print(
    f"\nValidation Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nFiles created:")

print(
    "1. model/correct_classification_report.txt"
)

print(
    "2. model/correct_confusion_matrix.png"
)

print("\n========================================")
print("DONE")
print("========================================")