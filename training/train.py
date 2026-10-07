import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random

# =========================================================
# 1. SETTINGS
# =========================================================

DATA_DIR = Path(
    "dataset/PlantVillage-Dataset-master/"
    "PlantVillage-Dataset-master/raw/color"
)

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 2. GET ALL CLASSES
# =========================================================

print("\n========================================")
print("PLANT DISEASE DATASET")
print("========================================")

class_names = sorted([
    folder.name
    for folder in DATA_DIR.iterdir()
    if folder.is_dir()
])

num_classes = len(class_names)

print("Number of classes:", num_classes)

# =========================================================
# 3. CREATE PROPER STRATIFIED TRAIN/VALIDATION SPLIT
# =========================================================

print("\nCreating proper train/validation split...")
print("80% training + 20% validation")
print("Every class will be represented in both sets.\n")

train_paths = []
train_labels = []

val_paths = []
val_labels = []

for class_index, class_name in enumerate(class_names):

    class_dir = DATA_DIR / class_name

    image_files = sorted([
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    # Shuffle images inside each class
    random.shuffle(image_files)

    total_images = len(image_files)

    split_index = int(total_images * 0.8)

    class_train = image_files[:split_index]
    class_val = image_files[split_index:]

    train_paths.extend(
        [str(path) for path in class_train]
    )

    train_labels.extend(
        [class_index] * len(class_train)
    )

    val_paths.extend(
        [str(path) for path in class_val]
    )

    val_labels.extend(
        [class_index] * len(class_val)
    )

    print(
        f"{class_index:2d} | "
        f"{class_name:<55} | "
        f"Total: {total_images:5d} | "
        f"Train: {len(class_train):5d} | "
        f"Val: {len(class_val):5d}"
    )

print("\n========================================")
print("SPLIT COMPLETED")
print("========================================")

print("Total training images:", len(train_paths))
print("Total validation images:", len(val_paths))

# =========================================================
# 4. SHUFFLE TRAINING DATA
# =========================================================

train_combined = list(
    zip(train_paths, train_labels)
)

random.shuffle(train_combined)

train_paths, train_labels = zip(
    *train_combined
)

train_paths = list(train_paths)
train_labels = list(train_labels)

# =========================================================
# 5. IMAGE LOADING FUNCTION
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
# 6. CREATE TF.DATA DATASETS
# =========================================================

print("\nCreating TensorFlow datasets...")

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

# =========================================================
# 7. CALCULATE CLASS WEIGHTS
# =========================================================

print("\nCalculating class weights...")

class_counts = np.bincount(
    train_labels,
    minlength=num_classes
)

total_train_images = len(train_labels)

class_weights = {}

for i in range(num_classes):

    class_weights[i] = (
        total_train_images /
        (num_classes * class_counts[i])
    )

print("\nClass weights:")

for i in range(num_classes):

    print(
        f"{i:2d}: "
        f"{class_names[i]:<55} "
        f"images={class_counts[i]:5d} "
        f"weight={class_weights[i]:.4f}"
    )

# =========================================================
# 8. DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.08
    ),

    tf.keras.layers.RandomZoom(
        0.08
    )
])

# =========================================================
# 9. CNN MODEL
# =========================================================

print("\n========================================")
print("CREATING CNN MODEL")
print("========================================")

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(96, 96, 3)
    ),

    data_augmentation,

    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # Convolution Block 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Convolution Block 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Convolution Block 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Feature extraction
    tf.keras.layers.GlobalAveragePooling2D(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.4
    ),

    # Output layer
    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])

model.summary()

# =========================================================
# 10. COMPILE MODEL
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

# =========================================================
# 11. CALLBACKS
# =========================================================

Path("model").mkdir(
    exist_ok=True
)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "model/plant_disease_cnn_correct_best.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# =========================================================
# 12. TRAIN MODEL
# =========================================================

print("\n========================================")
print("STARTING CORRECT CNN TRAINING")
print("========================================")

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Classes:", num_classes)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)

print("\nTraining started...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)

# =========================================================
# 13. SAVE FINAL MODEL
# =========================================================

model.save(
    "model/plant_disease_cnn_correct_final.keras"
)

# =========================================================
# 14. SAVE CLASS NAMES
# =========================================================

with open(
    "model/class_names.txt",
    "w",
    encoding="utf-8"
) as file:

    for class_name in class_names:
        file.write(
            class_name + "\n"
        )

# =========================================================
# 15. ACCURACY GRAPH
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.legend()

plt.savefig(
    "model/correct_accuracy.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# 16. LOSS GRAPH
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "CNN Training and Validation Loss"
)

plt.legend()

plt.savefig(
    "model/correct_loss.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# 17. FINAL RESULTS
# =========================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)

print("\n========================================")
print("CORRECT CNN TRAINING COMPLETED!")
print("========================================")

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print("\nSaved models:")

print(
    "model/plant_disease_cnn_correct_best.keras"
)

print(
    "model/plant_disease_cnn_correct_final.keras"
)

print("\nSaved class names:")

print(
    "model/class_names.txt"
)

print("\nSaved graphs:")

print(
    "model/correct_accuracy.png"
)

print(
    "model/correct_loss.png"
)

print("\n========================================")
print("TRAINING FINISHED SUCCESSFULLY! 🌱")
print("========================================")