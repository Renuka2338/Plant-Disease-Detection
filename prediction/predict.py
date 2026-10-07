import tensorflow as tf
import numpy as np
from pathlib import Path

# =========================================================
# 1. SETTINGS
# =========================================================

MODEL_PATH = "model/plant_disease_cnn_correct_best.keras"
CLASS_NAMES_PATH = "model/class_names.txt"

IMG_SIZE = (96, 96)

# =========================================================
# 2. LOAD MODEL
# =========================================================

print("\n========================================")
print("PLANT DISEASE PREDICTION")
print("========================================")

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

# =========================================================
# 3. LOAD CLASS NAMES
# =========================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]

print(
    "Number of classes:",
    len(class_names)
)

# =========================================================
# 4. GET IMAGE PATH
# =========================================================

image_path = input(
    "\nEnter the path of the plant leaf image: "
).strip()

# Remove quotes if path is pasted with quotes
image_path = image_path.strip('"').strip("'")

if not Path(image_path).exists():

    print("\nERROR: Image file not found!")

    print(
        "Please check the image path and try again."
    )

    exit()

# =========================================================
# 5. LOAD AND PREPROCESS IMAGE
# =========================================================

print("\nProcessing image...")

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMG_SIZE
)

image_array = tf.keras.utils.img_to_array(
    image
)

image_array = np.expand_dims(
    image_array,
    axis=0
)

# =========================================================
# 6. MAKE PREDICTION
# =========================================================

print("Predicting disease...\n")

predictions = model.predict(
    image_array,
    verbose=0
)

# =========================================================
# 7. GET TOP 3 PREDICTIONS
# =========================================================

top_3_indices = np.argsort(
    predictions[0]
)[-3:][::-1]

# =========================================================
# 8. DISPLAY RESULTS
# =========================================================

print("========================================")
print("TOP 3 PREDICTIONS")
print("========================================")

for rank, index in enumerate(
    top_3_indices,
    start=1
):

    disease = class_names[index]

    confidence = (
        predictions[0][index] * 100
    )

    print(
        f"\n{rank}. {disease}"
    )

    print(
        f"   Confidence: {confidence:.2f}%"
    )

print("\n========================================")
print("PREDICTION COMPLETED! 🌱")
print("========================================")