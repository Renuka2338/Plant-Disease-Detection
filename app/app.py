import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# 1. SETTINGS
# =========================================================

MODEL_PATH = "model/plant_disease_cnn_correct_best.keras"
CLASS_NAMES_PATH = "model/class_names.txt"

IMG_SIZE = (96, 96)

# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: bold;
        margin-top: 20px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 15px;
    }

    .top-result {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }

    .top-result h2 {
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 4. HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌿 Plant Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered plant leaf disease classification using a Convolutional Neural Network (CNN)'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# 5. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return [
            line.strip()
            for line in file
            if line.strip()
        ]


model = load_model()
class_names = load_class_names()

# =========================================================
# 6. MAIN LAYOUT
# =========================================================

left_column, right_column = st.columns(
    [1, 1]
)

# =========================================================
# 7. IMAGE UPLOAD
# =========================================================

with left_column:

    st.markdown(
        '<div class="section-title">📷 Upload Leaf Image</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a clear image of a plant leaf to detect the possible disease."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )

# =========================================================
# 8. PREDICTION
# =========================================================

with right_column:

    st.markdown(
        '<div class="section-title">🔍 Detection Result</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is None:

        st.info(
            "Please upload a plant leaf image to begin prediction."
        )

    else:

        # Resize image
        resized_image = image.resize(
            IMG_SIZE
        )

        # Convert to array
        image_array = np.array(
            resized_image
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Predict
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Get top 3
        top_3_indices = np.argsort(
            predictions
        )[-3:][::-1]

        # =================================================
        # TOP PREDICTION
        # =================================================

        top_index = top_3_indices[0]

        top_disease = class_names[
            top_index
        ]

        top_confidence = (
            predictions[top_index] * 100
        )

        # Make name readable
        readable_name = top_disease.replace(
            "___",
            " – "
        ).replace(
            "_",
            " "
        )

        st.markdown(
            f"""
            <div class="top-result">
                <h2>🥇 Most Likely Prediction</h2>
                <h3>{readable_name}</h3>
                <h2>{top_confidence:.2f}%</h2>
                <p>Confidence</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # HEALTH / DISEASE STATUS
        # =================================================

        if "healthy" in top_disease.lower():

            st.success(
                "🌱 The leaf appears to be healthy."
            )

        else:

            st.warning(
                "⚠️ A possible plant disease has been detected."
            )

        # =================================================
        # TOP 3 RESULTS
        # =================================================

        st.markdown(
            "### 🌿 Top 3 Predictions"
        )

        for rank, index in enumerate(
            top_3_indices,
            start=1
        ):

            disease = class_names[index]

            confidence = (
                predictions[index] * 100
            )

            readable_disease = disease.replace(
                "___",
                " – "
            ).replace(
                "_",
                " "
            )

            st.write(
                f"**{rank}. {readable_disease}**"
            )

            st.progress(
                float(predictions[index])
            )

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

# =========================================================
# 9. PROJECT INFORMATION
# =========================================================

st.divider()

st.markdown(
    "### 📌 About the Project"
)

st.write(
    """
    This project uses a Convolutional Neural Network (CNN) to
    classify plant leaf images into different disease categories.
    The model was trained using the PlantVillage dataset containing
    38 plant disease and healthy-leaf classes.
    """
)

# =========================================================
# 10. MODEL INFORMATION
# =========================================================

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Number of Classes",
        "38"
    )

with info2:

    st.metric(
        "Validation Accuracy",
        "81.35%"
    )

with info3:

    st.metric(
        "Model",
        "CNN"
    )

# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <p>🌱 Plant Disease Detection System</p>
        <p>Developed using Python, TensorFlow, CNN and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)