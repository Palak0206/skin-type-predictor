
import os
import joblib
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename

from preprocessing import (
    basic_preprocessing,
    spelling_correction,
    stopword_removal,
    lemmatization
)


app = Flask(__name__)


# --------------------------------
# Configuration
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMAGE_SIZE = (128,128)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


# --------------------------------
# Load pretrained models
# --------------------------------

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# Change "texts" to "text" if that is your actual folder name
NLP_DIR = os.path.join(
    MODEL_DIR,
    "text"
)

CNN_DIR = os.path.join(
    MODEL_DIR,
    "image"
)


# NLP Model 1: Skin Type Prediction
skin_type_model = joblib.load(
    os.path.join(
        NLP_DIR,
        "linear_skin_predictor.pkl"
    )
)


# NLP Model 2: Skin Characteristics
feature_intensity_model = joblib.load(
    os.path.join(
        NLP_DIR,
        "skin_attributes_lr_pipeline.pkl"
    )
)


# CNN Model: Image-Based Skin Type Prediction
cnn_model = load_model(
    os.path.join(
        CNN_DIR,
        "skin_type_cnn.keras"
    )
)


# --------------------------------
# Routes
# --------------------------------

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------
    # Get user inputs
    # --------------------------------

    skin_description = request.form.get(
        "skin_description",
        ""
    ).strip()

    image_file = request.files.get(
        "skin_image"
    )

    has_image = (
        image_file is not None
        and image_file.filename != ""
    )


    # --------------------------------
    # Validate inputs
    # --------------------------------

    if not skin_description and not has_image:

        return render_template(
            "home.html",
            error="Please enter a skin description or upload an image."
        )


    # --------------------------------
    # NLP Model 1: Skin Type Prediction
    # --------------------------------

    nlp_skin_type = None

    if skin_description:

        nlp_skin_type = skin_type_model.predict(
            [skin_description]
        )[0]


    # --------------------------------
    # NLP Model 2: Skin Characteristics
    # --------------------------------

    feature_intensities = {}

    if skin_description:

        intensity_prediction = feature_intensity_model.predict(
            [skin_description]
        )

        prediction = np.asarray(
            intensity_prediction
        )[0]

        feature_names = [
            "Oiliness",
            "Dryness",
            "Sensitivity"
        ]

        for name, value in zip(
            feature_names,
            prediction
        ):

            feature_intensities[name] = value


    # --------------------------------
    # CNN: Image-Based Skin Type Prediction
    # --------------------------------

    cnn_skin_type = None
    uploaded_image_path = None

    if has_image:

        # Secure the uploaded filename
        filename = secure_filename(
            image_file.filename
        )

        if not filename:

            return render_template(
                "home.html",
                error="Please upload a valid image filename."
            )


        # Check image extension
        extension = filename.rsplit(
            ".",
            1
        )[-1].lower()

        if extension not in ALLOWED_EXTENSIONS:

            return render_template(
                "home.html",
                error="Please upload a PNG, JPG, JPEG, or WEBP image."
            )


        # Save uploaded image
        full_image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        image_file.save(
            full_image_path
        )


        # Store path relative to the static folder
        uploaded_image_path = os.path.join(
            "uploads",
            filename
        ).replace("\\", "/")


        # Load and preprocess image
        image = Image.open(
            full_image_path
        ).convert("RGB")

        image = image.resize(
            IMAGE_SIZE
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )


        # Normalize pixel values
        # Assumes CNN was trained using pixel values in range 0-1
        image_array = image_array / 255.0


        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # --------------------------------
        # CNN Prediction
        # --------------------------------

        cnn_prediction = cnn_model.predict(
            image_array,
            verbose=0
        )


        # Debug output in Flask terminal
        print(
            "CNN output shape:",
            cnn_prediction.shape
        )

        print(
            "CNN raw prediction:",
            cnn_prediction
        )


        # Get predicted class index
        predicted_class_index = int(
            np.argmax(
                cnn_prediction,
                axis=1
            )[0]
        )

        print(
            "Predicted class index:",
            predicted_class_index
        )


        # IMPORTANT:
        # These must match the class order used during CNN training
        class_names = [
            "Oily",
            "Normal",
            "Combination",
            "Dry"
]

        # Check index BEFORE accessing the list
        if predicted_class_index >= len(class_names):

            raise ValueError(
                f"CNN predicted class index {predicted_class_index}, "
                f"but class_names only has {len(class_names)} classes. "
                f"CNN output shape: {cnn_prediction.shape}"
                f"The indices are : {train_generator.class_indices}"
            )


        # Get predicted skin type
        cnn_skin_type = class_names[
            predicted_class_index
        ]


    # --------------------------------
    # Display Results
    # --------------------------------

    return render_template(
        "results.html",
        nlp_skin_type=nlp_skin_type,
        feature_intensities=feature_intensities,
        cnn_skin_type=cnn_skin_type,
        uploaded_image_path=uploaded_image_path
    )


# --------------------------------
# Run Flask
# --------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )
