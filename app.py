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
from cosine import ProductRecommender


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

IMAGE_SIZE = (128, 128)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


# --------------------------------
# Product recommendation dataset
# --------------------------------

PRODUCT_DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "product-recomendation.xlsx"
)

# Load and fit the recommendation encoder only once when Flask starts.
# If the workbook is missing or invalid, skin predictions will still work.
product_recommender = None

try:
    product_recommender = ProductRecommender(
        dataset_path=PRODUCT_DATASET_PATH
    )
    app.logger.info("Product recommendation dataset loaded successfully.")

except Exception as exc:
    app.logger.exception(
        "Product recommendations are unavailable: %s",
        exc
    )


# --------------------------------
# Load pretrained models
# --------------------------------

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

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
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------
    # Get user inputs
    # --------------------------------

    skin_description = request.form.get(
        "skin_description",
        ""
    ).strip()

    image_file = request.files.get("skin_image")

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

        prediction = np.asarray(intensity_prediction)[0]

        feature_names = [
            "Oiliness",
            "Dryness",
            "Sensitivity"
        ]

        for name, value in zip(feature_names, prediction):
            feature_intensities[name] = str(value)


    # --------------------------------
    # CNN: Image-Based Skin Type Prediction
    # --------------------------------

    cnn_skin_type = None
    uploaded_image_path = None

    if has_image:

        # Secure the uploaded filename
        filename = secure_filename(image_file.filename)

        if not filename:
            return render_template(
                "home.html",
                error="Please upload a valid image filename."
            )


        # Check image extension
        extension = filename.rsplit(".", 1)[-1].lower()

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

        image_file.save(full_image_path)


        # Store path relative to the static folder
        uploaded_image_path = os.path.join(
            "uploads",
            filename
        ).replace("\\", "/")


        # Load and preprocess image
        image = Image.open(full_image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Normalize pixel values (CNN trained with values in range 0-1)
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)


        # CNN prediction
        cnn_prediction = cnn_model.predict(
            image_array,
            verbose=0
        )

        print("CNN output shape:", cnn_prediction.shape)
        print("CNN raw prediction:", cnn_prediction)

        predicted_class_index = int(
            np.argmax(cnn_prediction, axis=1)[0]
        )

        print("Predicted class index:", predicted_class_index)

        # This order must match the class order used during CNN training.
        class_names = [
            "Oily",
            "Normal",
            "Combination",
            "Dry"
        ]

        if predicted_class_index >= len(class_names):
            raise ValueError(
                f"CNN predicted class index {predicted_class_index}, "
                f"but class_names only has {len(class_names)} classes. "
                f"CNN output shape: {cnn_prediction.shape}"
            )

        cnn_skin_type = class_names[predicted_class_index]


    # --------------------------------
    # Product recommendations
    # --------------------------------

    recommendations = {}
    recommendation_skin_type = None
    recommendation_intensities = {}
    defaulted_intensities = []
    recommendation_error = None

    # Prefer the image prediction when available; otherwise use the NLP result.
    recommendation_skin_type = (
        cnn_skin_type if cnn_skin_type is not None else nlp_skin_type
    )

    if recommendation_skin_type is not None and product_recommender is not None:

        # NLP Model 2 outputs these feature names. If a feature is unavailable
        # (for example, image-only input), Medium is a neutral system default.
        intensity_mapping = {
            "Oily": "Oiliness",
            "Dry": "Dryness",
            "Sensitive": "Sensitivity"
        }

        for product_feature, model_feature in intensity_mapping.items():
            value = feature_intensities.get(model_feature)

            if value not in {"Low", "Medium", "High"}:
                value = "Medium"
                defaulted_intensities.append(model_feature)

            recommendation_intensities[product_feature] = value

        try:
            recommendations = product_recommender.recommend_by_category(
                skin_type=recommendation_skin_type,
                oily=recommendation_intensities["Oily"],
                dry=recommendation_intensities["Dry"],
                sensitive=recommendation_intensities["Sensitive"],
                top_n=3
            )

        except Exception as exc:
            app.logger.exception(
                "Could not generate product recommendations: %s",
                exc
            )
            recommendation_error = (
                "Product recommendations could not be generated for this result."
            )

    elif recommendation_skin_type is not None and product_recommender is None:
        recommendation_error = (
            "The product recommendation dataset is currently unavailable."
        )


    # --------------------------------
    # Display Results
    # --------------------------------

    return render_template(
        "results.html",
        nlp_skin_type=nlp_skin_type,
        feature_intensities=feature_intensities,
        cnn_skin_type=cnn_skin_type,
        uploaded_image_path=uploaded_image_path,
        recommendations=recommendations,
        recommendation_skin_type=recommendation_skin_type,
        recommendation_intensities=recommendation_intensities,
        defaulted_intensities=defaulted_intensities,
        recommendation_error=recommendation_error
    )


# --------------------------------
# Run Flask
# --------------------------------

if __name__ == "__main__":
    print("\n========== REGISTERED ROUTES ==========")
    print(app.url_map)
    print("========================================\n")

    app.run(debug=True)