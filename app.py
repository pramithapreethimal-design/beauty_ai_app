from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ------------------------------
# Flask Setup
# ------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------------------
# Load AI Model
# ------------------------------
model = load_model("model/skin_type_model.h5")

# ------------------------------
# Product Dictionary
# ------------------------------
products = {
    "oily": [
        {"name": "Oil-free cleanser", "description": "Removes excess oil", "price": 2500},
        {"name": "Salicylic acid face wash", "description": "Prevents acne", "price": 3000},
        {"name": "Matte moisturizer", "description": "Reduces shine", "price": 2800}
    ],
    "dry": [
        {"name": "Hydrating cream", "description": "Moisturizes skin", "price": 3200},
        {"name": "Gentle cleanser", "description": "Prevents dryness", "price": 2700},
        {"name": "Hyaluronic acid serum", "description": "Deep hydration", "price": 3500}
    ],
    "normal": [
        {"name": "Balanced moisturizer", "description": "Maintains skin balance", "price": 3000},
        {"name": "Sunscreen SPF50", "description": "Protects from UV", "price": 2800},
        {"name": "Vitamin C cream", "description": "Brightens skin", "price": 3200}
    ]
}

# ------------------------------
# Routes
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('image')

        if not file or file.filename == '':
            return render_template("index.html", skin=None, error="No file selected!")

        try:
            # Save uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # -------------------------
            # Image Preprocessing
            # -------------------------
            img = cv2.imread(filepath)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.reshape(img, (1, 224, 224, 3))

            # -------------------------
            # Prediction
            # -------------------------
            prediction = model.predict(img)
            class_names = ['dry', 'normal', 'oily']
            predicted_index = np.argmax(prediction)
            predicted_skin = class_names[predicted_index]
            confidence = prediction[0][predicted_index] * 100

            # Debug prints
            print("\n--- Prediction Probabilities ---")
            for i, cls in enumerate(class_names):
                print(f"{cls}: {prediction[0][i]*100:.2f}%")
            print(f"Predicted Skin Type: {predicted_skin}\n")

            # Get recommended products
            recommended = products.get(predicted_skin, [])

            return render_template(
                "index.html",
                skin=predicted_skin,
                confidence=confidence,
                probabilities=prediction[0],
                products=recommended,
                image_file=file.filename
            )

        except Exception as e:
            print("Error:", e)
            return render_template("index.html", skin=None, error="Prediction failed!")

    return render_template("index.html", skin=None)

# ------------------------------
# Run Server
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
