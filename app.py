from flask import Flask, render_template, request
import joblib
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
app = Flask(__name__)

try:
    model = joblib.load("credit_card_model.pkl")
except Exception as e:
    model = None
    print("Model load failed:", e)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", prediction="Model not available")

    if 'file' not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", prediction="No file selected")

    try:
        import pandas as pd
        df = pd.read_csv(file)
    except Exception:
        return render_template("index.html", prediction="Could not read CSV file")

    if df.shape[1] == 0:
        return render_template("index.html", prediction="Uploaded file has no features")

    # If the uploaded file has a target column, drop it if there are too many columns.
    if hasattr(model, 'n_features_in_') and df.shape[1] > model.n_features_in_:
        df = df.iloc[:, :-1]

    if hasattr(model, 'n_features_in_') and df.shape[1] != model.n_features_in_:
        return render_template(
            "index.html",
            prediction=f"CSV has {df.shape[1]} features but model requires {model.n_features_in_}."
        )

    try:
        predictions = model.predict(df.values)
    except Exception:
        return render_template("index.html", prediction="Prediction failed: feature mismatch or bad data")

    result = ", ".join(str(int(x)) for x in predictions)
    return render_template("index.html", prediction=f"Predictions: {result}")

if __name__ == "__main__":
    app.run(debug=True)
