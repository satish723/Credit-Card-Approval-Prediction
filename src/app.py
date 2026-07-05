import os
import io
import json
from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib

app = Flask(__name__)


def load_model_and_meta():
    model_path = os.path.join('models', 'best_model.joblib')
    meta_path = os.path.join('models', 'metadata.json')
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return model, meta


@app.route('/', methods=['GET', 'POST'])
def index():
    model, meta = load_model_and_meta()
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        df = pd.read_csv(file)
        if model is None or meta is None:
            return 'Model not found. Train model first (see README).', 500

        # Basic preprocessing: align features
        X = pd.get_dummies(df.drop(columns=[df.columns[-1]]) if len(df.columns)>1 else df)
        # ensure all columns present
        for c in meta['features']:
            if c not in X.columns:
                X[c] = 0
        X = X[meta['features']]

        preds = model.predict(X)
        out = df.copy()
        out['prediction'] = preds
        # render top rows
        return render_template('result.html', tables=[out.head(50).to_html(classes='data', index=False, escape=False)], titles=out.columns.values)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
