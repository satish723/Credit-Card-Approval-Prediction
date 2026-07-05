Credit Card Approval Prediction

Minimal scaffold for a credit-card-approval ML project.

Quick steps

- Create a virtual env and install deps: `pip install -r requirements.txt`
- Train a model: `python -m src.train --data creditcard.csv --target TARGET_COL` (omit `--target` to use last column)
- Run the prediction app: `python -m src.app`

Files created

- [src/data.py](src/data.py) — data loading & preprocessing helpers
- [src/train.py](src/train.py) — training script (LR + RF), saves best model to `models/`
- [src/app.py](src/app.py) — simple Flask app to upload CSV and get predictions
- [templates/index.html](templates/index.html) — upload form
- [templates/result.html](templates/result.html) — basic results view
- [requirements.txt](requirements.txt)
- [.gitignore](.gitignore)

What you need to do

1. Install dependencies: `pip install -r requirements.txt`
2. Train a model from your CSV: `python -m src.train --data creditcard.csv`
3. Start the app: `python -m src.app` and open `http://127.0.0.1:5000`

If you want, I can run training and start the app for you (I can't run processes without your permission).
