import argparse
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from src.data import load_data, prepare_for_model


def train(data_path, target=None, test_size=0.2, random_state=42):
    df = load_data(data_path)
    X, y = prepare_for_model(df, target_col=target)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=(y if len(y.unique())>1 else None)
    )

    models = {}
    # Logistic Regression
    pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe_lr.fit(X_train, y_train)
    pred_lr = pipe_lr.predict(X_val)
    acc_lr = accuracy_score(y_val, pred_lr)
    models['logistic'] = (pipe_lr, acc_lr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_val)
    acc_rf = accuracy_score(y_val, pred_rf)
    models['random_forest'] = (rf, acc_rf)

    # choose best
    best_name, (best_model, best_acc) = max(models.items(), key=lambda kv: kv[1][1])

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'best_model.joblib')
    joblib.dump(best_model, model_path)

    # save metadata (feature names)
    meta = {
        'features': list(X.columns),
        'model': best_name,
        'accuracy': best_acc
    }
    with open(os.path.join('models', 'metadata.json'), 'w') as f:
        json.dump(meta, f)

    print(f"Trained models: logistic={acc_lr:.4f}, rf={acc_rf:.4f}")
    print(f"Saved best model ({best_name}) to {model_path}, accuracy={best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to CSV file')
    parser.add_argument('--target', default=None, help='target column name (optional)')
    args = parser.parse_args()
    train(args.data, target=args.target)


if __name__ == '__main__':
    main()
