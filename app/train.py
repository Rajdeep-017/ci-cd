import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

def train():
    data = load_iris()
    X, y = data.data, data.target

    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)

    os.makedirs('app/models', exist_ok=True)
    joblib.dump(model, 'app/models/model.joblib')
    print("✓ Model trained and saved.")

if __name__ == "__main__":
    train()