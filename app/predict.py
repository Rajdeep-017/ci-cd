import joblib

def predict(sample):
    model = joblib.load("model.pkl")
    return model.predict([sample])