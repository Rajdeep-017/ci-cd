import joblib
import sys

def run_prediction(features):
    try:
        model = joblib.load('app/models/model.joblib')
        prediction = model.predict([features])
        return prediction[0]
    except FileNotFoundError:
        print("Error: Model file not found. Run train.py first.")
        sys.exit(1)

if __name__ == "__main__":
    # Example hardcoded features for testing: [5.1, 3.5, 1.4, 0.2]
    sample_input = [float(x) for x in sys.argv[1:5]] if len(sys.argv) > 1 else [5.1, 3.5, 1.4, 0.2]
    result = run_prediction(sample_input)
    print(f"Prediction for {sample_input}: {result}")