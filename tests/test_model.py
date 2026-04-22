import os
import pytest
from app.train import train
from app.predict import run_prediction

def test_training_pipeline():
    train()
    assert os.path.exists('app/models/model.joblib')

def test_prediction_logic():
    # Ensure model exists for the test
    if not os.path.exists('app/models/model.joblib'):
        train()
    
    result = run_prediction([5.1, 3.5, 1.4, 0.2])
    assert result in [0, 1, 2]  # Iris classes
