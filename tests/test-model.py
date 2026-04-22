from app.train import train
import os

def test_training():
    train()
    assert os.path.exists("model.pkl")