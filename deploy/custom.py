from pathlib import Path
import joblib


def load_model(code_dir):
    code_path = Path(code_dir) / Path("model.pkl")

    model = joblib.load(code_path)
    return model
