import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_raw_data():
    path = BASE_DIR / "data" / "raw" / "Churn_Modelling.csv"
    return pd.read_csv(path)
