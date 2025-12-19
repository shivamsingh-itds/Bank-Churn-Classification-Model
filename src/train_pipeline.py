import joblib
from pathlib import Path

from src.data_preprocessing import data_preprocessing
from src.train_model import train_model
from src.evaluate import evaluate_model

def main():
    # Preprocessing
    X_train, X_test, y_train, y_test = data_preprocessing()

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    accuracy, report, cm = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    # 🔥 SAVE MODEL USING JOBLIB
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "random_forest_model.pkl"
    joblib.dump(model, model_path)

    print(f"\nModel saved at: {model_path}")

if __name__ == "__main__":
    main()
