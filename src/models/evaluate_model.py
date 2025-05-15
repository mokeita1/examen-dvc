import pandas as pd
import os
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

def main():
    data_dir = "data/processed/"
    model_dir = "models/"
    metrics_dir = "metrics/"
    os.makedirs(metrics_dir, exist_ok=True)

    # Chargement des données test
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Chargement du modèle entraîné
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modèle introuvable. Lance train_model.py d'abord.")

    model = joblib.load(model_path)

    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "mse": mse,
        "r2": r2
    }

    # Sauvegarde des métriques
    with open(os.path.join(metrics_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=4)

    # Sauvegarde des prédictions
    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    predictions_df.to_csv(os.path.join(data_dir, "predictions.csv"), index=False)

    print("Évaluation terminée.")
    print(f"MSE: {mse:.4f} | 📈 R²: {r2:.4f}")

if __name__ == "__main__":
    main()