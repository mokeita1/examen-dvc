import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

def main():
    data_dir = "data/processed/"
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)

    # Chargement des données
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()

    # Chargement des meilleurs hyperparamètres
    best_params_path = os.path.join(model_dir, "best_params.pkl")
    if not os.path.exists(best_params_path):
        raise FileNotFoundError("Fichier best_params.pkl introuvable. Lance search_params.py d'abord.")

    best_params = joblib.load(best_params_path)

    # Entraînement du modèle
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Sauvegarde du modèle entraîné
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    print("Modèle entraîné et sauvegardé dans models/model.pkl")

if __name__ == "__main__":
    main()