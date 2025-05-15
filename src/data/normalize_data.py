import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib

def main():
    input_dir = "data/processed/"
    output_dir = "data/processed/"
    model_dir = "models/"

    os.makedirs(model_dir, exist_ok=True)

    # Chargement des jeux de données
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))

    # Sélection uniquement des colonnes numériques
    X_train_numeric = X_train.select_dtypes(include=["number"])
    X_test_numeric = X_test[X_train_numeric.columns] 

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # Sauvegarde des jeux de données normalisés
    pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns).to_csv(
        os.path.join(output_dir, "X_train_scaled.csv"), index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_train_numeric.columns).to_csv(
        os.path.join(output_dir, "X_test_scaled.csv"), index=False
    )

    # Sauvegarde du scaler
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    print("Données normalisées et sauvegardées.")

if __name__ == "__main__":
    main()