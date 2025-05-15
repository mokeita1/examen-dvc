import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    # Chemin vers le dataset brut
    raw_data_path = "data/raw_data/raw.csv"
    output_dir = "data/processed/"

    # Création du dossier de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)

    # Chargement du dataset
    df = pd.read_csv(raw_data_path)

    # Séparation X et y (target = dernière colonne)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Sauvegarde des fichiers
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print("Données splitées et sauvegardées dans data/processed/")

if __name__ == "__main__":
    main()