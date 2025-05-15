import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import os
import joblib

def main():
    data_dir = "data/processed/"
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)

    # Chargement des données
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()

    # Définition du modèle et des hyperparamètres à tester
    model = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    # Sauvegarde des meilleurs paramètres
    best_params = grid_search.best_params_
    joblib.dump(best_params, os.path.join(model_dir, "best_params.pkl"))

    print("Meilleurs paramètres enregistrés :", best_params)

if __name__ == "__main__":
    main()