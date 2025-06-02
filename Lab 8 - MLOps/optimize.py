import os
import pickle
from datetime import datetime
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    best_model_id = runs.sort_values("metrics.f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.xgboost.load_model(f"runs:/{best_model_id}/model")
    return best_model


def optimize_model():

    # Datos                                                           
    df = pd.read_csv("./water_potability.csv")
    df = df.dropna()

    X = df.drop(columns=["Potability"]).values
    y = df["Potability"].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=191919,
        stratify=y,
    )

    # Configurar MLflow
    experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    mlflow.xgboost.autolog(log_models=True)

    def objective(trial):
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        }

        mlflow.start_run(run_name=f"trial_{trial.number}")
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        f1 = f1_score(y_valid, preds)

        mlflow.log_metric("f1", f1)
        mlflow.end_run()
        return f1

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=5, show_progress_bar=True)

    best_trial = study.best_trial
    print(f"Mejor f1: {best_trial.value:.4f}")
    print(f"Param: {best_trial.params}")

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    best_model = get_best_model(experiment_id)

    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    with mlflow.start_run(experiment_id=experiment_id, run_name="Best_Model", nested=True):
        mlflow.log_artifact(model_path, artifact_path="models")

        try:
            importances = best_model.get_booster().get_score(importance_type="gain")

            sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            names = [k for k, _ in sorted_items][:15]
            scores = [v for _, v in sorted_items][:15]

            plt.figure(figsize=(8, 6))
            plt.barh(range(len(names)), scores)
            plt.yticks(range(len(names)), names)
            plt.gca().invert_yaxis()
            plt.title("XGB Feature Importance (gain)")
            plt.tight_layout()

            os.makedirs("plots", exist_ok=True)
            fi_path = "plots/feature_importance.png"
            plt.savefig(fi_path)
            plt.close()
            
            mlflow.log_artifact(fi_path, artifact_path="plots")
        except Exception as e:
            print(f"{e}")

    return best_model

if __name__ == "__main__":
    optimize_model()
