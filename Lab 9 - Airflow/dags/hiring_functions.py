import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
from datetime import datetime

def create_folders(**kwargs):
    execution_date = kwargs['ds'] 
    base_path = f"data/{execution_date}"
    os.makedirs(os.path.join(base_path, "raw"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "splits"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    print(f"Carpetas creadas en: {base_path}")

def split_data(**kwargs):
    execution_date = kwargs['ds']
    base_path = f"data/{execution_date}"
    data = pd.read_csv(os.path.join(base_path, "raw", "data_1.csv"))
    X = data.drop(columns=["HiringDecision"])
    y = data["HiringDecision"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Distribución en TRAIN:")
    print(y_train.value_counts(normalize=True))  # proporciones
    print("Distribución en TEST:")
    print(y_test.value_counts(normalize=True))

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(os.path.join(base_path, "splits", "train.csv"), index=False)
    test.to_csv(os.path.join(base_path, "splits", "test.csv"), index=False)
    print("Datos divididos y guardados correctamente")

def preprocess_and_train(**kwargs):
    import os
    import pandas as pd
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    execution_date = kwargs['ds']
    base_path = f"data/{execution_date}"

    numeric_features = [
        "Age", "ExperienceYears", "PreviousCompanies",
        "DistanceFromCompany", "InterviewScore",
        "SkillScore", "PersonalityScore"
    ] # features

    categorical_features = ["Gender", "EducationLevel", "RecruitmentStrategy"] # features

    # train/test splits
    train = pd.read_csv(os.path.join(base_path, "splits", "train.csv"))
    test  = pd.read_csv(os.path.join(base_path, "splits", "test.csv"))

    X_train = train.drop(columns=["HiringDecision"])
    y_train = train["HiringDecision"]
    X_test  = test.drop(columns=["HiringDecision"])
    y_test  = test["HiringDecision"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    base_clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ])

    param_dist = {
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5],
        "classifier__max_features": ["sqrt", "log2", 0.5],
    }

    search = RandomizedSearchCV(
        base_clf,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_clf = search.best_estimator_
    print("Mejores parámetros:", search.best_params_)

    calibrated = CalibratedClassifierCV(
        estimator=best_clf,
        method="isotonic",
        cv=3
    )
    calibrated.fit(X_train, y_train)

    y_pred = calibrated.predict(X_test)
    proba  = calibrated.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, pos_label=1)
    print(f"Accuracy: {acc:.4f}, F1-score clase positiva: {f1:.4f}")
    print(f"Probabilidad media clase Contratado en test: {proba.mean():.4f}")

    try: # Top 10 features
        importances = best_clf.named_steps['classifier'].feature_importances_
        feat_names = (
            numeric_features +
            list(best_clf.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .get_feature_names_out(categorical_features))
        )
        top_idx = np.argsort(importances)[::-1][:10]
        print("Top 10 features:")
        for i in top_idx:
            print(f"  {feat_names[i]}: {importances[i]:.3f}")
    except Exception:
        pass

    # 8) Save the calibrated model
    model_dir = os.path.join(base_path, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(calibrated, os.path.join(model_dir, "model.joblib"))
    print("Modelo entrenado y guardado en:", os.path.join(model_dir, "model.joblib"))

def predict(file, model_path):
    import joblib
    import pandas as pd

    pipeline = joblib.load(model_path)

    path = getattr(file, "name", file)
    df = pd.read_json(path)
    proba = pipeline.predict_proba(df)[0, 1] # Calcula la probabilidad de la clase “Contratado”
    threshold = 0.4
    pred = 1 if proba >= threshold else 0 # Decide en base al threshold

    labels = ["No contratado", "Contratado"]
    return {
        "Predicción": labels[pred],
    }

def gradio_interface():
    import gradio as gr
    today = datetime.today().strftime('%Y-%m-%d')
    model_path = f"data/{today}/models/model.joblib"

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)
