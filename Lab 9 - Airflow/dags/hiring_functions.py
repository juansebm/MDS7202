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

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(os.path.join(base_path, "splits", "train.csv"), index=False)
    test.to_csv(os.path.join(base_path, "splits", "test.csv"), index=False)
    print("Datos divididos y guardados correctamente")

def preprocess_and_train(**kwargs):
    execution_date = kwargs['ds']
    base_path = f"data/{execution_date}"

    train = pd.read_csv(os.path.join(base_path, "splits", "train.csv"))
    test = pd.read_csv(os.path.join(base_path, "splits", "test.csv"))

    X_train = train.drop(columns=["HiringDecision"])
    y_train = train["HiringDecision"]
    X_test = test.drop(columns=["HiringDecision"])
    y_test = test["HiringDecision"]

    numeric_features = ["Age", "ExperienceYears", "PreviousCompanies", "DistanceFromCompany",
                        "InterviewScore", "SkillScore", "PersonalityScore"]
    categorical_features = ["Gender", "EducationLevel", "RecruitmentStrategy"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"Accuracy: {acc:.4f}, F1-score clase positiva: {f1:.4f}")

    joblib.dump(clf, os.path.join(base_path, "models", "model.joblib"))

def predict(file, model_path):
    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]
    return {"Predicción": labels[0]}

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