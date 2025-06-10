import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from datetime import datetime


def create_folders(**kwargs):
    """Esta función tiene como finalidad crear las carpetas raw, preprocessed,
    splits y models.
    Nota: usamos os.makedirs(..., exist_ok=True) para evitar errores si las
    carpetas ya existen.
    
        Input:
                **kwargs: permite extraer ds (execution date) de Airflow
        Output:
                Folders."""
    execution_date = kwargs['ds']
    base_path = f"data/{execution_date}"
    os.makedirs(os.path.join(base_path, "raw"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "preprocessed"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "splits"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    print(f"Carpetas creadas en: {base_path}")


def load_and_merge(**kwargs):
    """Lee desde la carpeta raw los archivos data_1.csv y data_2.csv en caso
    de estar disponible. Luego concatena estos y genera un nuevo archivo resultante,
    guardándolo en la carpeta preprocessed.
    
        Input:
                **kwargs
        Output:
                merged_df"""
    execution_date = kwargs['ds']
    base_path = f"data/{execution_date}/raw"
    files = ["data_1.csv", "data_2.csv"]

    #el siguiente loop lo que hace es efectivamente concatenar los dataframes
    dfs = []
    for file in files:
        file_path = os.path.join(base_path, file)#esta linea construye la ruta completa
                                                 #del archivo combinando la ruta base (base_path)
                                                 #y el nombre del archivo
        
        if os.path.exists(file_path):# Verifica si el archivo existe en esa ruta
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Archivo leído: {file}")
        else:
            print(f"Archivo no encontrado (omitido): {file}")

    if not dfs:
        raise FileNotFoundError("No se encontraron archivos CSV para procesar.")

    #ignore_index=True reinicia el índice del dataframe resultante, lo que es útil
    #si queremos concatenar datos similares y no nos importa el índice original.
    merged_df = pd.concat(dfs, ignore_index=True)
    output_path = f"data/{execution_date}/preprocessed/merged.csv"

    #index=False evita que se escriba la columna de índice en el archivo CSV. Esto es
    #lo habitual cuando se exportan datos para análisis posteriores, ya que el índice no
    #suele ser útil fuera de Pandas.
    merged_df.to_csv(output_path, index=False)
    print(f"Datos preprocesados guardados en: {output_path}")


def split_data(**kwargs):
    """Función que lee la data guardada en la carpeta preprocessed y realiza un holdout
    sobre esta data. Esta crea un set de train y de test, manteniendo una semilla
    y 20% para el conjunto de test. Además, guarda los conjuntos resultantes en la carpeta
    splits.
    
        Input:
                **kwargs
        Output:
                train.csv
                test.csv"""
    execution_date = kwargs['ds']
    input_path = f"data/{execution_date}/preprocessed/merged.csv"
    output_path = f"data/{execution_date}/splits"

    data = pd.read_csv(input_path)
    X = data.drop(columns=["HiringDecision"])
    y = data["HiringDecision"]

    #hacemos el split tal como nos dice el enunciado: manteniendo proporciones con stratify
    #fijando la semilla con random_state y que el tamaño del test set fuese de 20% o 0.2.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test.to_csv(os.path.join(output_path, "test.csv"), index=False)
    print(f"Datos divididos y guardados en: {output_path}")


def train_model(model, **kwargs):
    """Esta función hace cuatro cosas: 1) comienza leyendo el conjunto de training
    desde la carpeta splits, 2) crea y aplica un pipeline con una etapa de preproce-
    samiento utilizando columntransformer para las transformaciones, 3) añade una etapa
    de entrenamiento utilizando un modelo que se ingresa a la función y 4) crea un
    archivo joblib con el pipeline entrenado, guardando el modelo con un nombre que
    permita una fácil identificación dentro de la carpeta models.
    
        Input:
               **kwargs
               model: modelo ingresado
        Output:
               archivo .joblib: con el pipeline entrenado"""
    execution_date = kwargs['ds']
    split_path = f"data/{execution_date}/splits"
    model_path = f"data/{execution_date}/models"

    #etapa 1: lectura del training set
    train = pd.read_csv(os.path.join(split_path, "train.csv"))
    X_train = train.drop(columns=["HiringDecision"])
    y_train = train["HiringDecision"]

    #etapa 2: pipeline y ColumnTransformer
    numeric_features = [
        "Age", "ExperienceYears", "PreviousCompanies",
        "DistanceFromCompany", "InterviewScore",
        "SkillScore", "PersonalityScore"
    ]
    categorical_features = ["Gender", "EducationLevel", "RecruitmentStrategy"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    #etapa 3: entrenamiento
    pipeline.fit(X_train, y_train)

    #etapa 4: se crea un archivo joblib.
    model_filename = f"{model.__class__.__name__}_{datetime.now().strftime('%H%M%S')}.joblib"
    full_model_path = os.path.join(model_path, model_filename)
    joblib.dump(pipeline, full_model_path)
    print(f"Modelo entrenado y guardado en: {full_model_path}")


def evaluate_models(**kwargs):
    """Función que recibe los modelos entrenados desde la carpeta models, evalúa
    su desempeño mediante accuracy en el test set y selecciona el mejor modelo obtenido.
    Además, guarda el mejor modelo como un archivo .joblib. Su función debe imprimir el
    nombre del modelo seleccionado y el accuracy obtenido.
    
        Input:
              **kwargs
        Output:
              best_model.joblib: el modelo con el mejor accuracy en el test set."""
    execution_date = kwargs['ds']
    base_path = f"data/{execution_date}"
    models_dir = os.path.join(base_path, "models")
    test_path = os.path.join(base_path, "splits", "test.csv")

    #recordemos que ahora queremos evaluar la función en el test set.
    test = pd.read_csv(test_path)
    X_test = test.drop(columns=["HiringDecision"])
    y_test = test["HiringDecision"]

    best_accuracy = -1
    best_model_name = None
    best_model = None

    for model_file in os.listdir(models_dir):
        if model_file.endswith(".joblib"):
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            print(f"Modelo {model_file} → Accuracy: {acc:.4f}")

            #vamos actualizando el modelo, el nombre y la métrica accuracy
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_model_name = model_file

    #vamos guardando el mejor modelo (o best_model)
    if best_model is not None:
        final_model_path = os.path.join(models_dir, "best_model.joblib")
        joblib.dump(best_model, final_model_path)
        print(f"Mejor modelo: {best_model_name} con Accuracy: {best_accuracy:.4f}")
        print(f"Guardado como: {final_model_path}")
    else:
        print("No se encontraron modelos para evaluar.")
