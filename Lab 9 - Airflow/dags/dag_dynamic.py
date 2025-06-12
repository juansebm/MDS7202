from airflow import DAG
from airflow.operators.python_operator import BranchPythonOperator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from airflow.utils.trigger_rule import TriggerRule


from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from hiring_dynamic_functions import (
    create_folders,
    split_data,
    train_model,
    load_and_merge,
    evaluate_models
)

default_args = {"owner": "airflow"}

def decide_branch(ds, **kwargs):
    """Esta función decide qué datos descargar según la fecha de ejecución, si
    antes o después del primero de Noviembre de 2024"""
    cutoff_date = datetime(2024, 11, 1)
    current_date = datetime.strptime(ds, "%Y-%m-%d")
    if current_date < cutoff_date:
        return "download_data_1"
    else:
        return ["download_data_1", "download_data_2"]

    #Ahora dividimos el enunciado de este script por puntos. Cada punto es una parte del DAG;

    
    #punto 1: Inicializamos el DAG con fecha de inicio el 1 de Octubre de 2024, el cual se debe ejecutar
    #el día 5 de cada mes a las 15:00 UTC, utilizando un dag_id interpretable para identificar fácilmente
    #habilitamos el backfill para que pueda ejecutar tareas programadas desde fechas pasadas.
    
with DAG(
    dag_id="hiring_dynamic",#cambiamos de hiring_lineal a hiring_dynamic
    default_args=default_args,
    start_date=datetime(2024, 10, 1),
    schedule_interval="0 15 5 * *",#cambiamos None por las 15 UTC, osea 0 15 5 * * 
    catchup=True,#aqui activamos backfilling para ejecutar tareas programadas desde fechas pasadas
    description="DAG para predicción dinámica de contrataciones",
) as dag:

    
    #punto 2; aquí definimos el marcador de posición que indica el inicio del pipeline
    start = EmptyOperator(task_id="start")

    #punto 3; creamos carpetas para la ejecución actual
    crear_carpetas = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,#esta es una función importada de hiring_dynamic_functions.py
        op_kwargs={"ds": "{{ ds }}"},
    )

    #punto 4; hacemos branching (es decir, "rama"-izado) para decidir qué datos descargar,
    #usando la función decide_branch que definimos antes.
    branching = BranchPythonOperator(
        task_id="branching_download",
        python_callable=decide_branch,
        op_kwargs={"ds": "{{ ds }}"},
    )

    #descargamos o bien data_1.csv o bien data_2.csv! Este punto es nuevo.
    #para data_1.csv (siempre)
    download_data_1 = BashOperator(
        task_id="download_data_1",
        bash_command=(
            "mkdir -p $AIRFLOW_HOME/data/{{ ds }}/raw && "
            "curl -sSf -o $AIRFLOW_HOME/data/{{ ds }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        ),
        env={"AIRFLOW_HOME": "/opt/airflow"},
    )

    #para data_2.csv (solo desde noviembre)
    download_data_2 = BashOperator(
        task_id="download_data_2",
        bash_command=(
            "mkdir -p $AIRFLOW_HOME/data/{{ ds }}/raw && "
            "curl -sSf -o $AIRFLOW_HOME/data/{{ ds }}/raw/data_2.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
        ),
        env={"AIRFLOW_HOME": "/opt/airflow"},
    )


    #punto 5: Concatenamos los datasets disponibles usando la función
    #load_and_merge(), configurando un trigger para que la tarea se eje
    #cute si encuentra disponible como mínimo uno de los archivos.
    load_merge = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,#usamos la función de hiring_dynamic_functions.py
        op_kwargs={"ds": "{{ ds }}"},
        
        # Ejecutamos este operador si al menos uno de
        #los archivos existe (usamos TriggerRule como ONE_SUCCESS)
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    #punto 6; Aplicamos el holdout y split de los datos
    #en entrenamiento y prueba usando la función split_data()
    split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,#usamos la función de hiring_dynamic_functions.py
        op_kwargs={"ds": "{{ ds }}"},
    )

    
    #punto 7; Entrenamientos paralelos; realizamos 3 entrenamientos
    #paralelos de 3 modelos diferentes, donde nos aseguramos de guardar
    #los modelos entrenados con nombres distintivos, usando la función
    #train_model().

    train_rf = PythonOperator(
        task_id="train_rf",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model": RandomForestClassifier(),
            "model_name": "rf"
        },
    )

    train_xgb = PythonOperator(
        task_id="train_xgb",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            "model_name": "xgb"
        },
    )

    train_lgbm = PythonOperator(
        task_id="train_lgbm",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model": LGBMClassifier(),
            "model_name": "lgbm"
        },
    )



    #punto 8;
    # Evaluación de modelos: registramos el accuracy de cada modelo en el set
    # de prueba, luego imprimimos el mejor modelo seleccionado para que la
    # tarea se ejecute solamente si los 3 modelos fueron entrenados y guardados
    evaluate = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_kwargs={"ds": "{{ ds }}"},
        # Ejecutar solo si los 3 modelos fueron entrenados (todos OK)
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    
    start >> crear_carpetas >> branching
    branching >> [download_data_1, download_data_2] >> load_merge
    load_merge >> split
    split >> [train_rf, train_xgb, train_lgbm] >> evaluate

