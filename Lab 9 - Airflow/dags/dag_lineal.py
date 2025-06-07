from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from hiring_functions import (
    create_folders,
    split_data,
    preprocess_and_train,
    gradio_interface,
)

default_args = {"owner": "airflow"}

with DAG(
    dag_id="hiring_lineal",
    default_args=default_args,
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,
    catchup=False,
    description="DAG para predicción de contrataciones",
) as dag:

    start = EmptyOperator(task_id="start")

    crear_carpetas = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"ds": "{{ ds }}"},
    )

    descargar_datos = BashOperator(
        task_id="download_data",
        bash_command=(
            "curl -o data/{{ ds }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        ),
    )

    split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"ds": "{{ ds }}"},
    )

    entrenar = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        op_kwargs={"ds": "{{ ds }}"},
    )

    gradio = PythonOperator(
        task_id="gradio_interface",
        python_callable=gradio_interface,
    )

    start >> crear_carpetas >> descargar_datos >> split >> entrenar >> gradio
