from airflow import DAG
from airflow.operators.empty import EmptyOperator
from datetime import datetime

with DAG(
    dag_id="test_simple",
    start_date=datetime(2025,6,1),
    schedule_interval=None,
    catchup=False,
) as dag:
    EmptyOperator(task_id="dummy")
