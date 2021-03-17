from datetime import timedelta
from airflow.utils.dates import days_ago
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

RUN_COMMAND = 'python /usr/local/airflow/dags/crypto_predictions.py'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'crypto_scheduler',
    default_args=default_args,
    description='Scheduler for crypto',
    schedule_interval=timedelta(hours=3),
    start_date=days_ago(2),
    tags=['crypto'],
    catchup=False
)

t1 = BashOperator(
    task_id='run_script',
    bash_command=RUN_COMMAND,
    dag=dag,
)