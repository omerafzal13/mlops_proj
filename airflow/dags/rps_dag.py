"""
Airflow DAG: rps_pipeline
- Extracts hourly data from OpenWeather
- Runs Data Quality gate (raise error if >1% nulls in key cols)
- Runs featurization
- Produces a ydata-profiling report and logs it to MLflow as artifact
- Versions processed dataset with DVC (assumes dvc is configured inside container)
- Triggers training script
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os, logging, subprocess

default_args = {
    'owner':'rps-team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

RAW_DIR = '/opt/airflow/data/raw'
PROCESSED_DIR = '/opt/airflow/data/processed'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract(**context):
    from src.ingest.fetch_openweather import fetch_and_save_raw
    # city can be parameterized; using Karachi as example
    parsed = fetch_and_save_raw(city='Karachi', out_dir=RAW_DIR)
    return parsed

def quality_gate(**context):
    import pandas as pd
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.parquet')])
    if not files:
        raise FileNotFoundError("No raw parquet found")
    latest = os.path.join(RAW_DIR, files[-1])
    df = pd.read_parquet(latest)
    key_cols = ['dt','temp']
    null_pct = df[key_cols].isnull().mean().max()
    logging.info(f"Null pct in key cols: {null_pct}")
    if null_pct > 0.01:
        raise ValueError(f"Data quality gate failed: null_pct={null_pct}")

def transform(**context):
    from src.transform.featurize import run_featurization
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.parquet')])
    raw = os.path.join(RAW_DIR, files[-1])
    out = os.path.join(PROCESSED_DIR, f"processed_{datetime.utcnow().isoformat(timespec='seconds')}.parquet")
    run_featurization(raw_path=raw, out_path=out, horizon=4)
    return out

def profile_and_version(**context):
    import mlflow, os
    from ydata_profiling import ProfileReport
    import pandas as pd
    processed_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('.parquet')])
    if not processed_files:
        raise FileNotFoundError("No processed dataset")
    proc = os.path.join(PROCESSED_DIR, processed_files[-1])
    df = pd.read_parquet(proc)
    prof = ProfileReport(df, title="RPS Profile", minimal=True)
    html = '/tmp/profile_report.html'
    prof.to_file(html)
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
    with mlflow.start_run(run_name='dataset_profile'):
        mlflow.log_artifact(html, artifact_path='data_profiles')
    # DVC add & push (assumes dvc configured and REMOTE default present)
    os.system(f"dvc add {proc}")
    os.system("git add . && git commit -m 'Add processed dataset' || true")
    os.system("dvc push")

def trigger_train(**context):
    # invoke training script
    os.system("python /opt/airflow/src/models/train.py --config /opt/airflow/config/train.yaml")

with DAG('rps_pipeline', default_args=default_args, schedule_interval='@hourly',
         start_date=datetime(2025,1,1), catchup=False, max_active_runs=1) as dag:

    t1 = PythonOperator(task_id='extract', python_callable=extract)
    t2 = PythonOperator(task_id='quality_gate', python_callable=quality_gate)
    t3 = PythonOperator(task_id='transform', python_callable=transform)
    t4 = PythonOperator(task_id='profile_and_version', python_callable=profile_and_version)
    t5 = PythonOperator(task_id='trigger_train', python_callable=trigger_train)

    t1 >> t2 >> t3 >> t4 >> t5
