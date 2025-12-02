# src/serving/app.py
from fastapi import FastAPI, Request
import pandas as pd, os, json
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

app = FastAPI(title="RPS Temperature Predictor")
Instrumentator().instrument(app).expose(app)

DRIFT_GAUGE = Gauge('rps_data_drift_ratio', 'Ratio of request features outside training range')
MODEL = None
STATS = None

def load_model():
    global MODEL, STATS
    # load MLflow registered model (Production stage)
    model_uri = os.environ.get('MLFLOW_MODEL_URI', 'models:/rps_temperature_model/Production')
    MODEL = mlflow.pyfunc.load_model(model_uri)
    # try to load stats saved with training run (assume stats.json is at MODEL_DIR/stats.json)
    stats_path = os.environ.get('STATS_PATH', 'model_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as fh:
            STATS = json.load(fh)
    else:
        STATS = {}

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
async def predict(payload: dict):
    # payload is a dict of features
    df = pd.DataFrame([payload])
    # simple drift proxy
    out_of_range_count = 0
    total = 0
    if STATS and 'columns' in STATS:
        for col, bounds in STATS['columns'].items():
            if col in df.columns:
                total += 1
                val = float(df[col].iloc[0])
                if val < bounds['min'] or val > bounds['max']:
                    out_of_range_count += 1
    ratio = float(out_of_range_count) / (total if total>0 else 1)
    DRIFT_GAUGE.set(ratio)
    # predict
    preds = MODEL.predict(df)
    return {"prediction": float(preds[0]), "drift_ratio": ratio}
