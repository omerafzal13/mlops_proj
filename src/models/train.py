# src/models/train.py
import os, argparse, json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow, mlflow.sklearn

def load_data(path):
    return pd.read_parquet(path)

def time_split(df, val_hours=48, test_hours=48):
    # assumes df sorted by dt
    n = len(df)
    test_sz = test_hours
    val_sz = val_hours
    train = df.iloc[:(n - val_sz - test_sz)]
    val = df.iloc[(n - val_sz - test_sz):(n - test_sz)]
    test = df.iloc[(n - test_sz):]
    return train, val, test

def prepare_xy(df, target_col):
    X = df.drop(columns=[target_col, 'dt', 'fetched_at'], errors='ignore')
    y = df[target_col]
    return X.fillna(method='ffill').fillna(0), y

def main(args):
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    processed_path = args.processed_path
    df = load_data(processed_path)
    df = df.sort_values('dt').reset_index(drop=True)
    train, val, test = time_split(df)
    target = f"temp_t+{args.horizon}"
    X_train, y_train = prepare_xy(train, target)
    X_val, y_val = prepare_xy(val, target)

    params = {'n_estimators': args.n_estimators, 'max_depth': args.max_depth, 'random_state': 42}
    with mlflow.start_run():
        mlflow.log_params(params)
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        mlflow.log_metric('rmse', rmse)
        # log model
        mlflow.sklearn.log_model(model, artifact_path='model')
        print(f"Validation RMSE: {rmse}")
        # register (optional)
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        try:
            mlflow.register_model(model_uri, "rps_temperature_model")
        except Exception as e:
            print("Model registration might fail if server not configured:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_path", type=str, default="data/processed/processed_test.parquet")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    main(args)
