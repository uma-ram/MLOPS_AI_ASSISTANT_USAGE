from prefect import flow, task
import joblib
import mlflow
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import pathlib
import pickle

@task
def load_data(path: str):
    return joblib.load(path)

@task(log_prints=True)
def train_model(X_train, y_train):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ai-assistant-satisfaction_prefect")

    with mlflow.start_run():
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        mlflow.log_params(model.get_params())
        
        #mlflow.sklearn.log_model(model, "model")
       # pathlib.Path("models").mkdir(exist_ok=True)
        #with open("models/preprocessor.b", "wb") as f_out:
        #    pickle.dump(dv, f_out)
        #mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(model, artifact_path="models_mlflow")
    return model

@task(log_prints=True)
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"RMSE: {rmse:.4f}")
    mlflow.log_metric("rmse", rmse)   

    

@flow(name="Train and Log Model")
def train_pipeline():
    X_train, X_test, y_train, y_test = load_data("data/train_test_split.pkl")
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    train_pipeline()
