import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# -----------------------------
# Step 1: Configure MLflow backend (SQLite)
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ai-assistant-satisfaction")

# -----------------------------
# Step 2: Load pre-saved train/test data
# -----------------------------
def load_data():
    X_train, X_test, y_train, y_test = joblib.load("data/train_test_split.pkl")
    return X_train, X_test, y_train, y_test

# -----------------------------
# Step 3: Train Model and Log with MLflow
# -----------------------------
def train():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run() as run:
        # Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log params and metrics
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Register the model
        model_name = "ai-assistant-satisfaction-model"
        run_id = run.info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", model_name)

        print(f"✅ Model registered as: {model_name}")
        print(f"🔗 Run ID: {run_id}")

if __name__ == "__main__":
    train()