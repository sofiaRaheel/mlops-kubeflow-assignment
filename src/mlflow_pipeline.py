# src/mlflow_pipeline.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    # Use the correct target column from your CSV
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

def run_pipeline():
    mlflow.set_experiment("mlops_assignment_experiment")
    with mlflow.start_run(run_name="mlflow_pipeline_run"):
        df = load_data("data/raw_data.csv")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_model(X_train, y_train)
        mse = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

        print(f"Run completed. MSE: {mse}")

if __name__ == "__main__":
    run_pipeline()
