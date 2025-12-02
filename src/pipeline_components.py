import os
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from kfp.dsl import component


# -------------------------------------------
# 1️⃣ DATA EXTRACTION COMPONENT
# -------------------------------------------
@component
def extract_data(dvc_remote_url: str, output_path: str):
    """
    Fetch dataset from DVC remote using `dvc get`.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    subprocess.run(
        ["dvc", "get", dvc_remote_url, "data/raw_data.csv", "-o", f"{output_path}/raw_data.csv"],
        check=True
    )
    print("Dataset fetched to:", output_path)


# -------------------------------------------
# 2️⃣ DATA PREPROCESSING COMPONENT
# -------------------------------------------
@component
def preprocess_data(input_csv: str, output_dir: str):
    """
    Cleans, scales and splits the dataset.
    Saves: X_train.csv, X_test.csv, y_train.csv, y_test.csv
    """
    df = pd.read_csv(input_csv)

    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Preprocessing complete. Files saved to:", output_dir)


# -------------------------------------------
# 3️⃣ MODEL TRAINING COMPONENT
# -------------------------------------------
@component
def train_model(preprocessed_dir: str, model_output_path: str):
    """
    Trains a RandomForest model and saves model.pkl.
    """
    X_train = pd.read_csv(f"{preprocessed_dir}/X_train.csv")
    y_train = pd.read_csv(f"{preprocessed_dir}/y_train.csv")

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    joblib.dump(model, model_output_path)
    print("Model saved to:", model_output_path)


# -------------------------------------------
# 4️⃣ MODEL EVALUATION COMPONENT
# -------------------------------------------
@component
def evaluate_model(preprocessed_dir: str, model_path: str, metrics_output: str):
    """
    Loads model, evaluates it, saves metrics.json.
    """
    X_test = pd.read_csv(f"{preprocessed_dir}/X_test.csv")
    y_test = pd.read_csv(f"{preprocessed_dir}/y_test.csv")

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    with open(metrics_output, "w") as f:
        f.write(f"mse: {mse}\n")
        f.write(f"r2: {r2}\n")

    print("Metrics saved:", metrics_output)
