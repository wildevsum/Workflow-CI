import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

DATA_PATH = args.data_path

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['is_fit'])
y = df['is_fit']
X = X.select_dtypes(include=[np.number])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow setup
mlruns_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

# **Set experiment di sini**
mlflow.set_experiment("FitnessExperiment")

mlflow.sklearn.autolog()

# Training
with mlflow.start_run(run_name="training"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
