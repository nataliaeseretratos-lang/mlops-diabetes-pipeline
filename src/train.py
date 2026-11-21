import json
from pathlib import Path
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_model(
    data_path: str = "data/processed/diabetes_processed.csv",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "diabetes_regression_mlops",
    model_name: str = "diabetes_progression_regressor",
    rmse_threshold: float = 60.0,
) -> float:
    """Entrena un GradientBoostingRegressor, registra en MLflow y asigna alias 'champion' si pasa el umbral."""
    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset procesado en: {csv_path}")

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    params = {
        "model_type": "GradientBoostingRegressor",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
    }

    with mlflow.start_run(run_name="gb_baseline_run"):
        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_params(params)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        metrics_file = artifacts_dir / "evaluation_metrics.json"
        metrics_file.write_text(
            json.dumps(
                {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "rmse_threshold": rmse_threshold,
                },
                indent=2,
            )
        )
        mlflow.log_artifact(str(metrics_file))

        registered_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
            serialization_format="pickle",
        )

        print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")

        if rmse < rmse_threshold:
            client = MlflowClient()
            version = registered_model.registered_model_version
            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=str(version),
            )
            print(
                f"¡Pasa el quality gate ({rmse:.2f} < {rmse_threshold})! "
                f"Alias 'champion' asignado a la versión {version}."
            )
        else:
            print(
                f"No pasa el quality gate ({rmse:.2f} >= {rmse_threshold}). "
                "No se asigna alias 'champion'."
            )

    return rmse


if __name__ == "__main__":
    train_model()
