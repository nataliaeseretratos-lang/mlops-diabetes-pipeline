# MLOps Diabetes Progression Pipeline

Proyecto end-to-end de Machine Learning y MLOps sobre el dataset de **Diabetes** de `scikit-learn` para predecir la progresión de la enfermedad un año después del inicio.

## Estructura del repositorio

```text
├── .github/workflows/ci.yml   # Pipeline de CI en GitHub Actions
├── artifacts/                 # Métricas y artefactos de evaluación
├── data/                    # Datos crudos (raw) y procesados (processed)
├── src/
│   ├── data_extraction.py   # Carga y guarda el dataset crudo
│   ├── data_validation.py   # Valida columnas, tipos y calidad
│   ├── data_cleaning.py     # Limpia nulos/duplicados y exporta CSV limpio
│   ├── train.py             # Entrena GradientBoosting, registra en MLflow y evalúa quality gate
│   └── api.py               # API REST con FastAPI sirviendo el modelo champion
├── Proyecto_MLOps.ipynb     # Cuaderno de experimentación y pruebas
├── requirements.txt         # Dependencias del proyecto
└── README.md
```

## Pipeline y servicio

- **Extracción, validación y limpieza**: scripts modulares en `src/` que generan `data/raw/diabetes_raw.csv` y `data/processed/diabetes_processed.csv`.
- **Entrenamiento y MLflow**: `GradientBoostingRegressor` registrado en `sqlite:///mlflow.db` (`diabetes_progression_regressor`); si `RMSE < 60.0`, se asigna el alias `champion`.
- **FastAPI (`src/api.py`)**: expone `GET /health`, `GET /model-info` y `POST /predict` con validación Pydantic.
- **CI (`GitHub Actions`)**: `.github/workflows/ci.yml` ejecuta el pipeline, el entrenamiento con *quality gate* y una comprobación de la API en `push` y `pull_request`.
