from pathlib import Path
import pandas as pd
from sklearn.datasets import load_diabetes


def extract_data(output_path: str = "data/raw/diabetes_raw.csv") -> pd.DataFrame:
    """Extrae el dataset de Diabetes de Scikit-learn y lo guarda en CSV."""
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame.copy()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)

    print(f"Datos crudos guardados en: {destination} (shape={df.shape})")
    return df


if __name__ == "__main__":
    extract_data()
