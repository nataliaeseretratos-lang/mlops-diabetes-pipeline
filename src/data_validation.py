from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "target",
]


def validate_data(input_path: str = "data/raw/diabetes_raw.csv") -> pd.DataFrame:
    """Valida la existencia, estructura, columnas y calidad básica del dataset crudo."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("El dataset está vacío.")

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

    numeric_df = df[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric_df.isna().any().any():
        raise ValueError("Existen valores no numéricos o nulos tras convertir a numérico.")

    if (numeric_df["target"] <= 0).any():
        raise ValueError("Se encontraron valores de 'target' menores o iguales a 0.")

    print(f"Validación correcta: {len(df)} filas y {len(EXPECTED_COLUMNS)} columnas verificadas.")
    return df


if __name__ == "__main__":
    validate_data()
