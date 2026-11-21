from pathlib import Path
import pandas as pd


def clean_data(
    input_path: str = "data/raw/diabetes_raw.csv",
    output_path: str = "data/processed/diabetes_processed.csv",
) -> pd.DataFrame:
    """Limpia el dataset (nulos, duplicados y tipos numéricos) y guarda el resultado procesado."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada en: {source}")

    df = pd.read_csv(source)
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.drop_duplicates()

    for col in cleaned_df.columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col])

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(destination, index=False)

    print(f"Datos limpios guardados en: {destination} (shape={cleaned_df.shape})")
    return cleaned_df


if __name__ == "__main__":
    clean_data()
