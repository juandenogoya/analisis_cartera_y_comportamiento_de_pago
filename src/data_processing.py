import pandas as pd
import numpy as np

def load_and_clean_data(filepath='evaluaciones_2022.csv'):
    """
    Loads data and performs initial cleaning steps defined in the original notebook.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

    # Original cleaning steps from notebook
    # 1. Filter out 'CANCELADO'
    if 'CD_SUBESTADO_PRODUCTO' in df.columns:
        df = df[df['CD_SUBESTADO_PRODUCTO'] != 'CANCELADO']

    # 2. Filter invalid income
    if 'VL_INGRESO_NOSIS' in df.columns:
        df = df[df['VL_INGRESO_NOSIS'] != '0']
        df = df[df['VL_INGRESO_NOSIS'] != 'NC']

    # 3. Filter other invalid states
    if 'CD_SUBESTADO_PRODUCTO' in df.columns:
        df = df[~df['CD_SUBESTADO_PRODUCTO'].isin(['ANULADO', 'EXTRACONTABLE', 'NO LIQUIDADO'])]

    # 4. Filter zeros in Sexo and Canal
    if 'TX_SEXO' in df.columns:
        df = df[df['TX_SEXO'] != '0']
    if 'TX_CANAL_ORIGINAL' in df.columns:
        df = df[df['TX_CANAL_ORIGINAL'] != '0']

    df = df.reset_index(drop=True)
    return df

def preprocess_data(df, drop_leakage=True):
    """
    Preprocesses data: One-Hot Encoding, Drop Leakage, Split X/y.
    """
    if df is None:
        return None, None

    # Target
    target_col = 'CD_SUBESTADO_PRODUCTO'

    # Columns to drop (IDs or irrelevant)
    cols_to_drop = ['FECHA_EVALUACION'] # Keep basic info, drop dates if not used as time series

    # Leakage Columns
    leakage_cols = [
        'CD_CAJON_MORA',
        'NU_CUOTA_PAGADAS',
        'NU_CUOTAS_EN_MORA',
        'NU_DIAS_MORA'
    ]

    if drop_leakage:
        print(f"Dropping leakage columns: {leakage_cols}")
        cols_to_drop.extend(leakage_cols)

    # Drop columns that exist in dataframe
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Separate X and y
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    # One-Hot Encoding for categorical variables
    # Using pd.get_dummies for simplicity in this baseline script
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Handle NaN if any (simple imputation)
    # Convert any categorical column remainders to object before fillna to avoid TypeError
    for col in X_encoded.columns:
        if isinstance(X_encoded[col].dtype, pd.CategoricalDtype):
             X_encoded[col] = X_encoded[col].astype(object)

    X_encoded = X_encoded.fillna(0) # Assuming 0 for now, or median

    return X_encoded, y
