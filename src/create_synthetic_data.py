import pandas as pd
import numpy as np
import random

def create_synthetic_data(filename='evaluaciones_2022.csv', n_rows=2000):
    np.random.seed(42)
    random.seed(42)

    data = {
        'FECHA_EVALUACION': np.random.randint(20220101, 20221231, n_rows),
        'TX_SEXO': np.random.choice(['MASCULINO', 'FEMENINO', '0'], n_rows, p=[0.48, 0.48, 0.04]),
        'NU_EDAD': np.random.randint(18, 90, n_rows),
        'TX_PROVINCIA': np.random.choice(['BUENOS AIRES', 'CAPITAL FEDERAL', 'CORDOBA', 'SANTA FE'], n_rows),
        'TX_TIPO_CLIENTE': np.random.choice(['NUEVO', 'EXISTENTE', 'CONOCIDO'], n_rows),
        'TX_ESTADO_EVALUACION_DWH': np.random.choice(['APROBADA', 'RECHAZADA'], n_rows, p=[0.7, 0.3]),
        'TX_CANAL_ORIGINAL': np.random.choice(['SUCURSAL', 'API', 'WHATSAPP', 'WEB', '0'], n_rows, p=[0.5, 0.2, 0.2, 0.05, 0.05]),
        'VL_INGRESO_NOSIS': np.random.choice(['A', 'B', 'C1', 'C2', 'C3', 'D1', 'D2', '0', 'NC'], n_rows),
        'VL_SCORE_NOSIS': np.random.uniform(0, 1000, n_rows),
        'VL_DESEMBOLSADO': np.random.uniform(10000, 500000, n_rows),
        'VL_CUOTA': np.random.uniform(1000, 50000, n_rows),
        'VL_TASA': np.random.uniform(30, 150, n_rows),

        # Leakage Columns (Simulated to correlate with target)
        'CD_CAJON_MORA': np.random.choice([0, 1, 2, 3, 4, 5], n_rows, p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05]),
        'NU_CUOTA_PAGADAS': np.random.randint(0, 36, n_rows),
        'NU_CUOTAS_EN_MORA': np.random.randint(0, 12, n_rows),
        'NU_DIAS_MORA': np.random.randint(0, 365, n_rows),
    }

    # Generate Target based on some logic (to make it learnable)
    # If delay cols are high, target is bad
    target = []
    for i in range(n_rows):
        if data['CD_CAJON_MORA'][i] == 0:
            target.append('AL DIA')
        elif data['CD_CAJON_MORA'][i] <= 3:
            target.append('MORA < 90')
        else:
            r = random.random()
            if r < 0.6: target.append('MORA > 90')
            elif r < 0.8: target.append('PROBLEMAS')
            else: target.append('WRITE OFF')

    # Introduce some noise and other categories
    for i in range(n_rows):
        if random.random() < 0.05:
            target[i] = 'CANCELADO' # To be filtered out
        elif random.random() < 0.02:
            target[i] = 'ANULADO' # To be filtered out

    data['CD_SUBESTADO_PRODUCTO'] = target

    df = pd.DataFrame(data)

    # Adjust logic for leakage correlation
    # If target is AL DIA, leakage variables should be 0/low
    mask_aldia = df['CD_SUBESTADO_PRODUCTO'] == 'AL DIA'
    df.loc[mask_aldia, 'NU_CUOTAS_EN_MORA'] = 0
    df.loc[mask_aldia, 'NU_DIAS_MORA'] = 0

    print(f"Generated synthetic data with {n_rows} rows.")
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    create_synthetic_data()
