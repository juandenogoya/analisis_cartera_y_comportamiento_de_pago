import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import data_processing

def run_benchmark():
    print("--- Loading Data ---")
    df = data_processing.load_and_clean_data()
    if df is None: return

    # --- Feature Engineering (Same as Stage 3) ---
    income_map = {'A': 500000, 'B': 300000, 'C1': 200000, 'C2': 150000, 'C3': 100000, 'D1': 50000, 'D2': 25000}
    if 'VL_INGRESO_NOSIS' in df.columns:
        df['EST_INCOME'] = df['VL_INGRESO_NOSIS'].map(income_map).fillna(25000)
        df['RATIO_AMOUNT_INCOME'] = df['VL_DESEMBOLSADO'] / (df['EST_INCOME'] + 1)
        df['RATIO_INSTALLMENT_INCOME'] = df['VL_CUOTA'] / (df['EST_INCOME'] + 1)

    if 'NU_EDAD' in df.columns:
        df['AGE_GROUP'] = pd.cut(df['NU_EDAD'], bins=[0, 25, 40, 60, 100], labels=['Young', 'Adult', 'Senior', 'Elderly'])
        df['AGE_GROUP'] = df['AGE_GROUP'].astype(object)

    print("--- Preprocessing ---")
    X, y = data_processing.preprocess_data(df, drop_leakage=True)

    # Label Encode Target for XGB/LGBM
    y_codes = pd.Categorical(y).codes
    target_map = dict(enumerate(pd.Categorical(y).categories))

    X_train, X_test, y_train, y_test = train_test_split(X, y_codes, test_size=0.25, random_state=42, stratify=y_codes)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE
    min_class_samples = pd.Series(y_train).value_counts().min()
    k = min(5, min_class_samples - 1)
    if k < 1: k = 1
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1)
    }

    results = []

    print("\n--- Model Benchmark ---")
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({'Model': name, 'Accuracy': acc, 'F1-Score (Weighted)': f1})
            print(f"{name} -> Acc: {acc:.4f}, F1: {f1:.4f}")
        except Exception as e:
            print(f"Failed to train {name}: {e}")

    print("\n--- Summary ---")
    results_df = pd.DataFrame(results).sort_values(by='F1-Score (Weighted)', ascending=False)
    print(results_df)

    return results_df

if __name__ == "__main__":
    run_benchmark()
