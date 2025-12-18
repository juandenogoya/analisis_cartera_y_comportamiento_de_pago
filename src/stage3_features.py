import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import data_processing
import matplotlib.pyplot as plt

def run_feature_engineering():
    print("--- Loading Data ---")
    df = data_processing.load_and_clean_data()

    if df is None: return

    # --- Feature Engineering ---
    # 1. Loan to Income Ratio (Approximation)
    # Income is categorical (A, B, C1...), we need to map it to numeric to be useful as a ratio
    # This mapping is an estimation based on typical credit scoring bands, actual values would be better
    income_map = {
        'A': 500000, 'B': 300000, 'C1': 200000, 'C2': 150000, 'C3': 100000, 'D1': 50000, 'D2': 25000
    }
    # Ensure VL_INGRESO_NOSIS exists
    if 'VL_INGRESO_NOSIS' in df.columns:
        df['EST_INCOME'] = df['VL_INGRESO_NOSIS'].map(income_map).fillna(25000) # Default to lowest if missing

        # Loan Amount / Income
        df['RATIO_AMOUNT_INCOME'] = df['VL_DESEMBOLSADO'] / (df['EST_INCOME'] + 1)

        # Installment / Income
        df['RATIO_INSTALLMENT_INCOME'] = df['VL_CUOTA'] / (df['EST_INCOME'] + 1)

    # 2. Age Binning
    if 'NU_EDAD' in df.columns:
        # Cast to object to ensure get_dummies picks it up as categorical
        df['AGE_GROUP'] = pd.cut(df['NU_EDAD'], bins=[0, 25, 40, 60, 100], labels=['Young', 'Adult', 'Senior', 'Elderly'])
        df['AGE_GROUP'] = df['AGE_GROUP'].astype(object)

    print("Added Features: RATIO_AMOUNT_INCOME, RATIO_INSTALLMENT_INCOME, AGE_GROUP")

    print("--- Preprocessing ---")
    X, y = data_processing.preprocess_data(df, drop_leakage=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE
    min_class_samples = y_train.value_counts().min()
    k = min(5, min_class_samples - 1)
    if k < 1: k = 1
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Train
    print("--- Training RF with New Features ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train_resampled, y_train_resampled)

    # Feature Importance
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

    print("\nTop 10 Feature Importances:")
    print(feature_imp_df.head(10))

    # Predict
    y_pred = rf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return feature_imp_df, accuracy, report

if __name__ == "__main__":
    run_feature_engineering()
