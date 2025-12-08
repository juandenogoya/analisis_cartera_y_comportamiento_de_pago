import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import data_processing

def run_baseline():
    print("--- Loading Data ---")
    df = data_processing.load_and_clean_data()

    if df is None:
        return

    print(f"Data Loaded. Shape: {df.shape}")

    print("--- Preprocessing (Dropping Leakage) ---")
    X, y = data_processing.preprocess_data(df, drop_leakage=True)

    print(f"Features Shape: {X.shape}")
    print("Features used:", list(X.columns))

    # Split
    print("--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Scaling (Random Forest doesn't strictly need it, but good for consistency)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("--- Training Random Forest (Baseline) ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = rf.predict(X_test_scaled)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("--- Evaluation Results (Leakage Removed) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, report

if __name__ == "__main__":
    run_baseline()
