import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import data_processing

def run_imbalance_fix():
    print("--- Loading Data ---")
    df = data_processing.load_and_clean_data()

    if df is None:
        return

    print("--- Preprocessing (Dropping Leakage) ---")
    X, y = data_processing.preprocess_data(df, drop_leakage=True)

    # Split
    print("--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("--- Applying SMOTE to Training Data ---")
    # SMOTE only on training data to avoid leaking validation info
    # k_neighbors need to be smaller than the number of samples in the smallest class
    # Check class distribution
    print("Class distribution before SMOTE:")
    print(y_train.value_counts())

    # Adjust k_neighbors if very few samples in minority class
    min_class_samples = y_train.value_counts().min()
    k = min(5, min_class_samples - 1)
    if k < 1: k = 1

    smote = SMOTE(random_state=42, k_neighbors=k)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Train Random Forest with Class Weights
    print("--- Training Random Forest (SMOTE + Balanced Weights) ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced', # Add class weights as well
        n_jobs=-1
    )
    rf.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = rf.predict(X_test_scaled)

    # Evaluate
    print("--- Evaluation Results (Leakage Removed + SMOTE + Balanced) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_imbalance_fix()
