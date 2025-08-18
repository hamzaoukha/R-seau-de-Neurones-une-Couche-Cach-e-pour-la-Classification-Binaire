import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLS_ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_and_prepare(csv_path, seed=42, test_size=0.20, val_size=0.25):
    df = pd.read_csv(csv_path)

    df[COLS_ZERO_AS_NAN] = df[COLS_ZERO_AS_NAN].replace(0, np.nan)
    df[COLS_ZERO_AS_NAN] = df[COLS_ZERO_AS_NAN].fillna(df[COLS_ZERO_AS_NAN].median())

    X = df.drop("Outcome", axis=1).astype(float).values
    y = df["Outcome"].astype(int).values.reshape(-1, 1)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y.ravel(), random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, stratify=y_tmp.ravel(), random_state=seed
    )

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    def _std(Z): return (Z - mu) / sigma

    return _std(X_train), y_train, _std(X_val), y_val, _std(X_test), y_test
