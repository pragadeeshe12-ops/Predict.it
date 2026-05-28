import os

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from huggingface_hub import upload_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. LOAD & PREPROCESS DATA

def load_data(file_path):
    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    latest_row = df.tail(1).copy()
    df_clean = df.dropna()

    return df_clean, latest_row


def encode_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert stock_name into one-hot numeric columns for XGBoost."""
    encoded = df.copy()
    if "stock_name" in encoded.columns:
        encoded = pd.get_dummies(encoded, columns=["stock_name"], prefix="stock")
    return encoded


# 2. FEATURE + TARGET SPLIT
def prepare_features(df, target_col="target_range"):
    drop_cols = ["Date", "target", "target_range"]

    # keep only available columns
    drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=drop_cols)
    X = encode_stock_features(X)
    y = df[target_col]

    return X, y


# 3. TIME-BASED SPLIT
def train_test_split_time(X, y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# 4. TRAIN MODEL
def train_model(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    # Keep training feature order for inference-time column alignment.
    model.feature_columns = list(X_train.columns)

    return model


# 5. EVALUATE MODEL
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nModel Performance:")
    print(f"MAE  : {mae}")
    print(f"RMSE : {rmse}")

    return y_pred



# 6. SAVE MODEL
def save_model(model, path="xgb_model.pkl"):
    joblib.dump(model, path)
    upload_file(
        path_or_fileobj=os.path.join(BASE_DIR, "main/xgb_model.pkl"),
        path_in_repo="xgb_model.pkl",
        repo_id="praga-deesh/predict.id",
        repo_type="model"
    )



if __name__ == "__main__":
    file_path = os.path.join(BASE_DIR, "processed_data", "processed_data.csv")
    print(file_path)

    # Step 1: Load (clean data for training + latest row for prediction)
    df, latest_row = load_data(file_path)
    print(df)

    # Step 2: Prepare features
    X, y = prepare_features(df)

    # Step 3: Split
    X_train, X_test, y_train, y_test = train_test_split_time(X, y)
    print(df["target_range"].describe())

    # Step 4: Train
    model = train_model(X_train, y_train, X_test, y_test)

    # Step 5: Evaluate
    evaluate_model(model, X_test, y_test)

    # Step 6: Save
    save_model(model)
