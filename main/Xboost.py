import os

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

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


# 2. FEATURE + TARGET SPLIT
def prepare_features(df, target_col="target_range"):
    drop_cols = ["Date", "target", "target_range"]

    # keep only available columns
    drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=drop_cols)
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
    print(f"Model saved at {path}")


# 7. LOAD MODEL

def load_model(path="xgb_model.pkl"):
    return joblib.load(path)


# 8. PREDICT NEXT DAY RANGE
def predict_next(model, latest):
    drop_cols = ["Date", "target", "target_range"]
    drop_cols = [col for col in drop_cols if col in latest.columns]

    latest_X = latest.drop(columns=drop_cols).fillna(0)

    # Prediction
    predicted_range = model.predict(latest_X)[0]
    current_price = latest["Close"].values[0]

    # Range
    lower = current_price - (predicted_range / 2)
    upper = current_price + (predicted_range / 2)

    # -------------------------------
    # SIGNALS (from your features)
    # -------------------------------
    rsi = latest["rsi_14"].values[0]
    volume_spike = latest["volume_spike"].values[0]
    support = latest["rolling_low_10"].values[0]
    resistance = latest["rolling_high_10"].values[0]

    # RSI condition
    if rsi < 30:
        rsi_signal = "Oversold"
    elif rsi > 70:
        rsi_signal = "Overbought"
    else:
        rsi_signal = "Neutral"

    # Support check
    near_support = abs(current_price - support) < predicted_range

    # Volume
    volume_signal = "YES" if volume_spike == 1 else "NO"

    return {
        "range": predicted_range,
        "lower": lower,
        "upper": upper,
        "rsi": rsi_signal,
        "volume": volume_signal,
        "near_support": near_support,
        "current_price": current_price
    }

def round_to_tick(price, tick=0.05):
    return round(price / tick) * tick

def generate_signal(data):
    score = 0

    if data["rsi"] == "Oversold":
        score += 1
    if data["near_support"]:
        score += 1
    if data["volume"] == "YES":
        score += 1

    if score >= 2:
        return "BUY"
    elif score == 1:
        return "WEAK BUY"
    else:
        return "NO TRADE"


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

    # Step 7: Predict using the actual latest row (today's data)
    result = predict_next(model, latest_row)
    lower = round_to_tick(result['lower'])
    upper = round_to_tick(result['upper'])
    signal = generate_signal(result)

    print("\n========== TRADING OUTPUT ==========")
    print(f"Current Price       : {result['current_price']:.2f}")
    print(f"Predicted Range     : {result['range']:.2f}")
    print(f"Expected Range      : {int(result['lower'])} - {int(result['upper'])}")
    print(f"RSI Signal          : {result['rsi']}")
    print(f"Volume Spike        : {result['volume']}")
    print(f"Near Support        : {'YES' if result['near_support'] else 'NO'}")
    print(f"\n👉 FINAL SIGNAL     : {signal}")
