import os

from data_source import get_stock_data, add_technical_indicators
from constant import stock_price,BASE_DIR
from Xboost import load_data, prepare_features, train_test_split_time, train_model, evaluate_model, save_model


def main():
    file_path = os.path.join(BASE_DIR, "processed_data", "processed_data.csv")
    print("Analyzing the stock market...")
    for stock in stock_price:
        get_stock_data(stock, '73mo', '1d')
        print("Stock data fetched and saved to historical_data.csv")
        print("Adding technical indicators to the dataset...")
        add_technical_indicators(stock)
    df, latest_row = load_data(file_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y)
    print(df["target_range"].describe())
    model = train_model(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    save_model(model)



if __name__ == "__main__":
    main()

