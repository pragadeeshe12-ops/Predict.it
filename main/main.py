import os

from data_source import get_stock_data, add_technical_indicators, cleanup_stock_files
from constant import stock_price,BASE_DIR
from Xboost import load_data, prepare_features, train_test_split_time, train_model, evaluate_model, save_model


def main():
    print("Analyzing the stock market...")
    for stock in stock_price:
        file_path = os.path.join(BASE_DIR, "processed_data", '{}/processed_data.csv'.format(stock))
        get_stock_data(stock, '80mo', '1d')
        print("Stock data fetched and saved to historical_data.csv")
        print("Adding technical indicators to the dataset...")
        add_technical_indicators(stock)
        df, latest_row = load_data(file_path)
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split_time(X, y)
        model = train_model(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test)
        try:
            save_model(model, stock)
        finally:
            cleanup_stock_files(stock)





if __name__ == "__main__":
    main()

