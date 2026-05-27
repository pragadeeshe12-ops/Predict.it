from data_source import get_stock_data, add_technical_indicators
from constant import stock_price


def main():
    print("Analyzing the stock market...")
    for stock in stock_price:
        get_stock_data(stock, '73mo', '1d')
        print("Stock data fetched and saved to historical_data.csv")
        print("Adding technical indicators to the dataset...")
        add_technical_indicators(stock)



if __name__ == "__main__":
    main()

