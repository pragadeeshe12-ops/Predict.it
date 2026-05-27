from data_source import get_stock_data, add_technical_indicators

def main():
    print("Analyzing the stock market...")
    get_stock_data('BEL.NS', '73mo', '1d')
    print("Stock data fetched and saved to historical_data.csv")
    print("Adding technical indicators to the dataset...")
    add_technical_indicators()



if __name__ == "__main__":
    main()

