import requests

def get_stock_price(stock_symbol):
    api_key = "54R4LUA9227S7G5W"  # Replace with your actual API key
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock_symbol}&apikey={api_key}"

    response = requests.get(url)
    data = response.json()

    if "Global Quote" in data and "05. price" in data["Global Quote"]:
        price = data["Global Quote"]["05. price"]
        return f"The current stock price of {stock_symbol} is ${price}."
    else:
        return f"Sorry, I couldn't fetch the stock price for {stock_symbol}. Please try again."