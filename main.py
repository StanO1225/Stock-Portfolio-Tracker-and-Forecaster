#Import statements
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime as dt

#Class definitions
# Stock Class:
"""
ticker - ticker tag string
date_added - datetime object

"""
class Stock:
    def __init__(self, ticker: str, date_added: object, data, share_count: float):
        self.ticker = ticker
        self.date_added = date_added
        self.data = data
        self.share_count = share_count
        
    def __eq__(self, __value: object) -> bool:
        return self.ticker == __value.ticker

    def __str__(self):
        return f"{self.share_count} shares of {self.ticker} added on {self.date_added.strftime('%Y-%m-%d')}."
    #Returns 
    def get_Profit(self):
        # print(self.data.loc[self.date_added])
        start_price = self.data.loc[self.date_added].iloc[4]
        curr_price = self.data.iloc[-1][4]
        return curr_price - start_price
    def visualize_Stock(self):
        dat = self.data.reset_index()
        plt.title(f"{self.ticker} Stock Performance")
        sns.lineplot(x="Date", y="Close", data = dat[dat["Date"] >= self.date_added.strftime('%Y-%m-%d')],
                     err_style="bars")
        plt.show()
        pass
    

class Portfolio:
    def __init__(self, stocks, transactions = ""):
        self.stock_list = stocks
        self.transactions = transactions

    # Input: Stock Ticker 
    # Return None if stock is not in portfolio
    def get_stock(self, stock_ticker: str):
        stock_ticker = stock_ticker.upper()
        for st in self.stock_list:
            if st.ticker == stock_ticker :
                return st
        return None
    
    #Adds stock to portfolio, updates transaction history
    def add_Stock(self, stock):
        st = self.get_stock(stock.ticker)
        if st != None:
            st.share_count += stock.share_count
        else:    
            self.stock_list.append(stock)
        self.transactions += ("\n" + str(stock))
        pass

    def remove_Stock(self, stock):
        pass

    #Calculates portfolio net profit 
    def calc_Net_Profit(self):
        profit = 0
        for st in self.stock_list:
            profit += st.get_Profit()
        return profit
    
    def visualize_Portfolio(self):
        pass

    def print_Transactions(self):
        print(self.transactions)
        pass

    def process_Request(self):
        while (1):
            print("Enter your request. For a list of options enter 'Help': ")
            line = input()
            
            match line:
                case "Help":
                    pass
                case "History":
                    self.print_Transactions()
                case "Profit":
                    print("$ " + str(self.calc_Net_Profit()))
                case "Quit":
                    return
                case "V":
                    req = input("Enter a stock ticker: \n")
                    st = self.get_stock(req)
                    # while not st is None:
                    #     st = self.get_stock(input("Stock not in portfolio, enter again:"))
                    st.visualize_Stock()
        pass

line = None
stock_list = []
pf = Portfolio(stock_list)

apple = pd.read_csv('AppleData.csv')
#print(apple.reset_index())
print(apple)
#print(apple[apple["Date"] >= "2020-01-01"])

#Continuously asks for stock input until user is done
while line != "Done":
    print("Enter a stock in the format Format: \"Ticker Share_Count Date\"")
    line = input("Stock: ")

    if line == "Done":
        break

    #Fetches tick + shares from input 
    stock_input = line.split()
    tick = stock_input[0].upper()
    share_count = stock_input[1]
    invest_datetime = dt.strptime(stock_input[2], "%Y-%m-%d")
    
    #Gets stock data from yfinance
    tick_data = yf.download(tick)
    print(tick_data)

    #Check if date is valid in data
    while not invest_datetime.strftime("%Y-%m-%d") in tick_data.iloc[:, 0] :
        invest_datetime += timedelta(days = 1)
        #print(f"Date added changed to {}")

    #Creates stock object and adds to portfolio
    stock = Stock(tick, invest_datetime, tick_data, share_count)
    pf.add_Stock(stock)
    pf.print_Transactions()
print(pf.get_stock("AAPL"))

pf.process_Request()
    

