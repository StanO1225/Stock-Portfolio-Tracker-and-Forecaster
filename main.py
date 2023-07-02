#Import statements
import yfinance as yf
import pandas as pd
import seaborn as sns
from datetime import date

#Class definitions
# Stock Class:
"""
ticker - ticker tag string
date_added - date added to 

"""
class Stock:
    def __init__(self, ticker: str, date_added: str, data, share_count: float):
        self.ticker = ticker
        self.date_added = date_added
        self.data = data
        self.share_count = share_count
        
    def __eq__(self, __value: object) -> bool:
        return self.ticker == __value.ticker

    def __str__(self):
        return f"{self.share_count} shares of {self.ticker} added on {self.date_added}."
    
    def get_Profit(self):

        pass
    
class Portfolio:
    def __init__(self, stocks, transactions = ""):
        self.stock_list = stocks
        self.transactions = transactions

    #Return None if stock is not in portfolio
    def get_stock(self, stock_ticker: str):
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
                case "Quit":
                    return

        pass

#Start + End dates for yfinance data used
start_date = "2000-01-01"
end_date = date.today()

line = None
stock_list = []
pf = Portfolio(stock_list)

#Continuously asks for stock input until user is done
while line != "Done":
    print("Enter a stock in the format Format: \"Ticker Share_Count Date\"")
    line = input("Stock: ")

    if line == "Done":
        break

    #Fetches tick + shares from input 
    stock_input = line.split()
    tick = stock_input[0]
    share_count = stock_input[1]

    #Gets stock data from yfinance
    tick_data = yf.download(tick)
    print(type(tick_data))

    #Creates stock object and adds to portfolio
    stock = Stock(tick, start_date, tick_data, share_count)
    pf.add_Stock(stock)

    
pf.process_Request()
    

