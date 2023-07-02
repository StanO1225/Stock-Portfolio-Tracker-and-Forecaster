#Import statements
import yfinance as yf
import pandas as pd
import seaborn as sns
from datetime import date


#Class definitions
class Stock:
    def __init__(self, ticker: str, date_added, data):
        self.ticker = ticker
        self.date_added = date_added
        self.data = data
        
    def __eq__(self, __value: object) -> bool:
        return self.ticker == __value.ticker

class Portfolio:
    def __init__(self, stocks):
        self.stock_list = stocks
    
    def add_Stock(self, stock):
        self.stock_list.append(stock)
        pass

    def print_Portfolio(self):
        pass

    def calc_Net_Profit(self):
        pass

#Start + End dates for yfinance data used
start_date = "2000-01-01"
end_date = date.today()
print(end_date)

line = None
stock_list = []
pf = Portfolio(stock_list)

while line != "Done":
    line = input("Format: \"Ticker Share_Count\"")

    if line == "Done":
        break

    #Fetches tick + shares from input 
    stock_input = line.split()
    tick = stock_input[0]
    share_count = stock_input[1]


    tick_data = yf.download(tick)

    stock = Stock(tick, tick_data)

    pf.add_Stock(stock)

    

