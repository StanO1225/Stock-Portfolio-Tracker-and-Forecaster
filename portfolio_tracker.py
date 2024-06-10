"""
Stock portfolio tracker with visualization and predictive capabilities.

- Limitation: Adding to an existing stock

"""

#Import statements
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import timedelta, datetime as dt
from modeling import get_Predictions
from typing import List, Dict, Tuple, Union, Optional, Any, Callable, Iterable
from plotnine import ggplot, geom_bar, geom_line, aes, labs, ggsave
import bokeh
from bokeh.plotting import figure, show, output_file, save
from bokeh.embed import server_document, file_html, json_item, components
from bokeh.resources import CDN
from bokeh.palettes import Light, tol, viridis  
from bokeh.models import DatetimeTickFormatter, NumeralTickFormatter, ColumnDataSource, BoxAnnotation, LabelSet
from bokeh.transform import cumsum
from math import pi


class Stock:
    def __init__(self, ticker: str, date_shares: List[Tuple[object, float]], data: object):
        self.ticker = ticker 
        self.date_shares = date_shares #List of (datetime, float) to represent each addition
        self.data = data # Pandas dataframe using yfinance data

    def __eq__(self, __value: object) -> bool:
        """
        Two stocks are equal if they have the same ticker.
        """
        return self.ticker == __value.ticker

    def __str__(self) -> str:
        """
        Prints out the information about the stock pertaining to the portfolio.
        """
        res = ""
        for date_added, share_count in self.date_shares: 
            res += f"{share_count} shares of {self.ticker} added on {date_added.strftime('%Y-%m-%d')}. \n"

        return res
    
    def getPurchasePrices(self, date: object=None) -> List[float]:
        """
        Returns a list of prices of the stock when purchased iniitally at each date. If date specified, only returns prices 
        of shares bought on or before the date.  
        """
        res = []

        if date:
            while not date.strftime("%Y-%m-%d") in self.data.iloc[:, 0] :
                date -= timedelta(days = 1)
            
        for date_added, share_count in self.date_shares:
            if date:
                if date_added <= date:
                    res.append(float(self.data.loc[date_added].iloc[3]) * float(share_count))
            else:
                res.append(float(self.data.loc[date_added].iloc[3]) * float(share_count))

        return res

    def getPriceAtDate(self, date=None) -> float:
        """
        Gets current total price of the stock at a given date. 
        """
        if date:
            while not date.strftime("%Y-%m-%d") in self.data.iloc[:, 0] :
                date += timedelta(days = 1)
            price = float(self.data.loc[date]["Close"])   

        else:
            price = float(self.data.iloc[-1][3])   

        return price * self.getShares(date)

    def netProfit(self, date=None) -> float:
        """
        Returns net profit/loss from this stock from dates of investment to date given. 
        """
        shares = [share_count for _, share_count in self.date_shares]

        investments = 0

        for v in self.getPurchasePrices(date):
            investments += v

        return self.getPriceAtDate(date) - investments # Current total value of stocks - Total invested initially 

    def addShares(self, date: str, num_shares) -> None:
        """
        Adds shares to the stock given a date. 
        """
        self.date_shares.append((dt.strptime(date, "'%Y-%m-%d'"), num_shares))
    
    def getShares(self, date: object=None) -> float:
        """
        Returns the total number of shares of this stock before or on given date. 
        """
        if date:
            shares = []
            for d, share_count in self.date_shares:
                if d <= date:
                    shares.append(share_count)
            return sum(shares)
        
        else:
            return sum([share_count for d, share_count in self.date_shares])
    
    def getDates(self) -> List[object]:
        """
        Returns list of dates when stock was purchased. 
        """
        return [dt for dt, _ in self.date_shares]
    
    def profitsData(self):
        """
        Refactors the stock data, used for plotting. 
        """

        min_date = min(self.getDates())
            

        profit_df = pd.DataFrame()

        profit_df["Date"] = self.data.reset_index()[self.data.reset_index()["Date"] >= min_date]["Date"]
        profit_df["Profit"] = profit_df["Date"].apply(self.netProfit)

        return profit_df
    
    def visualize_Stock(self) -> None:
        """
        Creates line graph of stock's performance.
        """
        dat = self.data.reset_index()
        start = min(self.getDates())

        plt.title(f"{self.ticker} Stock Performance")

        sns.lineplot(x="Date", y="Close", data = dat[dat["Date"] >= start.strftime('%Y-%m-%d')],
                     err_style="bars")

        plt.savefig(r"C:\Users\stano\Documents\Projects\Stock Portfolio\static\stock.jpg")
        pass

    def make_Predictions(self) -> float:
        return get_Predictions(self.data)

class Portfolio:
    def __init__(self, stock_list = [], transactions = [], net_worth = 0):
        self.stock_list = stock_list # List of stock objects 
        # self.inactive_stocks = []
        self.transactions = transactions # List of transactions 
        self.net_worth = net_worth # Total assets invested in the portfolio 
    def numStocks(self):
        return len(self.to_list())

    def to_list(self):
        stocks = []
        
        for s in self.stock_list:
            if s.ticker not in stocks:
                stocks.append(s.ticker)
        return stocks

    # Input: Stock Ticker 
    # Return None if stock is not in portfolio
    def get_stock(self, stock_ticker: str) -> Stock:
        """
        Retrieves stock if exists in portfolio, else returns None.
        """
        stock_ticker = stock_ticker.upper()
        for st in self.stock_list:
            if st.ticker == stock_ticker: 
                return st
        return None
    
    #Adds stock to portfolio, updates transaction history
    def add_Stock(self, tick: str, date: str, num_shares: float) -> None:
        st = self.get_stock(tick)
        invest_dt = dt.strptime(date, "%Y-%m-%d")

        if st:
            st.date_shares.append((invest_dt, num_shares))
            pass
    
        try: 
            tick_data = yf.download(tick)
        except:
            print("Invalid Stock")
            pass

        #Gets stock data from yfinance
        # print(tick_data)

        #Check if date is valid in data
        while not invest_dt.strftime("%Y-%m-%d") in tick_data.iloc[:, 0] :
            invest_dt += timedelta(days = 1)
            #print(f"Date added changed to {}")

        stock = Stock(tick, [(invest_dt, num_shares)], tick_data)   

        stock.visualize_Stock()

        self.stock_list.append(stock)

        self.transactions.append(str(stock))

        self.net_worth += stock.netProfit()

        pass

    def remove_Stock(self, tick: str, date: str, num_shares: float = None) -> None:
        """
        Removes num_shares of a stock from the portfolio. Removes all shares if num_shares is not specified. 
        Removed stock shares are represented by negative share counts. 
        """
        st = self.get_stock(tick)

        if not st:
            print(f"ERROR: Stock {tick} not found in portfolio.")
            pass

        elif not num_shares:
            st.addShares(tick, date, -st.getShares())
            self.net_worth -= st.get_Curr_Value()
            pass

        elif num_shares > st.getShares():
            print(f"ERROR: Removing {num_shares} of {tick} but portfolio only contains {st.getShares()}")
            pass

        else: 
            self.net_worth -= (st.get_Curr_Value() / st.getShares) * num_shares
            st.addShares(tick, date, -num_shares)
            pass

    #Calculates portfolio net profit 
    def calc_Net_Profit(self):
        profit = 0
        for st in self.stock_list:
            profit += st.netProfit()
        return profit
    
    def update_Net_Worth(self) -> None:
        worth = 0
        for st in self.stock_list:
            worth += st.get_Curr_Value()
        return worth

    #Visualizes the total profit from this portfolio over time (starting from earliest stock in portfolio)
    def visualize_Profits(self):
        all_profits = pd.DataFrame(data = {"Date":[]})
        stock_values = pd.DataFrame(data = {"Date":[]})

        for stock in self.stock_list:
            profit_data = stock.profitsData()

            stock_data = pd.DataFrame(stock.data.reset_index(inplace = False)["Date"])
            stock_data = stock_data[stock_data["Date"] >= min(stock.getDates())]
            stock_data[stock.ticker] = stock_data["Date"].apply(stock.getPriceAtDate)

            all_profits = pd.merge(all_profits, profit_data, how = "outer", on="Date", suffixes=(None, f"_{stock.ticker}"))
            stock_values = pd.merge(stock_values, stock_data, how = "outer", on = "Date", suffixes=(None, f".{stock.ticker}"))
            stock_values[stock.ticker] = stock_values.filter(like=stock.ticker).sum(axis=1)

        all_profits["Total_Profit"] = all_profits.sum(axis = 1, numeric_only=True, skipna=True)
        stock_values = stock_values.replace(np.nan, 0)    
        stock_values = stock_values.drop(columns=stock_values.filter(like='.').columns)
        
        stock_recents = stock_values.drop(columns="Date").iloc[-1]
        pf_total = stock_recents.sum()

        stock_pct = stock_recents / pf_total

        stock_pct = stock_pct.reset_index(name="Value").rename(columns={'index' : 'Stock'})

        stock_pct['Angle'] = stock_pct['Value'] * 2 * pi
        stock_pct['Color'] = viridis(len(self.to_list()))

        format = lambda x: f"{x * 100:.1f} %"
        stock_pct['Value_str'] = stock_pct['Value'].map(format)
        stock_pct['Value_str'] = stock_pct['Value_str'].str.pad(20, side="left")
        pct_cds = ColumnDataSource(stock_pct)

        print(stock_pct)

        # stocks_profit_pct = pd.DataFrame()
        # for stock in self.stock_list:
        #     stocks_profit_pct[stock.ticker] = stock_recents[stock.ticker] / stock_recents["Total Value"]

        # stocks_profit_pct = ColumnDataSource(stocks_profit_pct)

        low_box = BoxAnnotation(top = 0, fill_alpha=0.15, fill_color="red")
        high_box = BoxAnnotation(bottom = 0, fill_alpha = 0.15, fill_color="green")
        p1 = figure(title="Total Profits Over Time", x_axis_label="Date", y_axis_label="Total Profit", x_axis_type="datetime")
        p1.line(x=all_profits["Date"], y=all_profits["Total_Profit"])

        p1.yaxis[0].formatter = NumeralTickFormatter(format="$0")
        p1.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")

        p1.add_layout(low_box)
        p1.add_layout(high_box)

        p2 = figure(title = "Investment History", x_axis_label = "Date", y_axis_label = "Portfolio Value", x_axis_type = "datetime")
        p2.varea_stack(stackers = stock_values.drop(columns=["Date"]).columns, x = "Date", color = viridis(len(self.to_list())), legend_label= self.to_list(), source=stock_values)

        p2.yaxis[0].formatter = NumeralTickFormatter(format="$0")
        p2.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")
        p2.legend.orientation = "horizontal"

        p3 = figure(x_range = (-0.5, 1.0), title="Portfolio Performance", tools="hover", tooltips = "@Stock: @Value")
        p3.wedge(x = 0, y = 1, radius=0.4, start_angle=cumsum('Angle', include_zero=True), end_angle = cumsum('Angle'), line_color="white", fill_color="Color", source=stock_pct, legend_field="Stock")

        labels = LabelSet(x=0, y=1, text='Value_str', angle=cumsum('Angle', include_zero=True), source=pct_cds)

        p3.add_layout(labels)
        p3.axis.axis_label=None
        p3.axis.visible=False
        p3.grid.grid_line_color=None

        script, div = components({"Profit Line" : p1, "Investment History" : p2, "Portfolio Performance" : p3})

        return script, div

    def graphLine():
        pass

    def graphProfits():
        pass

    def graphInvestments():
        pass

    def graphBreakdown():
        pass

    def getTransactions(self):
        return self.transactions

    def process_Request(self):
        while (1):
            print("Enter your request. For a list of options enter 'Help': ")
            line = input().lower()
            
            match line:
                case "help":
                    pass
                case "predict":
                    req = input("Enter a stock ticker: \n")
                    st = self.get_stock(req)
                    print(st.make_Predictions())
                case "history":
                    print(self.getTransactions())
                case "profit":
                    print("$ " + str(self.calc_Net_Profit()))
                case "q":
                    return
                case "v":
                    req = input("Enter a stock ticker: \n")
                    st = self.get_stock(req)
                    # while not st is None:
                    #     st = self.get_stock(input("Stock not in portfolio, enter again:"))
                    st.visualize_Stock()
        pass

def run():
    line = None
    pf = Portfolio()

    #Continuously asks for stock input until user is done
    while True:
        print("Enter a stock in the format Format: 'Ticker Share_Count YYYY-MM-DD' or enter 'Done'.")
        line = input("Stock: ")

        if line.lower() == "done":
            break

        #Fetches tick + shares from input 
        stock_input = line.split()
        tick = stock_input[0].upper()
        shares = float(stock_input[1])
        date = stock_input[2]
        
        #Creates stock object and adds to portfolio
        pf.add_Stock(tick, date, shares)
        pf.getTransactions()

    # pf.visualize_Profits()

    pf.process_Request()

# pf = Portfolio()

# pf.add_Stock("AAPL", "2024-04-30", 1)

# pf.add_Stock("AAPL", "2023-05-31", 1)

# pf.add_Stock("AMZN", "2022-02-01", 1)

# pf.visualize_Profits()
# aapl = pf.get_stock("AAPL").profitsData()
# amzn = pf.get_stock("AMZN").profitsData()

# print(aapl)
# print(amzn)

# print(pd.merge(aapl, amzn, how = "outer", on="Date", suffixes=("AAPL", "AMZN")))



# df = yf.download("AAPL")
# print(df.head())

