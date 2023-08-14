"""
Stock portfolio tracker with visualization and predictive capabilities.


- Limitation: Adding to an existing stock

"""

#Import statements
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import timedelta, datetime as dt

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import r2_score



def create_window(data, predict_days=30):
    X_data, y_data = [], []
    #Using 30 days at a time to predict next val
    print(len(data))
    for x in range(predict_days, len(data)):
        X_data.append(data[x - predict_days: x])
        y_data.append(data[x])
    X_data, y_data = np.array(X_data), np.array(y_data)
    X_data = np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))
    return X_data, y_data

#Creates a neural network using the stock history data passed in. Last 180 days used - 120 for training, 60 for testing model
def test_Model(data : object) -> object:
    train_days = 120
    test_days = 60

    data = data.iloc[-(train_days + test_days):,]
    data = data.drop(set(data.columns).difference(['Close']), axis=1)

    test_data = np.reshape(data.iloc[-test_days:,0], (-1,1))
    train_data = np.reshape(data.iloc[-train_days - test_days: -test_days,0], (-1,1))

    #Transforming data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # print(train_data)

    #Using 10 days at a time to predict next val
    predict_days = 10
    X_train, y_train = create_window(train_data, predict_days)
    X_test, y_test = create_window(test_data, predict_days)

    print(X_train.shape, X_test.shape)
    # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


    #Building model
    model = tf.keras.Sequential()
    model.add(LSTM(units = 128, activation='relu', input_shape = (1, predict_days)))
    model.add(Dropout(0.2))
    model.add(Dense(units = 64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 32, activation= 'relu'))
    model.add(Dense(units = 16, activation= 'relu'))
    model.add(Dense(units=1, activation='relu'))

    model.compile(loss='mae', optimizer='adam')
    model.fit(X_train, y_train, epochs=50)
    print(model.summary())

    #Predictions
    # train_preds = np.reshape(model.predict(X_train), (X_train[1], 1))
    # test_preds = np.reshape(model.predict(X_test), (X_test[1], 1))

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print(y_test.shape, test_preds.shape)
    print(y_train.shape, train_preds.shape)
    print(r2_score(train_preds, y_train), r2_score(test_preds, y_test))

    return model



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

    def __str__(self) -> str:
        return f"{self.share_count} shares of {self.ticker} added on {self.date_added.strftime('%Y-%m-%d')}."
    
    def get_Start_Value(self) -> float:
        return float(self.data.loc[self.date_added].iloc[3]) * float(self.share_count)

    def get_Curr_Value(self) -> float:
        return float(self.data.iloc[-1][3]) * self.share_count

    #Returns net profit/loss from this stock
    def get_Profit(self):
        return self.get_Curr_Value() - self.get_Start_Value
    
    def visualize_Stock(self) -> None:
        dat = self.data.reset_index()
        plt.title(f"{self.ticker} Stock Performance")
        sns.lineplot(x="Date", y="Close", data = dat[dat["Date"] >= self.date_added.strftime('%Y-%m-%d')],
                     err_style="bars")
        plt.show()
        pass
    
    def predict(self):
        pass

class Portfolio:
    def __init__(self, stock_list = [], transactions = "", net_worth = 0):
        self.stock_list = stock_list
        self.transactions = transactions
        self.net_worth = net_worth

    # Input: Stock Ticker 
    # Return None if stock is not in portfolio
    def get_stock(self, stock_ticker: str) -> Stock:
        stock_ticker = stock_ticker.upper()
        for st in self.stock_list:
            if st.ticker == stock_ticker :
                return st
        return None
    
    #Adds stock to portfolio, updates transaction history
    def add_Stock(self, stock) -> None:
        st = self.get_stock(stock.ticker)
        if st:
            st.share_count += stock.share_count
        else:    
            self.stock_list.append(stock)
        self.transactions += ("\n" + str(stock))
        pass

    def remove_Stock(self, stock, num_shares) -> None:
        pass

    #Calculates portfolio net profit 
    def calc_Net_Profit(self):
        profit = 0
        for st in self.stock_list:
            profit += st.get_Profit()
        return profit
    
    def update_Net_Worth(self) -> None:
        worth = 0
        for st in self.stock_list:
            worth += st.get_Curr_Value()
        return worth

    #Visualizes the total profit from this portfolio over time (starting from earliest stock in portfolio)
    def visualize_Profits(self):
        profits_df = pd.DataFrame()
        for i, stock in enumerate(self.stock_list):
            dat = stock.data
            stock_profits = dat[stock.date_added : dt.now()].drop(dat.columns.difference(['Close']), axis=1)
            stock_profits.columns = [stock.ticker]
            stock_profits[stock.ticker] = (stock_profits[stock.ticker]*float(stock.share_count) - stock.get_Start_Value()) 
            if i == 0:
                profits_df = stock_profits
            else:
                if len(profits_df.index) < len(stock_profits.index):
                    profits_df = stock_profits.join(profits_df, on=stock_profits.index)
                else:
                    profits_df = profits_df.join(stock_profits, on=profits_df.index)
        profits_df['Total Profits'] = profits_df.sum(axis=1, numeric_only=True)
        profits_df['Positive?'] = profits_df['Total Profits'] >= 0
        print(profits_df)
        
        plt.style.use('ggplot')
        plt.title("Portfolio Net Profit by Time")
        graph_ = sns.lineplot(data=profits_df, x=profits_df.index, y="Total Profits", c='Green')
        graph_.xaxis.set_major_locator(ticker.LinearLocator(5))
        plt.show()

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
                case "Q":
                    return
                case "V":
                    req = input("Enter a stock ticker: \n")
                    st = self.get_stock(req)
                    # while not st is None:
                    #     st = self.get_stock(input("Stock not in portfolio, enter again:"))
                    st.visualize_Stock()
        pass
"""
line = None
pf = Portfolio()

#Continuously asks for stock input until user is done
while line != "Done":
    print("Enter a stock in the format Format: \"Ticker Share_Count YYYY-MM-DD\"")
    line = input("Stock: ")

    if line.lower() == "done":
        break

    #Fetches tick + shares from input 
    stock_input = line.split()
    tick = stock_input[0].upper()
    share_count = stock_input[1]
    invest_datetime = dt.strptime(stock_input[2], "%Y-%m-%d")
    
    #Gets stock data from yfinance
    tick_data = yf.download(tick)
    # print(tick_data)

    #Check if date is valid in data
    while not invest_datetime.strftime("%Y-%m-%d") in tick_data.iloc[:, 0] :
        invest_datetime += timedelta(days = 1)
        #print(f"Date added changed to {}")

    #Creates stock object and adds to portfolio
    stock = Stock(tick, invest_datetime, tick_data, share_count)
    pf.add_Stock(stock)
    pf.print_Transactions()


pf.visualize_Profits()
"""

data = yf.download('aapl')

mod = test_Model(data)

# pf.process_Request()
    

