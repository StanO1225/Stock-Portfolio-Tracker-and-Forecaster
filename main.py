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

# Prepares data used to fit model
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

def create_Model(max_units, num_Dense, num_LSTM, timeframe, active_func = 'None'):
    #Building model
    model = tf.keras.Sequential()

    model.add(LSTM(units = max_units, activation=active_func, return_sequences=True, input_shape = (1, timeframe)))
    model.add(Dropout(0.2))
    for i in range(num_LSTM):
        if i == num_LSTM - 1:
            model.add(LSTM(units = max_units//2, activation=active_func, return_sequences=False))
        else: 
            model.add(LSTM(units = max_units//2, activation=active_func, return_sequences=True))
            model.add(Dropout(0.2))

    for i in range(num_Dense):
        model.add(Dense(units = max_units//(2**i), activation=active_func))
    model.add(Dense(units=1))

    return model

#Creates a neural network using the stock history data passed in. Last 180 days used - 120 for training, 60 for testing model
def get_Predictions(data : object) -> object:
    #Using 10 days at a time to predict next val, train data size, test data size
    predict_days = 10
    train_days = 120
    test_days = 60

    data = data.iloc[-(train_days + test_days):,]
    data = data.drop(set(data.columns).difference(['Close']), axis=1)

    test_data = np.reshape(data.iloc[-test_days:,0], (-1,1))
    train_data = np.reshape(data.iloc[-train_days - test_days: -test_days,0], (-1,1))
    predict_data = np.reshape(data.iloc[-predict_days:, 0], (-1,1))

    #Transforming data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(train_data)

    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    predict_data = scaler.transform(predict_data)

    X_train, y_train = create_window(train_data, predict_days)
    X_test, y_test = create_window(test_data, predict_days)

    def hyperparameterize():
        #Hyperparameterization
        r2scores = {}
        max_nodes = [64, 96, 128, 256]
        num_Dense = range(3)
        num_LSTM = range(1, 4)
        activations = [None, 'relu']
        epoch_list = [10, 30, 50]

        for n in max_nodes:
            for d in num_Dense:
                for l in num_LSTM:
                    for a in activations:
                        for e in epoch_list:
                            model = create_Model(n, d, l, predict_days, a)
                            model.compile(loss='mae', optimizer='adam')
                            model.fit(X_train, y_train, epochs=e)
                            test_preds = model.predict(X_test)
                            r2scores[f"nodes: {n}, dense: {d}, lstm: {l}, {a}, epoch: {e}"] = r2_score(test_preds, y_test)
                            
        r2scores = {k: v for k, v in sorted(r2scores.items(), key=lambda item: item[1])}
        print(r2scores)

    # from hyperparameter optimization: nodes: 128, dense: 2, lstm: 2, None, epoch: 50': 0.6777185615111131
    r2scores=[]
    # for i in range(100):

    model = create_Model(128, 3, 2, predict_days, None)
    model.compile(loss='mae', optimizer='adam')
    model.fit(X_train, y_train, epochs=50)
    test_preds = model.predict(X_test)
    r2scores.append( r2_score(test_preds, y_test))

    print(sorted(r2scores), sum(r2scores)/len(r2scores))
    #Predictions + Plotting

    train_preds = model.predict(X_train)
    # test_preds = model.predict(X_test)
    # print(r2_score(train_preds, y_train), r2_score(test_preds, y_test))

    #Graphing Predictions on Train + Testing Data

    train_prices = scaler.inverse_transform(train_preds)
    test_prices = scaler.inverse_transform(test_preds)

    train_index = data.index[-train_days - test_days + predict_days : -test_days]
    test_index = data.index[-test_days + predict_days:]

    train_prices_series = pd.Series(train_prices.flatten(), index = train_index)
    test_prices_series = pd.Series(test_prices.flatten(), index = test_index)

    train_data, test_data = data.copy(), data.copy()
    train_data.columns, test_data.columns = ['Actual Price'], ['Actual Price']
    train_data = train_data.merge(right=train_prices_series.rename("Train Predictions"), left_index = True, right_index = True)
    test_data = test_data.merge(right=test_prices_series.rename("Test Predictions"), left_index = True, right_index = True)
     
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1,2, sharey = True, figsize=(12,8))
    fig.suptitle("Predictions on Train Set (left) and Test Set (right)")

    axes[0].set_title("Model Predictions on Training Data")
    sns.lineplot(ax = axes[0], data=train_data, x = train_data.index, y = "Actual Price", label="Actual Price")
    sns.lineplot(ax = axes[0], data=train_data, x = train_data.index, y = "Train Predictions", label = "Predicted Price")

    axes[1].set_title("Model Predictions on Testing Data")
    sns.lineplot(ax = axes[1], data=test_data, x = test_data.index, y = "Actual Price", label="Actual Price")
    sns.lineplot(ax = axes[1], data=test_data, x = test_data.index, y = "Test Predictions", label = "Predicted Price")

    plt.savefig("StockPredictions.jpg")
    plt.show()

    #Getting predictions for the next day
    predict_data = np.array(predict_data)
    predict_data = np.reshape(predict_data, (1, 1, predict_data.shape[0]))
    pred = model.predict(predict_data)

    pred = scaler.inverse_transform(pred)
    print(pred[0])

    return pred[0]



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
    def make_Predictions(self) -> float:
        return get_Predictions(self.data)

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
        plt.figure(figsize=(16,12))
        plt.title("Portfolio Net Profit by Time")
        graph_ = sns.lineplot(data=profits_df, x=profits_df.index, y="Total Profits", c='Green')
        graph_.xaxis.set_major_locator(ticker.LinearLocator(5))
        plt.savefig("Portfolio_Profits.jpg")
        plt.show()

        pass

    def print_Transactions(self):
        print(self.transactions)
        pass

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
                    self.print_Transactions()
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

pf.process_Request()
    

