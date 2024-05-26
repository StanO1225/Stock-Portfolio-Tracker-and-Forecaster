import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
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

    return pred[0]