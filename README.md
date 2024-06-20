# Stock-Portfolio-Tracker/Predictor

This Python project offers a comprehensive stock portfolio tracker with visualization and predictive capabilities. The project utilizes the Yahoo Finance API (yfinance) to fetch historical stock data for a given ticker. The system allows users to input stocks in the format "Ticker Share_Count YYYY-MM-DD" and tracks their portfolio's performance over time. Key features include:

**Portfolio Tracking:** The system keeps track of each stock's performance in the portfolio, including initial investment value, current value, and net profit/loss.

**Portfolio Visualization:** The system provides visualization tools to showcase the performance of individual stocks and the entire portfolio over time. It plots both actual and predicted stock prices, helping users gauge the effectiveness of the predictive model.

**Predictive Modeling:** The project implements a neural network predictive model using TensorFlow and LSTM layers to forecast future stock prices. The model takes historical data as input and provides predictions for the next day's closing price.

**Interactive Interface:** The program offers an interactive command-line interface where users can request stock predictions, view transaction history, calculate total net profit, and visualize portfolio profits.

**Limitations:** The current version has limitations, such as the inability to add more shares to an existing stock and removing stock shares.

The project aims to provide users with insights into their stock portfolio's performance, predictions, and trends using historical data and predictive modeling. The code is organized into classes for Stocks and Portfolios, promoting modularity and scalability.

https://www.loom.com/share/e0b1aaf46d5d4817a8482ebca265dd58?sid=0ea115bf-eaf5-4711-bc5d-52335984503b
