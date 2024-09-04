import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Input

def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

def preprocess_data(df):
    df_close = df.reset_index()['Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_close = scaler.fit_transform(np.array(df_close).reshape(-1, 1))
    return df_close, scaler

def split_data(df_close, perc=0.8):
    train_size = int(len(df_close) * perc)
    test_size = len(df_close) - train_size
    train_data, test_data = df_close[0:train_size, :], df_close[train_size:len(df_close), :1]
    return train_data, test_data

def create_dataset(dataset, lag=1):
    X, y = [], []
    for i in range(len(dataset) - lag - 1):
        a = dataset[i:(i + lag), 0]
        X.append(a)
        y.append(dataset[i + lag, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=64):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=True)
    return history

def make_predictions(model, X_train, X_test, scaler):
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    return train_predict, test_predict

def calculate_rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def plot_predictions(df_close, train_predict, test_predict, scaler):
    look_back = 100
    trainPredictPlot = np.empty_like(df_close)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    testPredictPlot = np.empty_like(df_close)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df_close) - 1, :] = test_predict

    plt.plot(scaler.inverse_transform(df_close))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

def predict_future(model, test_data, n_steps=100, n_days=30):
    x_input = test_data[605:].reshape(1, -1)
    temp_input = list(x_input[0])
    lst_output = []
    i = 0
    while i < n_days:
        if len(temp_input) > n_steps:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1
    return lst_output

def plot_future_predictions(df_close, lst_output, scaler):
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)

    plt.plot(day_new, scaler.inverse_transform(df_close[len(df_close) - 100:]))
    plt.plot(day_pred, scaler.inverse_transform(lst_output))

    df3 = df_close.tolist()
    df3.extend(lst_output)
    plt.plot(df3[1200:])

    df3 = scaler.inverse_transform(df3).tolist()
    plt.plot(df3)
    plt.show()

# Example usage
if __name__ == "__main__":
    ticker = 'AAPL'
    start = '2010-01-01'
    end = '2024-01-01'

    df = download_data(ticker, start, end)
    df_close, scaler = preprocess_data(df)
    train_data, test_data = split_data(df_close)
    
    lag = 100
    X_train, y_train = create_dataset(train_data, lag)
    X_test, y_test = create_dataset(test_data, lag)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = build_model((lag, 1))
    train_model(model, X_train, y_train, X_test, y_test)
    
    train_predict, test_predict = make_predictions(model, X_train, X_test, scaler)
    train_rmse = calculate_rmse(y_train, train_predict)
    test_rmse = calculate_rmse(y_test, test_predict)
    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)
    
    plot_predictions(df_close, train_predict, test_predict, scaler)
    
    lst_output = predict_future(model, test_data)
    plot_future_predictions(df_close, lst_output, scaler)