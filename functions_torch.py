#!/usr/bin/env python3.8
## coding=utf-8

import torch
import torch.nn as nn
import requests
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from statistics import mean
import math

BASE_URL = "https://api.gateio.ws"
CONTEX = "/api/v4"
URL = '/spot/candlesticks'
SYMBOL = 'BTC_USDT'
INTERVAL = '1m'
LIMIT = '1000'  



######################################################################
#
#   LOADING  DATA
#
######################################################################

class data_loading:

    @classmethod
    def rest_api_get_candlesticks(cls, base_url, contex, url, symbol, interval, limit):
        loading = requests.get(base_url+contex+url, 
                                params = {'currency_pair' : f'{symbol}', 'interval' : f'{interval}', 'limit' : f'{limit}'})
        loading = loading.json()
        return loading

    @classmethod
    def close(cls, r):
        
        close_long =list()
        index_time_long = list()
        
        for kline in r:
            close_long.append(float(kline[2])) 
            time_long = datetime.fromtimestamp(int(kline[0])).strftime("%Y-%m-%d %H:%M:%S")   
            index_time_long.append(time_long)   #  time  in  int()
        return [close_long,  index_time_long]

    @classmethod
    def data_frame(cls, series, time, columns):
        i=0
        ts = list()
        for t in time:
            ts.append(t)
            ts.append(series[i])
            i += 1
        close = pd.Series(series)
        close.index = pd.to_datetime(time)   
        data_f = pd.DataFrame (close, columns = [columns])
        return data_f

    @staticmethod
    def data_after_processing(BASE_URL, CONTEX, URL, SYMBOL, INTERVAL, LIMIT, end, _):
        long = _.rest_api_get_candlesticks(BASE_URL, CONTEX, URL, SYMBOL, INTERVAL, LIMIT)
        close_l, index_time_l = _.close(long)
        data_first = _.data_frame(close_l, index_time_l, "Real_datas")
        
        data = data_first[:-end]  
        # data['Date'] = data.index  
        data.loc[:,['Date']] = data.iloc[:].index    
        data.index = data['Date']
        data = data.drop('Date', axis=1)
        
        data_f = data_first[ : ]
        data_f.loc[:,['Date']] = data_f.iloc[:].index    
        data_f.index = data_f['Date']
        data_f = data_f.drop('Date', axis=1)
        return data, data_f



######################################################################
#
#  TRANSFORMING  DATA  TO  TENSOR
#
######################################################################
# dataset = pd.DataFrame.to_numpy(data)
#
# X = dataset[0:-1, 0:1]     ##   we select the same column  0  [ ... , 0:1 ]
# y = dataset[1:, 0:1]         ##   data shifted by 1 interval
# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32)
## OR 
## X = dataset[ : , 0:1]    ##  we select the same column 0
## y  = dataset[ : , 0] 
## X = torch.tensor(X, dtype=torch.float32)
## y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)   # (rows, columns)



######################################################################
#
#   BUILDING  A  MODEL
#
######################################################################

class lstm_mish(nn.Module):

    def __init__(self, hiddensize):
        
        super(lstm_mish, self).__init__()
        
        n = hiddensize
        """            input_size  ==  neurons            """
        
        self.lstm_1 = nn.LSTM( input_size=1,  hidden_size=n,  num_layers=12,  bias=True, 
                                                 batch_first=True, dropout=1e-3)   
        self.lstm_1_weight = self.lstm_1.weight_ih_l1.data.fill_(30)
        
        self.lstm_2 = nn.LSTM( input_size=n,  hidden_size=n,  num_layers=4,  bias=True,  batch_first=False )
        self.xav_1 = nn.init.xavier_normal_(self.lstm_2.weight_ih_l0)
        
        self.gru_1   = nn.GRU(  input_size=n,  hidden_size=n,  num_layers=3,  bias=True,  batch_first=False) 
        self.xav_2  = nn.init.xavier_normal_(self.gru_1.weight_ih_l0.data.normal_(mean=0.7, std=0.3))  
        
        self.act_1 = nn.Mish()
        
        self.linear_1 = nn.Linear(n, n, bias=True)    
        self.linear_2 = nn.Linear(n, 1, bias=True)       ##    nn.Linear(n, 1)    -    1. (one) neuron at the output    
        self.gru_1_weight = self.gru_1.weight_ih_l1.data.normal_(mean=0.7, std=0.85)   


    def forward(self, x):
 
        x, _ = self.lstm_1(x*2.9)     
        x = self.act_1(x)
        
        x, _ = self.lstm_2(x*3.1)     
        self.xav_1 
        self.lstm_1_weight
        x = self.act_1(x)
        
        x = self.linear_1(x) 
        
        x, _ = self.gru_1(x*3.2) 
        self.xav_2
        self.gru_1_weight
        x = self.act_1(x)
        
        x = self.linear_1(x)  
        x = self.act_1(x)
        
        x = self.linear_1(x)  
        x = self.act_1(x)

        x = self.linear_1(x)     
        
        x, _ = self.lstm_2(x)
        self.xav_1 
        x = self.act_1(x)

        x = self.linear_2(x*3.5)
        return x



######################################################################
#
#   TRAIN  MODEL
#
######################################################################

def  train_model(model_, loss_function, optimizer_, scheduler_, n_epochs, batch_size, X_, y_): 
    
    y_2 = y_ * y_
    y_ = np.sqrt(y_2)
    mean_y = torch.mean(y_)
    
    for epoch in range(n_epochs):
        
        model_.eval()  
        
        for i in range(0, len(X_), batch_size):
            
            optimizer_.zero_grad(set_to_none=True)
            
            X_batch = X_[i:i+batch_size]
            y_pred = model_(X_batch)
            
            y_batch = y_[i:i+batch_size]    
            loss = loss_function(y_pred, y_batch) 

            loss.backward()
            optimizer_.step()
        
        loss_numpy = loss.detach().numpy() 
        loss_sqr = np.sqrt(loss_numpy)
        print(f' Epoch {epoch+1} / {n_epochs},    loss   {loss_sqr}')
        
        scheduler_.step(loss_sqr)
        print( '  lr  =  ',  optimizer_.param_groups[0]["lr"] )
        
        ly =(loss_sqr / mean_y)
        accuracy =  abs(1 - ly )
        print(f" ---  abs[  1 -  loss_sqr / mean_y  ]    train       {accuracy}   ---")
        print("\n")
    return model_



######################################################################
#
#   FORECAST
#
######################################################################

def forecast_model(model_, X_, prediction_length):
   
    model_.eval()
    prediction = model_(X_)
    
    if prediction_length >= len(prediction): 
        prediction_length = len(prediction)
    
    forecast_ =  np.arange(0)
    i=0
    
    while i <= prediction_length-1:
    
        f = prediction[i][0].detach().numpy()
        forecast_ = np.append(forecast_, f)
        i+=1
    
    print(forecast_)
    return forecast_

#
# END
#
