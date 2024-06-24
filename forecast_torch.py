#!/usr/bin/env python3.8
## coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from  copy import deepcopy

from functions_torch import data_loading as _
from functions_torch import lstm_mish as model_lstm_mish
from functions_torch import train_model as train
from functions_torch import forecast_model
from sklearn.preprocessing import MinMaxScaler

from functions_torch import BASE_URL, CONTEX, URL, SYMBOL, INTERVAL, LIMIT



######################################################################
#
#   LOADING  DATA
#
######################################################################

length = 30 
data, data_first = _.data_after_processing(BASE_URL, CONTEX, URL, SYMBOL, INTERVAL, LIMIT, length, _)

data_first.plot()
plt.show()


######################################################################
#
#  TRANSFORMING  DATA  TO  TENSOR
#
######################################################################

dataset = pd.DataFrame.to_numpy(data)

X = dataset[0:-1, 0:1] 
y = dataset[1:, 0:1]        ##   data  shifted  1 (by one)  interval

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


######################################################################
#
#   TRAIN  THE  MODEL
#
######################################################################

b_s = 1

model = model_lstm_mish( hiddensize = 32 )
opt_model = torch.compile( model, mode="reduce-overhead" )
loss_fn = nn.HuberLoss( reduction='mean', delta=90 )  

optimizer = optim.Adam(opt_model.parameters(),  lr=2.4, betas=(1e-5, 11e-6), eps=8e-16, weight_decay= 3.3,
                                        amsgrad=False, fused=False, capturable=False)    

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=1.92,
                                        threshold=1e-12, threshold_mode='rel', cooldown=2.1, min_lr=0.098, eps=8e-16)  

model_trained = train( model_ = opt_model,  loss_function = loss_fn,  optimizer_= optimizer, scheduler_ = scheduler, 
                                       n_epochs=3, batch_size=b_s, X_= X, y_= y)   


for i in range(3):
    print(i)
    torch.save(model_trained.state_dict(), '/.../.../model_t.pt')
    model = torch.load('/.../.../model_t.pt')

    optimizer = optim.Adam(opt_model.parameters(),  lr=4.1, betas=(1e-5, 11e-6), eps=8e-16, weight_decay= 3.3,
                                            amsgrad=False, fused=False, capturable=False)        
                                            
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=1.92,
                                            threshold=1e-12, threshold_mode='rel', cooldown=10.1, min_lr=0.498, eps=8e-16)        

    model_trained = train( model_ = opt_model,  loss_function = loss_fn,  optimizer_= optimizer, scheduler_ = scheduler, 
                                           n_epochs=7, batch_size=b_s, X_= X, y_= y)       

   
######################################################################
#
#   PREDICTIONS  FOR  BACKTESTING
#
######################################################################

data_latest = deepcopy(data_first[-(length+1): ]) 
data_latest.loc[:,['Date']] = data_latest.iloc[:].index    
data_latest.index = data_latest['Date']
data_latest = data_latest.drop('Date', axis=1)

dataset = pd.DataFrame.to_numpy(data_latest)
X = dataset[0:-1, 0:1]
X = torch.tensor(X, dtype=torch.float32)

forecast = forecast_model(model_trained, X, prediction_length=length) 

print(forecast)
print(type(forecast))

data = deepcopy(data_first[-length: ]) 
data['Forecast'] = forecast

mean_forecast_error = mean( data['Forecast'] ) - data['Real_datas'][0]

data['Forecast'] = data['Forecast'] - mean_forecast_error

data['Forecast']  = data['Forecast'][1:] 
data['Real_datas'] = data['Real_datas'][1:]
data = data[1:]
print(data)


### ### ### ### ### ### ### ### ### ### ### ###
#
#   Accuracy  test 
#
### ### ### ### ### ### ### ### ### ### ### ###

forecast_errors_mean = round( mean(abs(data['Forecast'] - data['Real_datas'])), 2 )
print("\n"*2)
print(f"    Forecast  errors  mean        {forecast_errors_mean} $ ")

data_mean =  mean(data['Real_datas'])
accuracy = round( 100 * ( abs(1 - (forecast_errors_mean / data_mean)) ), 3 )
print(f"    Accuracy  test                {accuracy} %  ")
print(f"    Minimum  expected  accuracy   {99.73}  %  ")   
print("\n"*2)

plt.plot(data['Forecast'] , label = 'Forecast', marker="o", color = 'blue')
plt.plot(data['Real_datas'] , label = 'Real_datas', marker="o", color = 'black')
plt.ylabel('Price')
plt.legend()
plt.show()



#####################################################################
#
#   FORECAST
#
#####################################################################

b_s = 1

data_first = data_first[length: ]
dataset_2 = pd.DataFrame.to_numpy(data_first)  # for forecast

X_2 = dataset_2[0:-1, 0:1] 
y_2 = dataset_2[1: , 0:1]    

X_2 = torch.tensor(X_2, dtype=torch.float32)
y_2 = torch.tensor(y_2, dtype=torch.float32)

#################################################

model_prediction = model_lstm_mish( hiddensize = 32 )
opt_model = torch.compile( model_prediction, mode="reduce-overhead" )
loss_fn = nn.HuberLoss( reduction='mean', delta=90 )


optimizer = optim.Adam(opt_model.parameters(),  lr=2.4, betas=(1e-5, 11e-6), eps=8e-16, weight_decay= 3.3,
                                        amsgrad=False, fused=False, capturable=False)    

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=1.92,
                                        threshold=1e-12, threshold_mode='rel', cooldown=2.1, min_lr=0.098, eps=8e-16) 
                                        
model_trained_prediction = train( model_ = opt_model,  loss_function = loss_fn,  optimizer_= optimizer, 
                                       scheduler_ = scheduler,  n_epochs=3, batch_size=b_s, X_= X_2, y_= y_2 ) 
 

for i in range(3):
    print(i)
    torch.save(model_trained_prediction.state_dict(), '/.../.../model_t_fore.pt')
    model = torch.load('/.../.../model_t_fore.pt')
    
    optimizer = optim.Adam(opt_model.parameters(),  lr=4.1, betas=(1e-5, 11e-6), eps=8e-16, weight_decay= 3.3,
                                            amsgrad=False, fused=False, capturable=False)             
                                            
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=1.92,
                                            threshold=1e-12, threshold_mode='rel', cooldown=10.1, min_lr=0.498, eps=8e-16)  

    model_trained_prediction = train( model_ = opt_model,  loss_function = loss_fn,  optimizer_= optimizer, 
                                                                scheduler_ = scheduler, n_epochs=7, batch_size=b_s, X_= X_2, y_= y_2)


##################################
##    FORECAST
##################################
forecast_2 = forecast_model(model_trained_prediction, X_2, prediction_length=length)  
##################################
 
data = data_first[-length: ] 
data['Forecast'] = forecast_2

##################################

prediction_minutes  =  length 

df_past = data[['Real_datas']].reset_index()
df_past.rename(columns={'index': 'Date', 'Real_datas': 'Actual'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan
df_past['Forecast'].values[-1] = df_past['Actual'].values[-1].copy()

##################################
df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(minutes=1), 
                                                                                        periods=prediction_minutes, freq ='1T')
##################################
df_future['Forecast'] = forecast_2.flatten()

mean_forecast_error = mean( data['Forecast'] ) - data['Real_datas'][-1]
df_future['Forecast'] = round( df_future['Forecast'] - mean_forecast_error, 1 )

df_future['Forecast'] = df_future['Forecast'] 
df_future['Real_datas'] = df_future['Actual'] 
df_future = df_future 

df_future['Actual'] = np.nan
results = pd.concat([df_past, df_future], axis=0, ignore_index=True)
results.index = results['Date']
del(results['Date'])
del(results['Real_datas'])
print(results)

##################################
plt.plot(results['Forecast'] , label = 'Forecast', marker="o", color = 'brown')
plt.plot(results['Actual'] , label = 'Actual', marker="o", color = 'blue')
plt.ylabel('Price')
plt.legend()
plt.show()
#
## END
#
