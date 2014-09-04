import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
  train_data = pd.read_csv('train.csv',skipinitialspace=1,index_col=0,parse_dates=True)
  test_data = pd.read_csv('test.csv',skipinitialspace=1,index_col=0,parse_dates=True)
  
  # adding year/month/day/hour as feature
  train_data['hour']=train_data.index.hour
  test_data['hour']=test_data.index.hour
  train_data['month']=train_data.index.month
  test_data['month']=test_data.index.month
  train_data['day']=train_data.index.day
  test_data['day']=test_data.index.day
  train_data['year']=train_data.index.year
  test_data['year']=test_data.index.year

  # separate the data into working or non-working day models
  train_wd = train_data[train_data['workingday']==1]
  train_nwd = train_data[train_data['workingday']==0]
  test_wd = test_data[test_data['workingday']==1]
  test_nwd = test_data[test_data['workingday']==0]
  
  
  # selected column names
  selected_cols = [u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp',u'humidity', u'windspeed',u'hour',u'month',u'day',u'year']
  
#  X_train_wd, y_train_wd = train_wd[selected_cols], train_wd['count']
#  X_train_nwd, y_train_nwd = train_nwd[selected_cols], train_nwd['count']
    
  #params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,'learning_rate': 0.01, 'loss': 'ls'}

  clf_wd_cas = RandomForestRegressor(n_estimators=200)
  clf_nwd_cas = RandomForestRegressor(n_estimators=200)
  clf_wd_reg = RandomForestRegressor(n_estimators=200)
  clf_nwd_reg = RandomForestRegressor(n_estimators=200)
  

  clf_wd_cas.fit(train_wd[selected_cols], train_wd['casual'])
  clf_nwd_cas.fit(train_nwd[selected_cols], train_nwd['casual'])
  clf_wd_reg.fit(train_wd[selected_cols], train_wd['registered'])
  clf_nwd_reg.fit(train_nwd[selected_cols], train_nwd['registered'])

  train_wd['prediction_cas'] = clf_wd_cas.predict(train_wd[selected_cols])  
  train_nwd['prediction_cas'] = clf_nwd_cas.predict(train_nwd[selected_cols])
  train_wd['prediction_reg'] = clf_wd_reg.predict(train_wd[selected_cols])  
  train_nwd['prediction_reg'] = clf_nwd_reg.predict(train_nwd[selected_cols])
  train_wd['prediction'] = train_wd['prediction_cas'] + train_wd['prediction_reg']
  train_nwd['prediction'] = train_nwd['prediction_cas'] + train_nwd['prediction_reg']
  
  train_data = train_wd.append(train_nwd)

# round to closest integer
  train_data['prediction'].apply(round)
  if(any(train_data['prediction']<0.0)):
    train_data[train_data['prediction']<0.0] = 0.0
  
  mse = mean_squared_error(train_data['count'], train_data['prediction'])
  print("MSE: %.4f" % mse)


  test_wd['prediction_cas'] = clf_wd_cas.predict(test_wd[selected_cols])  
  test_nwd['prediction_cas'] = clf_nwd_cas.predict(test_nwd[selected_cols])
  test_wd['prediction_reg'] = clf_wd_reg.predict(test_wd[selected_cols])  
  test_nwd['prediction_reg'] = clf_nwd_reg.predict(test_nwd[selected_cols])
  test_wd['prediction'] = test_wd['prediction_cas'] + test_wd['prediction_reg']
  test_nwd['prediction'] = test_nwd['prediction_cas'] + test_nwd['prediction_reg']
  test_data = test_wd.append(test_nwd)
  
  
  test_data['prediction'].apply(round)
  if(any(test_data['prediction']<0.0)): 
    test_data[test_data['prediction']<0.0] = 0.0
  
  test_data['prediction'].to_csv('output.csv',header=['count'])  
