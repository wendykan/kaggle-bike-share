import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
  train_data = pd.read_csv('train.csv',skipinitialspace=1,index_col=0,parse_dates=True)
  test_data = pd.read_csv('test.csv',skipinitialspace=1,index_col=0,parse_dates=True)
  
  # adding hours as feature
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
  selected_cols = [u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp', u'humidity', u'windspeed',u'hour',u'month',u'day',u'year']
  
  X_train_wd, y_train_wd = train_wd[selected_cols], train_wd['count']
  X_train_nwd, y_train_nwd = train_nwd[selected_cols], train_nwd['count']
    
  params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,'learning_rate': 0.01, 'loss': 'ls'}

  clf_wd = ensemble.GradientBoostingRegressor(**params)
  clf_nwd = ensemble.GradientBoostingRegressor(**params)

  clf_wd.fit(X_train_wd, y_train_wd)
  clf_nwd.fit(X_train_nwd, y_train_nwd)
  
  train_wd['prediction'] = clf_wd.predict(X_train_wd)
  train_nwd['prediction'] = clf_nwd.predict(X_train_nwd)
  train_data = train_wd.append(train_nwd)
  train_data[train_data['prediction']<0] = 0
  
  mse = mean_squared_error(train_data['count'], train_data['prediction'])
  print("MSE: %.4f" % mse)
  
  test_wd['prediction'] = clf_wd.predict(test_wd[selected_cols])
  test_nwd['prediction'] = clf_nwd.predict(test_nwd[selected_cols])
  test_data = test_wd.append(test_nwd)
  
  test_data[test_data['prediction']<0] = 0
  
  test_data['prediction'].to_csv('output.csv',header=['count'])  
