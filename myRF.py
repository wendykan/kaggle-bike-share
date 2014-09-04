import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
  
  # selected column names
  selected_cols = [u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp', u'humidity', u'windspeed',u'hour',u'month',u'day',u'year']
  
  X_train, y_train = train_data[selected_cols], train_data['count']
  
#  params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,'learning_rate': 0.01, 'loss': 'ls'}

  clf = RandomForestRegressor(n_estimators=150)

  clf.fit(X_train, y_train)
  mse = mean_squared_error(y_train, clf.predict(X_train))
  print("MSE: %.4f" % mse)
  
  prd = clf.predict(test_data[selected_cols])
  
  prd[prd<0] = 0
  test_data['prediction']=prd
  test_data['prediction'].to_csv('output.csv',header=['count'])  
