import math
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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

  # add a column of time difference
  train_data['time'] = train_data.index
  test_data['time'] = test_data.index
  train_data['timeSince']=train_data.time-train_data.time[0]
  test_data['timeSince']=test_data.time-test_data.time[0]
  train_data['timeSince']=train_data['timeSince'].apply(lambda x: x / np.timedelta64(1,'D'))
  test_data['timeSince']=test_data['timeSince'].apply(lambda x: x / np.timedelta64(1,'D'))


  # transform target into log
  for col in ['casual', 'registered', 'count']:
    train_data['log-' + col] = train_data[col].apply(lambda x: math.log(1 + x))
    

  
  # selected column names
  selected_cols = [u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp',u'humidity', u'windspeed',u'hour',u'month',u'year',u'timeSince']
      
  params = {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 1,'learning_rate': 0.05, 'loss': 'ls'}

  clf_cas = GradientBoostingRegressor(**params)
  clf_reg = GradientBoostingRegressor(**params)

  clf_cas.fit(train_data[selected_cols], train_data['log-casual'])
  clf_reg.fit(train_data[selected_cols], train_data['log-registered'])
  
  train_data['prediction_cas'] = clf_cas.predict(train_data[selected_cols])
  train_data['prediction_reg'] = clf_reg.predict(train_data[selected_cols])
  train_data['prediction'] = train_data['prediction_cas'].apply(lambda x: math.exp(x)-1) + train_data['prediction_reg'].apply(lambda x: math.exp(x)-1)

# round to closest integer
  train_data['prediction'].apply(round)
  train_data['prediction'].apply(lambda x: x if x>0 else 0)
  
  mse = mean_squared_error(train_data['count'], train_data['prediction'])
  print("MSE: %.4f" % mse)
  
  test_data['prediction_cas'] = clf_cas.predict(test_data[selected_cols])
  test_data['prediction_reg'] = clf_reg.predict(test_data[selected_cols])
  test_data['prediction'] = test_data['prediction_cas'].apply(lambda x: math.exp(x)-1) + test_data['prediction_reg'].apply(lambda x: math.exp(x)-1)
  
  
  test_data['prediction'].apply(round)
  test_data['prediction'].apply(lambda x: x if x>0 else 0)
  
  test_data['prediction'].to_csv('output.csv',header=['count'])  
