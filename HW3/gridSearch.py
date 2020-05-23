import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV

now = datetime.datetime.now()

# Load the data
train = pd.read_csv('yelp_train.csv')
test = pd.read_csv('yelp_val.csv')
id_test = test.stars
train.sample(3)

y_train_full = train['stars']
x_train_full = train.drop(["stars"], axis=1)

x_test = test.drop(["stars"], axis=1)

# Convert columns that are not numeric to a numeric value
for c in x_train_full.columns:
    if x_train_full[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train_full[c].values))
        x_train_full[c] = lbl.transform(list(x_train_full[c].values))
        # x_train_full.drop(c,axis=1,inplace=True)

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))
        # x_test.drop(c,axis=1,inplace=True)

# Various hyper-parameters to tune
xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07, 0.3, 0.7, 0.5], #so called `eta` value
              'max_depth': [5, 6, 7,4,8,9,10],
              'min_child_weight': [3,4,5,6,7,8],
              'silent': [1],
              'subsample': [0.7, 0.8,0.9,0.6,0.5],
              'colsample_bytree': [0.5,0.6,0.7, 0.8, 0.9],
              'n_estimators': [500, 1000, 200, 100, 800]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 15,
                        verbose=True)

xgb_grid.fit(x_train_full,
         y_train_full)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)