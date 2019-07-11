import calendar
import pandas as pd
import numpy as np
np.set_printoptions(precision=3)
import re
from datetime import timedelta
import datetime as dt
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
from math import sin, cos, sqrt, atan2, radians,asin
df = pd.read_csv("train_cab.csv")
df.head()
for i in range(len(df)):
    try:
       df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'][i],format='%Y-%m-%d %H:%M:%S UTC')

    except:
        df = df.drop(df.index[i], axis= 0)

df['pickup_day'] = df['pickup_datetime'].apply(lambda x: x.day)
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
df['pickup_day_of_week'] = df['pickup_datetime'].apply(lambda x: calendar.day_name[x.weekday()])
df['pickup_month'] = df['pickup_datetime'].apply(lambda x: x.month)
df['pickup_year'] = df['pickup_datetime'].apply(lambda x: x.year)
df['pickup_latitude_round3']=df['pickup_latitude'].apply(lambda x:round(x,3)).astype(float)
df['pickup_longitude_round3']=df['pickup_longitude'].apply(lambda x:round(x,3)).astype(float)
df['dropoff_latitude_round3']=df['dropoff_latitude'].apply(lambda x:round(x,3)).astype(float)
df['dropoff_longitude_round3']=df['dropoff_longitude'].apply(lambda x:round(x,3)).astype(float)

def distance(lat1, lat2, lon1,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
X = []
for i in range(len(df)):
    lat1 = df["pickup_latitude"][i]
    lat2 =df["dropoff_latitude"][i]
    lon1 =df["pickup_longitude"][i]
    lon2 =df["dropoff_longitude"][i]
    d = distance(lat1,lat2,lon1,lon2)
    X.append(d)
print(X[0:10])

df["Distance"] = X
from scipy import stats
df = df[(np.abs(stats.zscore(df[["pickup_latitude","dropoff_latitude","pickup_longitude","dropoff_longitude"]]))<3).all(axis=1)]
df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors='coerce')
df = df[df["passenger_count"] > 0]
df = df[df["passenger_count"] < 8]
df = df[df["Distance"] > 0]
print(df.info())

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(df[["fare_amount"]])
df[["fare_amount"]] = imputer.transform(df[["fare_amount"]])
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['pickup_day_of_week'].drop_duplicates())
df['pickup_day_of_week'] = encoder.transform(df['pickup_day_of_week'])
#imputer1 = Imputer(missing_values='NaN', strategy='mode', axis=0)
#imputer = imputer.fit(df[["passenger_count"]])
#df[["passenger_count"]] = imputer.transform(df[["passenger_count"]]).astype(int)

#pickup_fare_amount=df.groupby(['pickup_latitude_round3','pickup_longitude_round3'])['fare_amount'].mean().reset_index().rename(columns={'fare_amount':'avg_fare'})
X = df[["fare_amount"]]
from sklearn.model_selection import train_test_split

#colsToDrop = df[['fare_amount']]
#X = df.drop(colsToDrop, axis=1)
X = df.iloc[:,1:17]
y = df[["fare_amount"]]

df_train, df_test,y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=42)
X = df_train[["Distance", "passenger_count","pickup_year","pickup_month","pickup_day"]]
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y_train)
X_test = df_test[["Distance","passenger_count","pickup_year","pickup_month","pickup_day"]]
y_pred = reg.predict(X_test)

from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test, y_pred)
print(score)
from sklearn.metrics import mean_absolute_error
score1 = mean_absolute_error(y_test, y_pred)
print(score1)

from sklearn.svm import LinearSVR
dec = LinearSVR(random_state=0, tol=1e-5)
dec.fit(X, y_train)

y_dec_pred = dec.predict(X_test)

from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test, y_dec_pred)
print("DECISSION rmse" , score)
from sklearn.metrics import mean_absolute_error
score1 = mean_absolute_error(y_test, y_dec_pred)
print("MAPE " , score1)

'''
#colsToDrop = df_train[['pickup_datetime']]
Xl = df_train.iloc[:,1:16]
yl = y_train
import statsmodels.formula.api as sm

#X_opt =  X
regressor_OLS = sm.OLS (endog = yl, exog = Xl).fit()
print(regressor_OLS.summary())


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(yl, x).fit()
        maxVar = max(regressor_OLS.pvalues)

        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())

    return x


SL = 0.05
X_opt = Xl
X_modeled = backwardElimination(X_opt.values, SL)
print(X_modeled)

'''
