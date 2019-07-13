import calendar
import pandas as pd
import numpy as np
np.set_printoptions(precision=3)
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
import matplotlib.pyplot as plt
df = pd.read_csv("train_cab.csv")
#df.head()
df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC',errors='coerce' )
df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors='coerce')

#df.pickup_datetime = df.pickup_datetime.drop
#df = df[((df['pickup_longitude'] > -78) & (df['pickup_longitude'] < -70)) & ((df['dropoff_longitude'] > -78) & (df['dropoff_longitude'] < -70)) & ((df['pickup_latitude'] > 37) & (df['pickup_latitude'] < 45))
#        & ((df['dropoff_latitude'] > 37) & (df['dropoff_latitude'] < 45))]

df = df[df["pickup_datetime"] < df["pickup_datetime"].max()]
df = df[df["pickup_datetime"] > df["pickup_datetime"].min()]
df.reset_index(inplace=True)
df = df.drop(['index'], axis= 1)

df['pickup_day'] = df['pickup_datetime'].apply(lambda x: x.day)
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
df['pickup_day_of_week'] = df['pickup_datetime'].apply(lambda x: calendar.day_name[x.weekday()])
df['pickup_month'] = df['pickup_datetime'].apply(lambda x: x.month)
df['pickup_year'] = df['pickup_datetime'].apply(lambda x: x.year)

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

df["Distance_km"] = X

len(df.loc[df["passenger_count"] > 8])
df = df[df["Distance_km"] < 50]
df = df[df["Distance_km"] > 0 ]
df = df[df["passenger_count"] >= 1]
df = df[df["passenger_count"] <= 8]
df = df.loc[df["fare_amount"] >= 1]
df = df.loc[df["fare_amount"] < 200]

print(df.describe())



#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imputer = imputer.fit(df[["fare_amount"]])
#df[["fare_amount"]] = imputer.transform(df[["fare_amount"]])

#imputer1 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
#imputer = imputer1.fit(df[["passenger_count"]])
#df[["passenger_count"]] = imputer1.transform(df[["passenger_count"]])

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['pickup_day_of_week'].drop_duplicates())
df['pickup_day_of_week'] = encoder.transform(df['pickup_day_of_week'])
#encoder.fit(df['pickup_year'].drop_duplicates())
#df['pickup_year'] = encoder.transform(df['pickup_year'])

df_corr = df.drop(["pickup_datetime"], axis= 1)

corr = df_corr.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.05:
            if columns[j]:
                columns[j] = False
selected_columns = df_corr.columns[columns]
data = df_corr[selected_columns]

selected_columns = selected_columns[1:].values

import statsmodels.formula.api as sm
x = (df_corr.iloc[:,5:])
Y = (df_corr.iloc[:, 0])
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues)

        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())

    return x
SL = 0.05
data_modeled = backwardElimination(x.values, SL )

from sklearn.model_selection import train_test_split

#colsToDrop = df[['fare_amount']]
#X = df.drop(colsToDrop, axis=1)
#new = df_corr.drop(["pickup_latitude","dropoff_latitude","passenger_count","pickup_month","pickup_day_of_week"], axis = 1)
X = df_corr.iloc[:,9:]
y = df_corr[["fare_amount"]]
#X =X.to_frame()
#y = y.to_frame()
df_train, df_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
#X_pass = df_train.values
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(df_train, y_train)
#X_test = df_test[[""]]
y_pred = reg.predict(df_test)

from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test, y_pred)
print(score)
from sklearn.metrics import mean_absolute_error
score1 = mean_absolute_error(y_test, y_pred)
print(score1)

#from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
rf = LinearSVR(random_state=0, tol=1e-5)
#rf = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=200)
rf.fit(df_train, y_train)

y_dec_pred = rf.predict(df_test)

from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test, y_dec_pred)
print("DECISSION rmse" , score)
from sklearn.metrics import mean_absolute_error
score1 = mean_absolute_error(y_test, y_dec_pred)
print("MAPE " , score1)

#plt.plot(y_test, 'o-', color="r")
#plt.plot(y_dec_pred, 'o-', color="b")
#plt.plot(y_pred, 'o-', color="g")
#plt.show()



