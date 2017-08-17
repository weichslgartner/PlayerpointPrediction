# coding=utf-8
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.kernel_ridge import KernelRidge
import pandas as pd

kicker_data_raw = pd.read_csv('../data/player_list.csv', delimiter=";",decimal=",")
kicker_data_clean = kicker_data_raw.dropna()
#print(kicker_data_clean)
kicker_data_clean = pd.get_dummies(kicker_data_clean,columns = ['Verein', 'Pos'])

print(kicker_data_clean)
#print(sorted(kicker_data_raw.Verein.unique()))

#todo: merge with kicker data
whoscored_data = pd.read_csv('../data/whoscored.csv', delimiter=";", encoding="utf8")
whoscored_data.tm.replace(['Bayern Munich' , 'RasenBallsport Leipzig' , 'Borussia Dortmund', 'FC Cologne', 'Hertha Berlin' , 'Werder Bremen',
 'Borussia M.Gladbach', 'Schalke 04', 'Eintracht Frankfurt',
 'Bayer Leverkusen'  , 'Hamburger SV', 'Mainz 05' ],
           ['Bayern' , 'Leipzig', 'Dortmund', 'Köln' , 'Hertha' , 'Bremen', 'Gladbach', 'Schalke', 'E. Frankfurt', 'Leverkusen', 'HSV', 'Mainz' ],inplace=True)


print(kicker_data_clean.dtypes)

#print(kicker_data_raw['Name']+ whoscored_data['name'])


y = kicker_data_clean['PI_2017'] 
X = pd.concat([kicker_data_clean.loc[:, 'Verein_Augsburg':'Pos_TOR'] ,kicker_data_clean['Preis'], kicker_data_clean['PI_2016']], axis=1) 
#print(X)
print(X.dtypes )


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
#est = linear_model.SGDRegressor()


est = SVR(kernel='rbf', C=1e3, gamma=0.1)
est =  KernelRidge(alpha=1.0)

est = est.fit(X_train, y_train)
y_pred = est.predict(X_test)
print("MSE (60/40 test split) %f" % mean_squared_error(y_test, y_pred))
print( pd.concat( [kicker_data_raw.iloc[y_test.index.values]['Name'],kicker_data_raw.iloc[y_test.index.values]['PI_2016'],kicker_data_raw.iloc[y_test.index.values]['PI_2017'],pd.DataFrame(index = y_test.index, data =y_pred, dtype=np.float64, columns=[ "Pred. PI_2017"])], axis=1))

predicted_cross_val = cross_val_predict(est, X, y, cv=10)
print("MSE cross validation %f" % mean_squared_error(y, predicted_cross_val))
print(predicted_cross_val)

print( pd.concat( [kicker_data_clean['Name'],kicker_data_clean['PI_2016'],kicker_data_clean['PI_2017'],pd.DataFrame(index = kicker_data_clean.index,data =predicted_cross_val, dtype=np.float64, columns=[ "Pred. PI_2017"])], axis=1))


kicker_data_2017_raw = pd.read_csv('../data/spieler2017.csv', delimiter=";",decimal=",",encoding="latin1")
kicker_data_2017_clean = kicker_data_2017_raw.dropna()
kicker_data_2017_clean.Verein.replace(["Hannover","Stuttgart"],["Ingolstadt","Darmstadt"])
print(kicker_data_2017_clean)

    
kicker_data_2017_clean = pd.get_dummies(kicker_data_2017_clean,columns = ['Verein', 'Pos'])
X_2017 = pd.concat([kicker_data_2017_clean.loc[:, 'Verein_Augsburg':'Pos_TOR'] ,kicker_data_2017_clean['Preis'], kicker_data_2017_clean['PI_2016']], axis=1) 
y_pred_2017 = est.predict(X_2017)
prediction = pd.concat( [kicker_data_2017_clean['Name'],kicker_data_2017_raw['Verein'],pd.DataFrame(index = kicker_data_2017_clean.index,data =y_pred_2017, dtype=np.float64, columns=[ "Pred. PI_2018"])], axis=1)
prediction = prediction.sort_values(["Name"]).dropna()
prediction["Pred. PI_2018"] = prediction["Pred. PI_2018"].map('{:,.0f}'.format)
prediction.to_csv("../data/prediction.csv", sep=";", index=False)