import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier, BayesianRidge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from six import StringIO
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from numpy.ma.core import sqrt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from numpy.polynomial.polynomial import polyfit
from sklearn.metrics import accuracy_score,classification_report,mean_squared_error,r2_score,mean_absolute_error, confusion_matrix
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Importing The Dataset
import os
current_path = os.getcwd()
print("Current Path:", current_path)
df= pd.read_csv(current_path+'/Dataset/StudentInfo.csv')

df.drop(df[df['G3'] < 1].index, inplace = True)
#onehot encoding
df_ohe = pd.get_dummies(df, drop_first=True)

THRESHOLD = 0.1  # Adjust the threshold as needed
G3_corr = df_ohe.corr()["G3"]
df_ohe_after_drop_features = df_ohe.copy()
for key, value in G3_corr.items():
    if abs(value) < THRESHOLD:
        df_ohe_after_drop_features.drop(columns=key, inplace=True)

X = df_ohe_after_drop_features.drop('G3',axis = 1)
y = df_ohe_after_drop_features['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
print("G3: ", X_train.columns)
model_xgbr = GradientBoostingRegressor()
model_xgbr.fit(X_train, y_train)
# Make predictions on the test set
y_test_pred = model_xgbr.predict(X_test)
# Calculate evaluation metrics
mse_xgbr = mean_squared_error(y_test, y_test_pred)
mae_xgbr = mean_absolute_error(y_test, y_test_pred)
rmse_xgbr = np.sqrt(mse_xgbr)
r_squared_xgbr = r2_score(y_test, y_test_pred)
# Cross-validation
scores = cross_val_score(model_xgbr, X_test, y_test, scoring='neg_mean_squared_error', cv=5)
rmse_cross_val = np.sqrt(-scores.mean())
print("G3 MSE: ", mse_xgbr )
print("G3 MAE: ", mae_xgbr )
print("G3 RMSE: ", rmse_xgbr )
print("G3 Rsquare: ", r_squared_xgbr )

pickle.dump(model_xgbr, open("model_G3.pkl", 'wb'))

#------------------------------------------------------------------------------------#

df2= pd.read_csv(current_path+'/Dataset/StudentInfo.csv')

df2.drop(df2[df2['G2'] < 1].index, inplace = True)
#onehot encoding
df_ohe2 = pd.get_dummies(df2, drop_first=True)

THRESHOLD = 0.09  # Adjust the threshold as needed
G2_corr = df_ohe2.corr()["G2"]
df_ohe_after_drop_features2 = df_ohe2.copy()
for key, value in G2_corr.items():
    if abs(value) < THRESHOLD:
        df_ohe_after_drop_features2.drop(columns=key, inplace=True)

X = df_ohe_after_drop_features2.drop(['G2','G3'],axis = 1)
y = df_ohe_after_drop_features2['G2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
print("G2: ", X_train.columns)
model_lr2 = LinearRegression()
model_lr2.fit(X_train, y_train)
# Make predictions on the test set
y_test_pred2 = model_lr2.predict(X_test)
# Calculate evaluation metrics
mse_lr2 = mean_squared_error(y_test, y_test_pred2)
mae_lr2 = mean_absolute_error(y_test, y_test_pred2)
rmse_lr2 = np.sqrt(mse_lr2)
r_squared_lr2 = r2_score(y_test, y_test_pred2)
# Cross-validation
scores2 = cross_val_score(model_lr2, X_test, y_test, scoring='neg_mean_squared_error', cv=5)
rmse_cross_val = np.sqrt(-scores2.mean())
print("G2 MSE: ", mse_lr2 )
print("G2 MAE: ", mae_lr2 )
print("G2 RMSE: ", rmse_lr2 )
print("G2 Rsquare: ", r_squared_lr2 )

pickle.dump(model_lr2, open("model_G2.pkl", 'wb'))

#-------------------------------------------------------------#
# df3= pd.read_csv(current_path+'/Dataset/StudentInfo.csv')

# df3.drop(df3[df3['G1'] < 1].index, inplace = True)
# #onehot encoding
# df_ohe3 = pd.get_dummies(df3, drop_first=True)

# THRESHOLD = 0.1  # Adjust the threshold as needed
# G3_corr = df_ohe3.corr()["G1"]
# df_ohe_after_drop_features3 = df_ohe3.copy()
# for key, value in G3_corr.items():
#     if abs(value) < THRESHOLD:
#         df_ohe_after_drop_features3.drop(columns=key, inplace=True)

# X = df_ohe_after_drop_features3.drop(['G1','G3'],axis = 1)
# y = df_ohe_after_drop_features3['G1']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
# print("G1: ", X_train.columns)
# model_xgbr3 = GradientBoostingRegressor()
# model_xgbr3.fit(X_train, y_train)
# # Make predictions on the test set
# y_test_pred = model_xgbr3.predict(X_test)
# # Calculate evaluation metrics
# mse_xgbr3 = mean_squared_error(y_test, y_test_pred)
# mae_xgbr3 = mean_absolute_error(y_test, y_test_pred)
# rmse_xgbr3 = np.sqrt(mse_xgbr3)
# r_squared_xgbr3 = r2_score(y_test, y_test_pred)
# # Cross-validation
# scores = cross_val_score(model_xgbr3, X_test, y_test, scoring='neg_mean_squared_error', cv=5)
# rmse_cross_val = np.sqrt(-scores.mean())
# # print("G1 MSE: ", mse_xgbr3 )
# # print("G1 MAE: ", mae_xgbr3 )
# # print("G1 RMSE: ", rmse_xgbr3)
# # print("G1 Rsquare: ", r_squared_xgbr3 )
# pickle.dump(mse_xgbr3, open("model_G1.pkl", 'wb'))
