import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import streamlit as st

header = st.container()
dataset = st.container()
Data_Preprocessing = st.container()
model_training = st.container()
model_plotten = st.container()


with header:
    st.title("CO2 Emissions USA")
    st.text("Dataset We will be using a dataset that encapsulates the carbon dioxide emissions generated from burning coal\n for producing electricity power in the United States of America between 1973 and 2016."
            )


with dataset:
    st.header("CO2 Emissions USA-Dataset")
    st.text("I found this Dataset on Kaggle.com\n")
    st.subheader("Dataset-Head:")
    data = pd.read_csv("../input/co2-dataset-in-usa/co2.csv")
    st.write(data.head())

    st.subheader("Dataset Description:")
    st.write(data.describe())

    st.subheader("CO2-Dataset-Shape:")
    st.write(data.shape)


with Data_Preprocessing:
    st.header("This is the data after preprocessing")
    data["Month"] = data.YYYYMM.astype(str).str[4:6].astype(float)
    data["Year"] = data.YYYYMM.astype(str).str[0:6].astype(float)
    data.drop(["YYYYMM"], axis=1, inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    st.subheader("Dataset-Head:")
    st.write(data.head())
    st.subheader("Dataset-Tail:")
    st.write(data.tail())

X=data.loc[:,["Month", "Year"]].values
y=data.loc[:,"Value"].values
data_dmatrix = xgb.DMatrix(X,label=y)
print(data_dmatrix)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


with model_training:
    st.subheader("Time to train the model!")
    st.text("Using XGBoost:")

    n_estimators, learningRate, subsample, colsample_bytree, max_depth, gamma = st.columns(6)

    estimator = n_estimators.slider("What should be the Estimator of the model?", min_value=1, max_value=1000, value=1000, step=10)

    learning_rate = learningRate.slider("What should be the Learning Rate of the model?", min_value=0.01, max_value=100.0, value=0.08, step=0.01)

    SubSample = subsample.slider("What should be the Sub Sample of the model?", min_value=0.01, max_value=100.0, value=0.75, step=0.01)

    ColsampleBytree = colsample_bytree.slider("What should be the Col-sample Bytree of the model?", min_value=1, max_value=100, value=1, step=1)

    MaxDepth = max_depth.slider("What should be the Max-Depth of the model?", min_value=1, max_value=100, value=7, step=1)

    Gamma = gamma.slider("What should be the Max-Depth of the model?", min_value=1, max_value=100, value=0, step=1)

    reg_mod = xgb.XGBRegressor(n_estimators=estimator,learning_rate=learning_rate,subsample=SubSample,colsample_bytree=ColsampleBytree,max_depth=MaxDepth,gamma=Gamma,)

    reg_mod.fit(X_train, y_train)

    scores = cross_val_score(reg_mod, X_train, y_train, cv=10)

    st.subheader("RESULTS:")

    st.write("Mean cross-validation score: %.2f" % scores.mean())
    reg_mod.fit(X_train, y_train)

    predictions = reg_mod.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    st.write("RMSE: %f" % (rmse))

    r2 = np.sqrt(r2_score(y_test, predictions))
    st.write("R_Squared Score : %f" % (r2))

st.set_option('deprecation.showPyplotGlobalUse', False)

with model_plotten:
    st.subheader("Time to plott the model!")

    plt.figure(figsize=(10, 5), dpi=80)
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="test")
    plt.plot(x_ax, predictions, label="predicted")
    plt.title("Carbon Dioxide Emissions - Test and Predicted data")
    plt.legend()
    st.pyplot(plt.show())

st.title("End......Thanks a lot")
