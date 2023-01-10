import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
import streamlit as st

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('NAMtotal_s.csv').rename(columns={'index':'Date','y':'target'})
data['Date'] = pd.to_datetime(data["Date"])
data.sort_values(by=['Date'], inplace=True)

data['month'] = data['Date'].dt.month
data['season'] = (data['Date'].dt.month % 12 + 3) // 3

def report_metric(pred, test, model_name):
    mae = mean_absolute_error(pred, test)
    mse = mean_squared_error(pred, test)
    rmse = np.sqrt(mse)
    r2 = r2_score(test, pred)
    metric_data = {'Metric': ['MAE', 'RMSE', 'R2'], model_name: [mae, rmse, r2]}
    metric_df = pd.DataFrame(metric_data)
    return metric_df

def plot_preds(data_date,test_date, target, pred):
    fig = plt.figure(figsize=(20,10))
    plt.plot(data_date, target, label = 'Real')
    plt.plot(test_date, pred, label = 'Pred')
    plt.legend()
    st.pyplot(fig)

test_period = -10
test = data[test_period:]; train = data[:test_period]
x_trainm1 = train[["month", "season"]]; y_trainm1 = train[["target"]]
x_testm1 = test[["month", "season"]]; y_testm1 = test[["target"]]

lr = LinearRegression()
lr.fit(x_trainm1, y_trainm1)
m1pred = lr.predict(x_testm1)
metric1 = report_metric(m1pred, y_testm1, "Linear Regression")

page = st.sidebar.selectbox("""
Please select model""", ["Main Page", "Linear Regressor"])

if page == "Main Page":
    st.title("Hello, welcome to volume predictor!")
    st.write("""  
    - Date: date format time feature
    - col1: categorical feature 
    - col2: second categorical feature
    - col3: third categorical feature
    - target: target variable to be predicted
    """)
    st.write("Lets plot volume data!")
    st.line_chart(data[["Date", "target"]].set_index("Date"))
else: #if page == "Linear Regressor":
    st.title("Model 1: ")
    st.write("Model 1 works with linear regression as base model.")
    st.write("The columns it used are: col1, col2, col3, day_of_week, day_of_month, month, week_of_year, season")
    st.write(metric1)
    plot_preds(data["Date"],test["Date"], data["target"], m1pred)
