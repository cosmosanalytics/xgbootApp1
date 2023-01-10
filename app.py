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
# from fbprophet import Prophet

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('NAMtotal_s.csv').rename(columns={'index':'Date','y':'target'})
data['Date'] = pd.to_datetime(data["Date"])
data.sort_values(by=['Date'], inplace=True)

data['day_of_week'] = data['Date'].dt.dayofweek
data['day_of_month'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['week_of_year'] = data['Date'].dt.week
data['season'] = (data['Date'].dt.month % 12 + 3) // 3

# # Encode col1, col2, col3 variables.
# le = preprocessing.LabelEncoder()
# data['col1'] = le.fit_transform(data['col1'])
# data['col2'] = le.fit_transform(data['col2'])
# data['col3'] = le.fit_transform(data['col3'])

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

test_period = -3 
test = data[test_period:]
train = data[:test_period]

### Perpare for Model 1
x_trainm1 = train[["day_of_week", "day_of_month", "month", "week_of_year", "season"]]
# x_trainm1 = train[["col1", "col2", "col3", "day_of_week", "day_of_month", "month", "week_of_year", "season"]]
y_trainm1 = train[["target"]]

x_testm1 = test[["day_of_week", "day_of_month", "month", "week_of_year", "season"]]
# x_testm1 = test[["col1", "col2", "col3", "day_of_week", "day_of_month", "month", "week_of_year", "season"]]
y_testm1 = test[["target"]]

lr = LinearRegression()
lr.fit(x_trainm1, y_trainm1)
m1pred = lr.predict(x_testm1)
metric1 = report_metric(m1pred, y_testm1, "Linear Regression")

### Prepare for model 2
x_trainm2 = train[["day_of_week", "day_of_month", "month", "week_of_year", "season"]]
# x_trainm2 = train[["col1", "col2", "col3", "day_of_week", "day_of_month", "month", "week_of_year", "season"]]
y_trainm2 = train[["target"]]

x_testm2 = test[["day_of_week", "day_of_month", "month", "week_of_year", "season"]]
# x_testm2 = test[["col1", "col2", "col3", "day_of_week", "day_of_month", "month", "week_of_year", "season"]]
y_testm2 = test[["target"]]

xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb.fit(x_trainm2, y_trainm2)
m2pred = xgb.predict(x_testm2)
metric2 = report_metric(m2pred, y_testm2, "XGB Regression")

### Prepare for model 3 
x_trainm3 = train[["day_of_week", "day_of_month"]]
# x_trainm3 = train[["col1", "col3", "day_of_week", "day_of_month"]]
y_trainm3 = train[["target"]]

x_testm3 = test[["day_of_week", "day_of_month"]]
# x_testm3 = test[["col1", "col3", "day_of_week", "day_of_month"]]
y_testm3 = test[["target"]]

norm = MinMaxScaler().fit(x_trainm3)
x_train_normm3 = pd.DataFrame(norm.transform(x_trainm3))
x_test_normm3 = pd.DataFrame(norm.transform(x_testm3))

lgb_tune = LGBMRegressor(learning_rate=0.1,
                                max_depth=2,
                                min_child_samples=25,
                                n_estimators=100,
                                num_leaves=31)
lgb_tune.fit(x_train_normm3, y_trainm3)
m3pred = lgb_tune.predict(x_test_normm3)
metric3 = report_metric(m3pred, y_testm3, "LGBM Regression")

page = st.sidebar.selectbox("""
Please select model""", ["Main Page", "Linear Regressor"])
# page = st.sidebar.selectbox("""
# Please select model""", ["Main Page", "Linear Regressor", "XGB Regressor", "LGBM Regressor", "Compare Models"])

if page == "Main Page":
    ### INFO
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
# elif page == "XGB Regressor":
#     st.title("Model 2: ")
#     st.write("Model 2 works with XGB Regressor.")
#     st.write("The columns it used are: col1, col2, col3, day_of_week, day_of_month, month, week_of_year, season")
#     st.write(metric2)
#     plot_preds(data["Date"],test["Date"], data["target"], m2pred)
# elif page == "LGBM Regressor":
#     st.title("Model 3:")
#     st.write("Model 3 works with LGBM Regressor.")
#     st.write("The columns it used are: col1, col3, day_of_week, day_of_month")
#     st.write("We normalized and scaled data and we choose most important features.")
#     st.write()
#     st.write(metric3)
#     plot_preds(data["Date"],test["Date"], data["target"], m3pred)
# elif page == "Compare Models":
#     # Compare models.
#     st.title("Compare Models: ")
#     all_metrics = metric1.copy()
#     all_metrics["XGB Regression"] =  metric2["XGB Regression"].copy()
#     all_metrics["LGBM Regression"] = metric3["LGBM Regression"].copy()
#     st.write(all_metrics)

#     st.title("Best Model /XGB Regressor: ")
#     st.write("Lets plot best models predictions in detail.")
#     plot_preds(test["Date"],test["Date"], test["target"], m2pred)
#     st.write("Best Model Predictions vs Real")
#     best_pred = pd.DataFrame(test[["target"]].copy())
#     best_pred["pred"] = m2pred
#     st.write(best_pred)
