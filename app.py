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

test_period = -3
test = data[test_period:]; train = data[:test_period]
x_train = train[["GDPC1", "Inflation", "month", "season"]]; y_train = train[["target"]]
x_test = test[["GDPC1", "Inflation", "month", "season"]]; y_test = test[["target"]]
# fit scaler on training data
norm = MinMaxScaler().fit(x_train)
x_train_norm = pd.DataFrame(norm.transform(x_train))
x_test_norm = pd.DataFrame(norm.transform(x_test))
#########################
lr = LinearRegression()
lr.fit(x_train_norm, y_train)
pred_lr = lr.predict(x_test_norm)
metric_lr = report_metric(pred_lr, y_test, "Linear Regression")
#########################
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb.fit(x_train_norm, y_train)
pred_xgb = xgb.predict(x_test_norm)
metric_xgb = report_metric(pred_xgb, y_test, "XGB Regression")
#########################
lgb = LGBMRegressor(learning_rate=0.1, max_depth=2, min_child_samples=25, n_estimators=100, num_leaves=31)
lgb.fit(x_train_norm, y_train)
pred_lgb = lgb.predict(x_test_norm)

metric_lgb = report_metric(pred_lgb, y_test, "LGBM Regression")
#########################
st.title("Hello, welcome to volume predictor!")
st.write("""  
- Date: date format time feature
- Real GDP
- Nominal GDP
- Inflation: = Nominal GDP/Real GDP - 1
- Volume: KG
""")

st.write("Model 1 works with linear regression as base model.")
st.write(metric_lr)
plot_preds(data["Date"],test["Date"], data["target"], pred_lr)

st.write("Model 2 works with XGB Regressor.")
st.write(metric_xgb)
plot_preds(data["Date"],test["Date"], data["target"], pred_xgb)

st.write("Model 3 works with LighGBM Regressor.")
st.write(metric_lgb)
plot_preds(data["Date"],test["Date"], data["target"], pred_lgb)

