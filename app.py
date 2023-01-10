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

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

df1 = pd.read_csv('NAMtotal_s.csv')
df1['ds'] = pd.to_datetime(df1["ds"]).dt.tz_localize(None)
data = df1.rename(columns={'ds':'Date','y':'target'})
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
x_train = train[["GDPC1", "Inflation", "month", "season"]]; y_train = train[["target"]]
x_test = test[["GDPC1", "Inflation",  "month", "season"]]; y_test = test[["target"]]
#########################
lr = LinearRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_train.append(x_test))
metric_lr = report_metric(pred_lr, y_train.append(y_test), "Linear Regression")
#########################
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb.fit(x_train, y_train)
pred_xgb = xgb.predict(x_train.append(x_test))
metric_xgb = report_metric(pred_xgb, y_train.append(y_test), "XGB Regression")
#########################
mode = 'additive'
m1 = Prophet(seasonality_mode=mode, daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=5);
m1 = m1.add_seasonality(name='quarterly', period=365.24/4, fourier_order=5)
m1.add_regressor(name='Inflation',
                  prior_scale=10,
                  standardize='auto',
                  mode=mode)
m1.add_regressor(name='GDPC1',
                  prior_scale=10,
                  standardize='auto',
                  mode=mode)
m1.fit(df1)
forecast = m1.predict()
df1_cv = cross_validation(m1,
                      horizon='30 days', #
                      period='30 days', #
                      initial='1260 days')
# df1_p = performance_metrics(df1_cv)
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
plot_preds(data["Date"],data["Date"], data["target"], pred_lr)

st.write("Model 2 works with XGB Regressor.")
st.write(metric_xgb)
plot_preds(data["Date"],data["Date"], data["target"], pred_xgb)

st.write("Model 3 works with Prophet.")
# st.write(df1_p)
plot_preds(df1["ds"], forecast["ds"], df1["y"], forecast["yhat"])


