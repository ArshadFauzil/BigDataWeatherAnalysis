from pyspark.sql import SparkSession
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

# df1 = pd.read_csv("DelhiWeather.csv")

spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('DelhiWeather.csv', header = True, inferSchema = True)
# df.printSchema()

df = df.select('datetime_utc', '_dewptm', '_pressurem', '_tempm')

# data = sc.textFile("DelhiWeather.csv") \
#     .map(lambda line: line.split(",")) \
#     .filter(lambda line: len(line) > 5)
#
# header = data.first()
#
# # If the line in the dataset is not the header, then the date is converted to a datetime object
# data = data.map(lambda line: (line[0], line[2], line[8], line[11]) if line == header
#                 else (datetime.strptime(line[0], '%Y%m%d-%H:%M'), float(line[2]), float(line[8]), float(line[11]))).toDF()
#
# # sdf = sc.parallelize(data).toDF()

df = df.toPandas()
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

# df = df.groupby('datetime_utc')
df = df.set_index('datetime_utc')

df = df.replace('', np.nan, regex=True)
# df = df.apply(lambda x: x.str.strip()).replace('', np.nan)
df = df.fillna(method='ffill')     #Fills the last valid observation to the next missing value

# the dataset is resampled to monthly data where the mean of the
# respective month is taken for the monthly values. Also null values are filled with
# most recent previous observation.
y = df['_tempm'].resample('MS').mean().ffill()                # change argument of resample() to 'D' for daily analysis
# y.plot(figsize=(15, 6))
# print(df.info(verbose=True))
# print(y['2000':])

# rcParams['figure.figsize'] = 18, 8
#
# decomposition = sm.tsa.seasonal_decompose(y, model='additive')
# fig = decomposition.plot()
# plt.show()
# print(df.loc['1996-11-08 02:00:00 '])
# print(y.isnull().sum().sum())

# Finding the optimal parameter(p,d,q) combination by finding the minimum AIC value
# p = d = q = range(0, 2)
# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
#
# print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
# print()
#
# min_aic = []
# for i, param in enumerate(pdq):
#     for j, param_seasonal in enumerate(seasonal_pdq):
#         try:
#             mod = sm.tsa.statespace.SARIMAX(y,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#
#             results = mod.fit()
#
#             if i == 0 and j == 0:
#                 min_aic.append(param)
#                 min_aic.append(param_seasonal)
#                 min_aic.append(results.aic)
#             else:
#                 if results.aic < min_aic[2]:
#                     min_aic[0] = param
#                     min_aic[1] = param_seasonal
#                     min_aic[2] = results.aic
#
#             if i == len(pdq)-1 and j == len(seasonal_pdq)-1:
#                 print('ARIMA{}x{}12 - MIN_AIC:{}'.format(min_aic[0], min_aic[1], min_aic[2]))
#
#         except:
#             continue

# Parameter Config for daily predictions
# mod = sm.tsa.statespace.SARIMAX(y,
#                                 order=(1, 1, 1),
#                                 seasonal_order=(0, 0, 1, 12),
#                                 enforce_stationarity=False,
#                                 enforce_invertibility=False)

# Parameter Config for monthly/weekly predictions
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 0, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

# print(results.summary().tables[1])
#
# results.plot_diagnostics(figsize=(16, 8))
# plt.show()

# predictions of the model for 2017 against the actual values
# pred = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False)
# pred_ci = pred.conf_int()
#
# ax = y['2014':].plot(label='observed')
# pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
#
# ax.fill_between(pred_ci.index,
#                 pred_ci.iloc[:, 0],
#                 pred_ci.iloc[:, 1], color='k', alpha=.2)
#
# ax.set_xlabel('Date')
# ax.set_ylabel('Temperature')
# plt.legend()
#
# plt.show()
#
# y_forecasted = pred.predicted_mean
# y_truth = y['2016-01-01':]
#
# mse = ((y_forecasted - y_truth) ** 2).mean()
# print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#
# print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# Forecast for the future period after April 2017
pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()

ax = y['2014':].plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')
ax.set_ylabel('Temperature')

plt.legend()
plt.show()

