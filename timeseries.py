__author__ = 'Liana'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as st
import numpy as np
import statsmodels.tsa.arima_process as ap
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.api import qqplot

# Important: It might be necessary to install xlrd
# pip install xlrd

# Download data from: http://www.eia.gov/dnav/pet/pet_pri_spt_s1_m.htm
# Create an Excel file object
excel = pd.ExcelFile('data/PET_PRI_SPT_S1_M.xls' )

# Parse the first sheet
df = excel.parse(excel.sheet_names[1])

# Rename the columns
df = df.rename(columns=dict(zip(df.columns, ['Date','WTI','Brent'])))

# Cut off the first 18 rows because these rows
# contain NaN values for the Brent prices
df = df[18:]

#print df.head()

# Index the data set by date
df.index = df['Date']

# Remove the date column after re-indexing
df = df[['WTI','Brent']]

#print df

#===========================
#      VISUALISATION
#===========================

# Use seaborn to control figure aesthetics
sns.set_style("darkgrid")  # try "dark", "whitegrid"

fig, ax = plt.subplots(figsize=(10,5))
plt.title('Crude Oil Prices')
plt.xlabel('Year')
plt.ylabel('Price [USD]')
ax.plot(df.index, df['WTI'], c='b', label='WTI')
ax.plot(df.index, df['Brent'], c='r', label='Brent')
plt.legend(loc='upper left')
plt.show()


df[-11*4:].to_csv('data/Spot_Prices_2012_2015.csv')

newdf = pd.read_csv('data/Spot_Prices_2012_2015.csv')

dates = pd.Series([pd.to_datetime(date) for date in newdf['Date']])
fig, ax = plt.subplots(figsize=(10,5))
plt.title('Crude Oil Prices')
plt.xlabel('Year')
plt.ylabel('Price [USD]')
ax.plot(dates, newdf['WTI'], c='b', label='WTI')    #np.log()
plt.legend(loc='upper left')
plt.show()

#print newdf.head()


#===========================
#    TIME SERIES ANALYSIS
#===========================

# Building ARIMA model

from statsmodels.tsa.base.datetools import dates_from_range

trainWTI = newdf[:int(0.95*len(newdf))]

# 2012m12 means to start counting months from the 12th month of 2012
# To know the starting month, print trainWTI.head()
dates1 = dates_from_range('2012m1', length=len(trainWTI.WTI))
trainWTI.index = dates1
trainWTI = trainWTI[['WTI']]

print trainWTI.tail()

# Determine whether AR or MA terms are needed to correct any
# autocorrelation that remains in the series.
# Looking at the autocorrelation function (ACF) and partial autocorrelation (PACF) plots of the series,
# it's possible to identify the numbers of AR and/or MA terms that are needed
# In this example, the autocorrelations are significant for a large number of lags,
# but perhaps the autocorrelations at lags 2 and above are merely due to the propagation of the autocorrelation at lag 1.
# This is confirmed by the PACF plot.
# RULES OF THUMB:
# Rule 1: If the PACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is positive,
# then consider adding an AR term to the model. The lag at which the PACF cuts off is the indicated number of AR terms.
# Rule 2: If the ACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative,
# then consider adding an MA term to the model. The lag at which the ACF cuts off is the indicated number of MA terms.
fig1 = sm.graphics.tsa.plot_acf(trainWTI['WTI'])
ax = fig1.add_subplot(111)
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
plt.show()

fig2 = sm.graphics.tsa.plot_pacf(trainWTI['WTI'])
ax = fig2.add_subplot(111)
ax.set_xlabel("Lag")
ax.set_ylabel("PACF")
plt.show()


# Parameter freq indicates that monthly statistics is used
arima_mod100 = ARIMA(trainWTI, (2,0,0), freq='M').fit()  # try (1,0,1)
print arima_mod100.summary()

# Check assumptions:
# 1) The residuals are not correlated serially from one observation to the next.
# The Durbin-Watson Statistic is used to test for the presence of serial correlation among the residuals
# The value of the Durbin-Watson statistic ranges from 0 to 4.
# As a general rule of thumb, the residuals are uncorrelated is the Durbin-Watson statistic is approximately 2.
# A value close to 0 indicates strong positive correlation, while a value of 4 indicates strong negative correlation.
print "==================== Durbin-Watson ====================="
print sm.stats.durbin_watson(arima_mod100.resid.values)
print "========================================================"

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax = arima_mod100.resid.plot(ax=ax)
ax.set_title("Residual series")
plt.show()

resid = arima_mod100.resid

print "============== Residuals normality test ================"
print st.normaltest(resid)
print "========================================================"

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_title("Residuals test for normality")
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.show()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax = trainWTI.ix['2012':].plot(ax=ax)
fig = arima_mod100.plot_predict('2014m1', '2015m12', dynamic=True, ax=ax, plot_insample=False)
ax.set_title("Prediction of spot prices")
ax.set_xlabel("Dates")
ax.set_ylabel("Price [USD]")
plt.show()