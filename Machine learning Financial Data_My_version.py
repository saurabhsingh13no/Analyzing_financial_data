"""Code to analyse the Stock Market Index data by QuantUniversity
Credits : GOOGLE INC for providing the link to dataset and stubs of code
to get started

Date Created: June 8 2017
Author: QuantUniversity : SS
"""

# Libraries used in the code
import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot,scatter_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# Reading the dataset to for Data Anlaysis
matplotlib.style.use('ggplot')
cd=pd.read_pickle("./closing_data_pickle")
cd=cd.sort_index()

# Plotting the stock market closing index
# pd.concat([cd['snp_close'],
#   cd['nyse_close'],
#   cd['djia_close'],
#   cd['nikkei_close'],
#   cd['hangseng_close'],
#   cd['ftse_close'],
#   cd['dax_close'],
#   cd['aord_close']], axis=1).plot(figsize=(20, 15))
# plt.show()

# scaling the closing value of features
cd['snp_close_scaled'] = cd['snp_close'] / max(cd['snp_close'])
cd['nyse_close_scaled'] = cd['nyse_close'] / max(cd['nyse_close'])
cd['djia_close_scaled'] = cd['djia_close'] / max(cd['djia_close'])
cd['nikkei_close_scaled'] = cd['nikkei_close'] / max(cd['nikkei_close'])
cd['hangseng_close_scaled'] = cd['hangseng_close'] / max(cd['hangseng_close'])
cd['ftse_close_scaled'] = cd['ftse_close'] / max(cd['ftse_close'])
cd['dax_close_scaled'] = cd['dax_close'] / max(cd['dax_close'])
cd['aord_close_scaled'] = cd['aord_close'] / max(cd['aord_close'])

# transerfing cd dataframe into external file for analysis into R
cd.to_csv('closing_date_scaled')

# Plotting the sclaed value of stock market closing index
# pd.concat([cd['snp_close_scaled'],
#   cd['nyse_close_scaled'],
#   cd['djia_close_scaled'],
#   cd['nikkei_close_scaled'],
#   cd['hangseng_close_scaled'],
#   cd['ftse_close_scaled'],
#   cd['dax_close_scaled'],
#   cd['aord_close_scaled']], axis=1).plot(figsize=(20, 15))
#plt.show()



# Plotting autocorrelation
# fig = plt.figure()
# fig.set_figwidth(20)
# fig.set_figheight(15)

# autocorrelation_plot(cd['snp_close'], label='snp_close')
# autocorrelation_plot(cd['nyse_close'], label='nyse_close')
# autocorrelation_plot(cd['djia_close'], label='djia_close')
# autocorrelation_plot(cd['nikkei_close'], label='nikkei_close')
# autocorrelation_plot(cd['hangseng_close'], label='hangseng_close')
# autocorrelation_plot(cd['ftse_close'], label='ftse_close')
# autocorrelation_plot(cd['dax_close'], label='dax_close')
# autocorrelation_plot(cd['aord_close'], label='aord_close')
# plt.legend(loc='upper right')
# plt.show()

# scatter_matrix(pd.concat([cd['snp_close_scaled'],
#   cd['nyse_close_scaled'],
#   cd['djia_close_scaled'],
#   cd['nikkei_close_scaled'],
#   cd['hangseng_close_scaled'],
#   cd['ftse_close_scaled'],
#   cd['dax_close_scaled'],
#   cd['aord_close_scaled']], axis=1), figsize=(20, 20), diagonal='kde')
# plt.show()

# creating a dataframe with log returns of closed indexes
log_return_data = pd.DataFrame()
log_return_data['snp_log_return'] = np.log(cd['snp_close']/cd['snp_close'].shift())
log_return_data['nyse_log_return'] = np.log(cd['nyse_close']/cd['nyse_close'].shift())
log_return_data['djia_log_return'] = np.log(cd['djia_close']/cd['djia_close'].shift())
log_return_data['nikkei_log_return'] = np.log(cd['nikkei_close']/cd['nikkei_close'].shift())
log_return_data['hangseng_log_return'] = np.log(cd['hangseng_close']/cd['hangseng_close'].shift())
log_return_data['ftse_log_return'] = np.log(cd['ftse_close']/cd['ftse_close'].shift())
log_return_data['dax_log_return'] = np.log(cd['dax_close']/cd['dax_close'].shift())
log_return_data['aord_log_return'] = np.log(cd['aord_close']/cd['aord_close'].shift())

print (log_return_data.describe())

pd.concat([log_return_data['snp_log_return'],
  log_return_data['nyse_log_return'],
  log_return_data['djia_log_return'],
  log_return_data['nikkei_log_return'],
  log_return_data['hangseng_log_return'],
  log_return_data['ftse_log_return'],
  log_return_data['dax_log_return'],
  log_return_data['aord_log_return']], axis=1).plot(figsize=(20, 15))
plt.show()

# fig = plt.figure()
# fig.set_figwidth(20)
# fig.set_figheight(15)

# Plotting the autocorrelation plot of log returns
autocorrelation_plot(log_return_data['snp_log_return'], label='snp_log_return')
autocorrelation_plot(log_return_data['nyse_log_return'], label='nyse_log_return')
autocorrelation_plot(log_return_data['djia_log_return'], label='djia_log_return')
autocorrelation_plot(log_return_data['nikkei_log_return'], label='nikkei_log_return')
autocorrelation_plot(log_return_data['hangseng_log_return'], label='hangseng_log_return')
autocorrelation_plot(log_return_data['ftse_log_return'], label='ftse_log_return')
autocorrelation_plot(log_return_data['dax_log_return'], label='dax_log_return')
autocorrelation_plot(log_return_data['aord_log_return'], label='aord_log_return')
plt.legend(loc='upper right')
plt.show()

# Plotting scatter matrix for log_return_data
scatter_matrix(log_return_data, figsize=(20, 20), diagonal='kde')
plt.title("scatter matrix for log_return_data")
plt.show()






