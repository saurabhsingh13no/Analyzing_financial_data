from Dataset import *
from Logger import *
"""Exploratory Data Analysis Class"""
class EDA(Dataset):

    """Initialize instance variables"""
    def __init__(self,filename=None):
        Dataset.__init__(self,filename)
        Dataset.extractData(self)

    """Run the EDA"""
    def runAnalysis(self):
        cd=self.cd


        # Plotting the stock market closing index
        cd.plot(figsize=(20, 15), title="Plot of Index Value v/s Time")
        plt.xlabel("Date")
        plt.ylabel("Index Value")
        plt.show()


        # Plotting autocorrelation
        column_names = cd.columns.values
        plt.figure(figsize=(20, 10))
        for i in range(0, len(column_names)):
            autocorrelation_plot(cd[column_names[i]], label=column_names[i])
        plt.title("Autocorrelation Plot - 1")


        # scaling the closing value of features for better analysis
        column_names = cd.columns.values + "_scaled"
        for i in range(0, len(column_names)):
            cd[column_names[i]] = cd.iloc[:, i] / max(cd.iloc[:, i])
        cd = cd.iloc[:, len(column_names):]


        # Transerfing cd dataframe into external file for analysis into R
        cd.to_csv('closing_date_scaled')


        # Plotting the scaled value of stock market closing index
        cd.plot(figsize=(20, 10), title="Plot of scaled Index Value v/s Time")
        plt.xlabel("Date")
        plt.ylabel("Index Value")
        plt.show()


        # Draw a matrix of scatter plots among themselves
        scatter_matrix(cd, figsize=(20, 20), diagonal='kde')
        plt.show()


        # creating a dataframe with log returns of closed indexes
        log_return_data = pd.DataFrame()
        for i in range(0, len(column_names)):
            log_return_data[column_names[i]] = np.log(cd.iloc[:, i] / cd.iloc[:, i].shift())


        # Plotting log_return_data to identify any correlation among
        #       themselves
        log_return_data.plot(figsize=(15, 5))
        plt.title("Log_return_data v/s Date")
        plt.xlabel("Date")
        plt.ylabel("Log return data")
        plt.show()


        # Plotting the autocorrelation plot of log returns
        column_names = log_return_data.columns.values
        plt.figure(figsize=(20, 10))
        for i in range(0, len(column_names)):
            autocorrelation_plot(log_return_data[column_names[i]], label=column_names[i])
        plt.legend(loc='upper right')
        plt.title("Autocorrelation plot of log returns")
        plt.show()


        # Draw a matrix of scatter plots of log_return_data
        scatter_matrix(log_return_data, figsize=(20, 20), diagonal='kde')
        plt.show()

        new_column = column_names[0] + "_positive"
        log_return_data[new_column] = 0
        log_return_data.loc[log_return_data.iloc[:, 0] >= 0, new_column] = 1

        self.setLogReturnData(log_return_data)

