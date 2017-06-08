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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier,MLPRegressor

"""Super class defining dataset"""
class Dataset:

    """Initialize instance variables"""
    def __init__(self,filename=None):
        self.filename=filename
        self.cd=pd.DataFrame()
        self.log_return_data=pd.DataFrame()
        self.cd_pa=pd.DataFrame()
        self.training_test_data=pd.DataFrame()

    """Used to fetch data from the given filename"""
    def extractData(self,filename=None):
        if len(self.cd)>0:
            return self.cd
        if filename!=None:
            self.filename=filename
        filename=self.filename
        matplotlib.style.use('ggplot')
        cd = pd.read_pickle("./"+filename)
        cd = cd.sort_index()
        self.cd=cd
        return self.cd.copy()

    def getLogReturnData(self):
        return self.log_return_data

    def setLogReturnData(self,log_return_data):
        self.log_return_data=log_return_data

    def modify_closing_date(self):
        # Adding scaled columns in cd dataset
        cd=self.cd
        cd['snp_close_scaled'] = cd['snp_close'] / max(cd['snp_close'])
        cd['nyse_close_scaled'] = cd['nyse_close'] / max(cd['nyse_close'])
        cd['djia_close_scaled'] = cd['djia_close'] / max(cd['djia_close'])
        cd['nikkei_close_scaled'] = cd['nikkei_close'] / max(cd['nikkei_close'])
        cd['hangseng_close_scaled'] = cd['hangseng_close'] / max(cd['hangseng_close'])
        cd['ftse_close_scaled'] = cd['ftse_close'] / max(cd['ftse_close'])
        cd['dax_close_scaled'] = cd['dax_close'] / max(cd['dax_close'])
        cd['aord_close_scaled'] = cd['aord_close'] / max(cd['aord_close'])

        self.cd_pa = cd.copy()

    def extractData_helper(self):
        cd=Dataset.extractData(self)
        print (self.cd.head())

        self.modify_closing_date()

        # Creating another dataframe log_return_data
        log_return_data = pd.DataFrame()
        log_return_data['snp_log_return'] = \
            np.log(cd['snp_close'] / cd['snp_close'].shift())
        log_return_data['nyse_log_return'] = \
            np.log(cd['nyse_close'] / cd['nyse_close'].shift())
        log_return_data['djia_log_return'] = \
            np.log(cd['djia_close'] / cd['djia_close'].shift())
        log_return_data['nikkei_log_return'] = \
            np.log(cd['nikkei_close'] / cd['nikkei_close'].shift())
        log_return_data['hangseng_log_return'] = \
            np.log(cd['hangseng_close'] / cd['hangseng_close'].shift())
        log_return_data['ftse_log_return'] = \
            np.log(cd['ftse_close'] / cd['ftse_close'].shift())
        log_return_data['dax_log_return'] = \
            np.log(cd['dax_close'] / cd['dax_close'].shift())
        log_return_data['aord_log_return'] = \
            np.log(cd['aord_close'] / cd['aord_close'].shift())

        # Making these columns to do predictive data analysis
        log_return_data['snp_log_return_positive'] = 0
        log_return_data.ix[log_return_data['snp_log_return'] >= 0, 'snp_log_return_positive'] = 1
        log_return_data['snp_log_return_negative'] = 0
        log_return_data.ix[log_return_data['snp_log_return'] < 0, 'snp_log_return_negative'] = 1

        # Creating training_test_data dataframe to do logistic analysis
        training_test_data = pd.DataFrame(
            columns=[
                'snp_log_return_positive', 'snp_log_return_negative',
                'snp_log_return_1', 'snp_log_return_2', 'snp_log_return_3',
                'nyse_log_return_1', 'nyse_log_return_2', 'nyse_log_return_3',
                'djia_log_return_1', 'djia_log_return_2', 'djia_log_return_3',
                'nikkei_log_return_0', 'nikkei_log_return_1', 'nikkei_log_return_2',
                'hangseng_log_return_0', 'hangseng_log_return_1', 'hangseng_log_return_2',
                'ftse_log_return_0', 'ftse_log_return_1', 'ftse_log_return_2',
                'dax_log_return_0', 'dax_log_return_1', 'dax_log_return_2',
                'aord_log_return_0', 'aord_log_return_1', 'aord_log_return_2'])

        for i in range(7, len(log_return_data)):
            snp_log_return_positive = log_return_data['snp_log_return_positive'].iloc[i]
            snp_log_return_negative = log_return_data['snp_log_return_negative'].iloc[i]
            snp_log_return_1 = log_return_data['snp_log_return'].iloc[i - 1]
            snp_log_return_2 = log_return_data['snp_log_return'].iloc[i - 2]
            snp_log_return_3 = log_return_data['snp_log_return'].iloc[i - 3]
            nyse_log_return_1 = log_return_data['nyse_log_return'].iloc[i - 1]
            nyse_log_return_2 = log_return_data['nyse_log_return'].iloc[i - 2]
            nyse_log_return_3 = log_return_data['nyse_log_return'].iloc[i - 3]
            djia_log_return_1 = log_return_data['djia_log_return'].iloc[i - 1]
            djia_log_return_2 = log_return_data['djia_log_return'].iloc[i - 2]
            djia_log_return_3 = log_return_data['djia_log_return'].iloc[i - 3]
            nikkei_log_return_0 = log_return_data['nikkei_log_return'].iloc[i]
            nikkei_log_return_1 = log_return_data['nikkei_log_return'].iloc[i - 1]
            nikkei_log_return_2 = log_return_data['nikkei_log_return'].iloc[i - 2]
            hangseng_log_return_0 = log_return_data['hangseng_log_return'].iloc[i]
            hangseng_log_return_1 = log_return_data['hangseng_log_return'].iloc[i - 1]
            hangseng_log_return_2 = log_return_data['hangseng_log_return'].iloc[i - 2]
            ftse_log_return_0 = log_return_data['ftse_log_return'].iloc[i]
            ftse_log_return_1 = log_return_data['ftse_log_return'].iloc[i - 1]
            ftse_log_return_2 = log_return_data['ftse_log_return'].iloc[i - 2]
            dax_log_return_0 = log_return_data['dax_log_return'].iloc[i]
            dax_log_return_1 = log_return_data['dax_log_return'].iloc[i - 1]
            dax_log_return_2 = log_return_data['dax_log_return'].iloc[i - 2]
            aord_log_return_0 = log_return_data['aord_log_return'].iloc[i]
            aord_log_return_1 = log_return_data['aord_log_return'].iloc[i - 1]
            aord_log_return_2 = log_return_data['aord_log_return'].iloc[i - 2]
            training_test_data = training_test_data.append(
                {'snp_log_return_positive': snp_log_return_positive,
                 'snp_log_return_negative': snp_log_return_negative,
                 'snp_log_return_1': snp_log_return_1,
                 'snp_log_return_2': snp_log_return_2,
                 'snp_log_return_3': snp_log_return_3,
                 'nyse_log_return_1': nyse_log_return_1,
                 'nyse_log_return_2': nyse_log_return_2,
                 'nyse_log_return_3': nyse_log_return_3,
                 'djia_log_return_1': djia_log_return_1,
                 'djia_log_return_2': djia_log_return_2,
                 'djia_log_return_3': djia_log_return_3,
                 'nikkei_log_return_0': nikkei_log_return_0,
                 'nikkei_log_return_1': nikkei_log_return_1,
                 'nikkei_log_return_2': nikkei_log_return_2,
                 'hangseng_log_return_0': hangseng_log_return_0,
                 'hangseng_log_return_1': hangseng_log_return_1,
                 'hangseng_log_return_2': hangseng_log_return_2,
                 'ftse_log_return_0': ftse_log_return_0,
                 'ftse_log_return_1': ftse_log_return_1,
                 'ftse_log_return_2': ftse_log_return_2,
                 'dax_log_return_0': dax_log_return_0,
                 'dax_log_return_1': dax_log_return_1,
                 'dax_log_return_2': dax_log_return_2,
                 'aord_log_return_0': aord_log_return_0,
                 'aord_log_return_1': aord_log_return_1,
                 'aord_log_return_2': aord_log_return_2},
                ignore_index=True)

        training_test_data.describe()
        training_test_data = training_test_data.dropna()
        self.training_test_data=training_test_data.copy()

    def sample_data(self):
        training_test_data = self.training_test_data.copy()
        # Creating training and testing dataset
        predictors_tf = training_test_data[training_test_data.columns[2:]]
        # predictors_tf =t
        # raining_test_data[training_test_data.columns[[3,6,9,12,15,18]]]
        classes_tf = training_test_data[training_test_data.columns[:2]]

        # 80% of the training data
        training_set_size = int(len(training_test_data) * 0.8)
        test_set_size = len(training_test_data) - training_set_size

        training_predictors_tf = predictors_tf[:training_set_size]
        training_classes_tf = classes_tf[:training_set_size]
        test_predictors_tf = predictors_tf[training_set_size:]
        test_classes_tf = classes_tf[training_set_size:]
        return training_predictors_tf,training_classes_tf,\
               test_predictors_tf,test_classes_tf

    def sample_data_regression(self):
        # Creating dataframe for Linear Regression Analysis
        closing_data = self.cd_pa[self.cd_pa.columns[8:]]  # using
        # all scaled features

        # Sample train and test data : we are using 80% of the
        # dataset as training data
        predictors_tf = closing_data[closing_data.columns[1:]]
        classes_tf = closing_data[closing_data.columns[:1]]

        training_set_size = int(len(closing_data) * 0.8)
        test_set_size = len(closing_data) - training_set_size

        training_predictors_tf = predictors_tf[:training_set_size]
        training_classes_tf = classes_tf[:training_set_size]
        test_predictors_tf = predictors_tf[training_set_size:]
        test_classes_tf = classes_tf[training_set_size:]

        return training_predictors_tf, training_classes_tf, \
               test_predictors_tf, test_classes_tf




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
        k=pd.concat([cd['snp_close'],
                       cd['nyse_close'],
                       cd['djia_close'],
                       cd['nikkei_close'],
                       cd['hangseng_close'],
                       cd['ftse_close'],
                       cd['dax_close'],
                       cd['aord_close']], axis=1).\
            plot(figsize=(20, 15), title="Plot of Index Value v/s Time")
        k.set_xlabel("Date")
        k.set_ylabel("Index Value")
        plt.show()


        # scaling the closing value of features for better analysis
        cd['snp_close_scaled'] = cd['snp_close'] / max(cd['snp_close'])
        cd['nyse_close_scaled'] = cd['nyse_close'] / max(cd['nyse_close'])
        cd['djia_close_scaled'] = cd['djia_close'] / max(cd['djia_close'])
        cd['nikkei_close_scaled'] = cd['nikkei_close'] / max(cd['nikkei_close'])
        cd['hangseng_close_scaled'] = \
            cd['hangseng_close'] / max(cd['hangseng_close'])
        cd['ftse_close_scaled'] = cd['ftse_close'] / max(cd['ftse_close'])
        cd['dax_close_scaled'] = cd['dax_close'] / max(cd['dax_close'])
        cd['aord_close_scaled'] = cd['aord_close'] / max(cd['aord_close'])


        # Transerfing cd dataframe into external file for analysis into R
        cd.to_csv('closing_date_scaled')


        # Plotting the scaled value of stock market closing index
        k=pd.concat([cd['snp_close_scaled'],
          cd['nyse_close_scaled'],
          cd['djia_close_scaled'],
          cd['nikkei_close_scaled'],
          cd['hangseng_close_scaled'],
          cd['ftse_close_scaled'],
          cd['dax_close_scaled'],
          cd['aord_close_scaled']], axis=1).\
            plot(figsize=(20, 15),title="Plot of Index Value v/s Time")
        k.set_xlabel("Date")
        k.set_ylabel("Scaled Index Value")
        plt.show()


        # Plotting autocorrelation
        autocorrelation_plot(cd['snp_close'], label='snp_close')
        autocorrelation_plot(cd['nyse_close'], label='nyse_close')
        autocorrelation_plot(cd['djia_close'], label='djia_close')
        autocorrelation_plot(cd['nikkei_close'], label='nikkei_close')
        autocorrelation_plot(cd['hangseng_close'], label='hangseng_close')
        autocorrelation_plot(cd['ftse_close'], label='ftse_close')
        autocorrelation_plot(cd['dax_close'], label='dax_close')
        autocorrelation_plot(cd['aord_close'], label='aord_close')
        plt.legend(loc='upper right')
        plt.title("Autocorelation Plot of Stock Index")
        plt.show()


        # Draw a matrix of scatter plots among themselves
        scatter_matrix(pd.concat([cd['snp_close_scaled'],
          cd['nyse_close_scaled'],
          cd['djia_close_scaled'],
          cd['nikkei_close_scaled'],
          cd['hangseng_close_scaled'],
          cd['ftse_close_scaled'],
          cd['dax_close_scaled'],
          cd['aord_close_scaled']], axis=1), figsize=(20, 20), diagonal='kde')
        plt.title("Scatter matrix of all scaled variables with each other")
        plt.show()


        # creating a dataframe with log returns of closed indexes
        log_return_data = pd.DataFrame()
        log_return_data['snp_log_return'] = \
            np.log(cd['snp_close'] / cd['snp_close'].shift())
        log_return_data['nyse_log_return'] = \
            np.log(cd['nyse_close'] / cd['nyse_close'].shift())
        log_return_data['djia_log_return'] = \
            np.log(cd['djia_close'] / cd['djia_close'].shift())
        log_return_data['nikkei_log_return'] = \
            np.log(cd['nikkei_close'] / cd['nikkei_close'].shift())
        log_return_data['hangseng_log_return'] = \
            np.log(cd['hangseng_close'] / cd['hangseng_close'].shift())
        log_return_data['ftse_log_return'] = \
            np.log(cd['ftse_close'] / cd['ftse_close'].shift())
        log_return_data['dax_log_return'] = \
            np.log(cd['dax_close'] / cd['dax_close'].shift())
        log_return_data['aord_log_return'] = \
            np.log(cd['aord_close'] / cd['aord_close'].shift())


        # Plotting log_return_data to identify any correlation among
        #       themselves
        pd.concat([log_return_data['snp_log_return'],
                   log_return_data['nyse_log_return'],
                   log_return_data['djia_log_return'],
                   log_return_data['nikkei_log_return'],
                   log_return_data['hangseng_log_return'],
                   log_return_data['ftse_log_return'],
                   log_return_data['dax_log_return'],
                   log_return_data['aord_log_return']], axis=1).\
            plot(figsize=(20, 15),title="Log_return_data v/s Date")
        plt.xlabel("Date")
        plt.ylabel("Log return data")
        plt.show()


        # Plotting the autocorrelation plot of log returns
        autocorrelation_plot(log_return_data['snp_log_return'],
                             label='snp_log_return')
        autocorrelation_plot(log_return_data['nyse_log_return'],
                             label='nyse_log_return')
        autocorrelation_plot(log_return_data['djia_log_return'],
                             label='djia_log_return')
        autocorrelation_plot(log_return_data['nikkei_log_return'],
                             label='nikkei_log_return')
        autocorrelation_plot(log_return_data['hangseng_log_return'],
                             label='hangseng_log_return')
        autocorrelation_plot(log_return_data['ftse_log_return'],
                             label='ftse_log_return')
        autocorrelation_plot(log_return_data['dax_log_return'],
                             label='dax_log_return')
        autocorrelation_plot(log_return_data['aord_log_return'],
                             label='aord_log_return')
        plt.legend(loc='upper right')
        plt.title("Autocorrelation plot of log returns")
        plt.show()


        # Draw a matrix of scatter plots of log_return_data
        scatter_matrix(log_return_data, figsize=(20, 20), diagonal='kde')
        plt.title("Scatter matrix of log_return_data variables"+
                  " with each other")
        plt.show()

        log_return_data['snp_log_return_positive'] = 0
        log_return_data.ix[log_return_data['snp_log_return'] >= 0,
                           'snp_log_return_positive'] = 1
        log_return_data['snp_log_return_negative'] = 0
        log_return_data.ix[log_return_data['snp_log_return'] < 0,
                           'snp_log_return_negative'] = 1

        self.setLogReturnData(log_return_data)





class linear_model(Dataset):
    """Initialize instance variables"""

    def __init__(self, filename=None):
        Dataset.__init__(self, filename)
        Dataset.extractData(self)

    # Predicting the S&P index value using Linear Regression
    def linearRegression(self):
        if len(self.cd)<=0:
            Dataset.extractData(self)
            Dataset.modify_closing_date(self)

        if len(self.cd_pa)<=0:
            Dataset.modify_closing_date(self)

        # sampling dataset into training and testing
        [training_predictors_tf, training_classes_tf,
         test_predictors_tf, test_classes_tf] = \
            Dataset.sample_data_regression(self)

        # Step 1) Create Linear Regression object
        regr = LinearRegression()

        # Step 2) Use object created above to build the Linear
        # Regression model and train on the dataset
        model = regr.fit(training_predictors_tf.as_matrix(),
                         training_classes_tf.as_matrix()[:, 0])

        # Step 3) Predict on the test data using model created above
        predicted = regr.predict(test_predictors_tf.as_matrix())

        # Step 4) Calculate the accuracy achieved
        print ("\nRunning Linear Regression Analysis")
        print("Mean squared error: ",
              np.mean((predicted - test_classes_tf.as_matrix()[:, 0]) ** 2))

        SSE = np.sum((regr.predict(test_predictors_tf.as_matrix()) -
                      test_classes_tf.as_matrix()[:, 0]) ** 2)
        SST = np.sum((np.mean(training_classes_tf.as_matrix()[:, 0]) -
                      test_classes_tf.as_matrix()[:, 0]) ** 2)
        R2 = 1.0 - 1.0 * SSE / SST
        RMSE = np.sqrt(SSE / len(test_classes_tf.as_matrix()[:, 0]))

        print("R squared value: ", R2)
        print("Root Mean Squared Error: ", RMSE)

        # Step 5) Plot the  outputs
        plt.figure(figsize=(10, 5))
        plt.plot(pd.Series(range(0, len(test_classes_tf.as_matrix()[:, 0])))
                 , test_classes_tf.as_matrix()[:, 0],
                 color='red', label='Test Data', linewidth=2)
        plt.plot(pd.Series(range(0, len(test_classes_tf.as_matrix()[:, 0]))),
                 regr.predict(test_predictors_tf.as_matrix()), color='blue',
                 linewidth=2, label="Predicted Data")
        plt.xlabel("Data Points")
        plt.ylabel("snp_close_scaled")
        plt.legend(loc='upper left')
        plt.show()




    def extractData(self):
        if len(Dataset.getLogReturnData(self))>0:
            return
        else:
            Dataset.extractData_helper(self)

    # Predicting the S&P trend(whether it will go up or down)
    # using Logistic Regression
    def logistic_regression(self):

        if len(self.getLogReturnData())<=0:
            self.extractData()

        # sampling dataset into training and testing
        [training_predictors_tf,training_classes_tf,
         test_predictors_tf,test_classes_tf]=Dataset.sample_data(self)


        # Step 1) Create Logistic Regression object with personlized
        # arguments
        logistic = LogisticRegression()

        # Step 2) Use object created above to build the Logistic
        # Regression model and train on the dataset
        model = logistic.fit(training_predictors_tf.as_matrix(),
                             training_classes_tf.as_matrix()[:, 0])

        # Step 3) Predict on the test data using model created above
        predicted = logistic.predict(test_predictors_tf.as_matrix())

        # Step 4) Calculate the accuracy acheived
        temp = predicted == test_classes_tf.as_matrix()[:, 0]
        accuracy = len(temp[temp == True]) / len(temp)
        print("\nAccuracy using Logistic Regression Model = ", accuracy, "\n")

        # Or you can combine Step 3 and Step 4 into one using below state
        # print ("Accuracy using Logistic Regression Model = ",
        #        model.score(test_predictors_tf.as_matrix(),test_classes_tf.as_matrix()[:,0]))

        # Generating the classfication Report
        print("Classification Report : \n\n",
              metrics.classification_report(test_classes_tf.as_matrix()[:, 0], predicted))

        # Verifying model using 10 fold cross validation
        scores = cross_val_score(LogisticRegression(),
                                 training_predictors_tf.as_matrix(),
                                 training_classes_tf.as_matrix()[:, 0],
                                 scoring='accuracy', cv=10)
        print("Mean accuracy validated using Cross Validation: ",
              scores.mean())





class ensemble(Dataset):
    """Initialize instance variables"""

    def __init__(self, filename=None):
        Dataset.__init__(self, filename)
        Dataset.extractData(self)
        self.training_test_data = pd.DataFrame()

    def extractData(self):
        if len(Dataset.getLogReturnData(self))>0:
            return
        else:
            Dataset.extractData_helper(self)

    # Predicting the S&P trend(whether it will go up or down)
    # using RandomForestClassifier
    def randomForestClassifier(self):

        if len(self.getLogReturnData())<=0:
            self.extractData()

        # sampling dataset into training and testing
        [training_predictors_tf, training_classes_tf,
         test_predictors_tf, test_classes_tf] = Dataset.sample_data(self)

        # Step 1) Create Random Forest Classifier object with personlized
        # arguments
        clf = RandomForestClassifier(n_estimators=100)

        # Step 2) Use object created above to build the
        # RandomForestClassifier model and train on the dataset
        model = clf.fit(training_predictors_tf.as_matrix(),
                        training_classes_tf.as_matrix()[:, 0])

        # Step 3) Predict on the test data using model created above
        predicted = clf.predict(test_predictors_tf.as_matrix())

        # Step 4) Calculate the accuracy acheived
        temp = predicted == test_classes_tf.as_matrix()[:, 0]
        accuracy = len(temp[temp == True]) / len(temp)
        print("\nAccuracy using Random Forest Classifier Model = ",
              accuracy, "\n")

        # Or you can combine Step 3 and Step 4 into one using below state
        # print ("Accuracy using Random Forest Classifier Model = ",
        #        model.score(test_predictors_tf.as_matrix(),
        # test_classes_tf.as_matrix()[:,0]))

        # Generating the classfication Report
        print("Classification Report : \n\n",
              metrics.classification_report(test_classes_tf.as_matrix()[:, 0],
                                            predicted))

        # Verifying model using 10 fold cross validation
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=100),
            training_predictors_tf.as_matrix(),
            training_classes_tf.as_matrix()[:, 0],
            scoring='accuracy', cv=10)
        print("Mean accuracy validated using Cross Validation: ",
              scores.mean())

    # Predicting the S&P trend(whether it will go up or down)
    # using RandomForestRegressor
    def randomForestRegressor(self):

        if len(self.cd)<=0:
            Dataset.extractData(self)
            Dataset.modify_closing_date(self)

        if len(self.cd_pa)<=0:
            Dataset.modify_closing_date(self)

        # sampling dataset into training and testing
        [training_predictors_tf, training_classes_tf,
         test_predictors_tf, test_classes_tf] = \
            Dataset.sample_data_regression(self)

        # Step 1) Create Random Forest Regressor object with personlized
        # arguments
        rfr = RandomForestRegressor(n_estimators=1000)

        # Step 2) Use object created above to build the
        # RandomForestRegressor model and train on the dataset
        model = rfr.fit(training_predictors_tf.as_matrix(),
                        training_classes_tf.as_matrix()[:, 0])

        # Step 3) Predict on the test data using model created above
        predicted = rfr.predict(test_predictors_tf.as_matrix())

        # Step 4) Calculate the accuracy achieved
        print("Mean squared error: ",
              np.mean((predicted - test_classes_tf.as_matrix()[:, 0]) ** 2))

        SSE = np.sum((rfr.predict(test_predictors_tf.as_matrix()) -
                      test_classes_tf.as_matrix()[:, 0]) ** 2)
        SST = np.sum((np.mean(training_classes_tf.as_matrix()[:, 0]) -
                      test_classes_tf.as_matrix()[:, 0]) ** 2)
        R2 = 1.0 - 1.0 * SSE / SST
        RMSE = np.sqrt(SSE / len(test_classes_tf.as_matrix()[:, 0]))

        print("R squared value: ", R2)
        print("Root Mean Squared Error: ", RMSE)

        # Step 5) Plot the  outputs
        plt.figure(figsize=(10, 5))
        plt.plot(pd.Series(range(0, len(test_classes_tf.as_matrix()[:, 0]))),
                 test_classes_tf.as_matrix()[:, 0],
                 color='red', label='Test Data', linewidth=2)
        plt.plot(pd.Series(range(0, len(test_classes_tf.as_matrix()[:, 0]))),
                 predicted, color='blue',
                 linewidth=2, label="Predicted Data")
        plt.xlabel("Data Points")
        plt.ylabel("snp_close_scaled")
        plt.legend(loc='upper left')




class neural_network(Dataset):
    """Initialize instance variables"""

    def __init__(self, filename=None):
        Dataset.__init__(self, filename)
        Dataset.extractData(self)
        self.training_test_data = pd.DataFrame()

    def extractData(self):
        if len(Dataset.getLogReturnData(self)) > 0:
            return
        else:
            Dataset.extractData_helper(self)

    # Predicting the S&P trend(whether it will go up or down)
    # using Neural Network MLPClassfier
    def mlpClassifier(self):
        if len(self.getLogReturnData())<=0:
            self.extractData()

        # sampling dataset into training and testing
        [training_predictors_tf, training_classes_tf,
         test_predictors_tf, test_classes_tf] = Dataset.sample_data(self)

        # Step 1) Create Mulit-Layer Perceptron Classifier
        # object with personlized arguments
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(2,), random_state=1,
                            activation='logistic', max_iter=200)

        # Step 2) Use object created above to build the Neural Network
        # model and train on the dataset
        model = clf.fit(training_predictors_tf.as_matrix(),
                        training_classes_tf.as_matrix()[:, 0])

        # Step 3) Predict on the test data using model created above
        predicted = clf.predict(test_predictors_tf.as_matrix())

        # Step 4) Calculate the accuracy acheived
        temp = predicted == test_classes_tf.as_matrix()[:, 0]
        accuracy = len(temp[temp == True]) / len(temp)
        print("\nAccuracy using Neural Network Classifier Model = ",
              accuracy, "\n")

        # Or you can combine Step 3 and Step 4 into one using below state
        # print ("Accuracy using Neural Network Classifier Model = ",
        #        model.score(test_predictors_tf.as_matrix(),
        # test_classes_tf.as_matrix()[:,0]))

        # Generating the classfication Report
        print("Classification Report : \n\n",
              metrics.classification_report(test_classes_tf.as_matrix()[:, 0],
                                            predicted))

    # Predicting the S&P index value using Neural Network MLP Regressor
    def mlpRegressor(self):

        if len(self.cd)<=0:
            Dataset.extractData(self)
            Dataset.modify_closing_date(self)

        if len(self.cd_pa)<=0:
            Dataset.modify_closing_date(self)

        # sampling dataset into training and testing
        [training_predictors_tf, training_classes_tf,
         test_predictors_tf, test_classes_tf] = \
            Dataset.sample_data_regression(self)

        # Step 1) Create Neural Network MLP Regressor object with
        # personlized arguments
        mlpr = MLPRegressor(solver='adam', alpha=1e-5,
                            hidden_layer_sizes=(2,), random_state=1,
                            activation='logistic', max_iter=200)

        # Step 2) Use object created above to build the Neural Network
        # model and train on the dataset
        model = mlpr.fit(training_predictors_tf.as_matrix(),
                         training_classes_tf.as_matrix()[:, 0])

        # Step 3) Predict on the test data using model created above
        predicted = mlpr.predict(test_predictors_tf.as_matrix())

        # Step 4) Calculate the accuracy achieved
        print("\nRunning Neural Network Regressor Analysis")
        print("Mean squared error: ",
              np.mean((predicted - test_classes_tf.as_matrix()[:, 0]) ** 2))

        SSE = np.sum((mlpr.predict(test_predictors_tf.as_matrix()) -
                      test_classes_tf.as_matrix()[:, 0]) ** 2)
        SST = np.sum((np.mean(training_classes_tf.as_matrix()[:, 0]) -
                      test_classes_tf.as_matrix()[:, 0]) ** 2)
        R2 = 1.0 - 1.0 * SSE / SST
        RMSE = np.sqrt(SSE / len(test_classes_tf.as_matrix()[:, 0]))

        print("R squared value: ", R2)
        print("Root Mean Squared Error: ", RMSE)

        # Step 5) Plot the  outputs
        plt.figure(figsize=(10, 5))
        plt.plot(pd.Series(range(0, len(test_classes_tf.as_matrix()[:, 0]))),
                 test_classes_tf.as_matrix()[:, 0],
                 color='red', label='Test Data', linewidth=2)
        plt.plot(pd.Series(range(0, len(test_classes_tf.as_matrix()[:, 0]))),
                 predicted, color='blue',
                 linewidth=2, label="Predicted Data")
        plt.xlabel("Data Points")
        plt.ylabel("snp_close_scaled")
        plt.legend(loc='upper left')
        plt.show()

# eda_analyis=EDA('closing_data_pickle')
# eda_analyis.runAnalysis()

# linear_analysis=linear_model('closing_data_pickle')
# #linear_analysis.extractData()
# linear_analysis.linearRegression()

# logistic_analysis=ensemble('closing_data_pickle')
# logistic_analysis.randomForestClassifier()

neural_analysis=neural_network('closing_data_pickle')
# neural_analysis.mlpClassifier()
neural_analysis.mlpRegressor()











