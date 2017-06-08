from Dataset import *
from Logger import *

class linear_model(Dataset):
    """Initialize instance variables"""

    def __init__(self, filename=None):
        Dataset.__init__(self, filename)
        Dataset.extractData(self)

    # Predicting the S&P index value using Linear Regression
    def linearRegression(self):

        logger.debug("Running Linear regression analysis")
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
        logger.debug("Mean squared error: %s",
              np.mean((predicted - test_classes_tf.as_matrix()[:, 0]) ** 2))

        SSE = np.sum((regr.predict(test_predictors_tf.as_matrix()) -
                      test_classes_tf.as_matrix()[:, 0]) ** 2)
        SST = np.sum((np.mean(training_classes_tf.as_matrix()[:, 0]) -
                      test_classes_tf.as_matrix()[:, 0]) ** 2)
        R2 = 1.0 - 1.0 * SSE / SST
        RMSE = np.sqrt(SSE / len(test_classes_tf.as_matrix()[:, 0]))

        print("R squared value: ", R2)
        logger.debug("R squared value: %s", R2)
        print("Root Mean Squared Error: ", RMSE)
        logger.debug("Root Mean Squared Error: %s", RMSE)

        # Step 5) Plot the  outputs
        try:
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
            plt.title("Stock index value v/s Data points")
            plt.show()
            logger.debug("Graph plotted - Stock index value v/s Data Points")
        except Exception as e:
            print("Failed to plot graph. Check logs")
            logger.error("%s error occured while plotting graph",e)




    def extractData(self):
        if len(Dataset.getLogReturnData(self))>0:
            return
        else:
            Dataset.extractData_helper(self)

    # Predicting the S&P trend(whether it will go up or down)
    # using Logistic Regression
    def logistic_regression(self):

        logger.debug("Running Logistic regression Model")

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
        logger.debug("Accuracy using Logistic Regression Model = %s", accuracy)

        # Or you can combine Step 3 and Step 4 into one using below state
        # print ("Accuracy using Logistic Regression Model = ",
        #        model.score(test_predictors_tf.as_matrix(),test_classes_tf.
        # as_matrix()[:,0]))

        # Generating the classfication Report
        print("Classification Report : \n\n",
              metrics.classification_report(test_classes_tf.as_matrix()[:, 0],
                                            predicted))

        # Generating confusion matrix
        print("Confusion Matrix : \n",
              metrics.confusion_matrix(test_classes_tf.as_matrix()[:, 0],
                                       predicted))

        # Verifying model using 10 fold cross validation
        scores = cross_val_score(LogisticRegression(),
                                 training_predictors_tf.as_matrix(),
                                 training_classes_tf.as_matrix()[:, 0],
                                 scoring='accuracy', cv=10)
        print("Mean accuracy validated using Cross Validation: ",
              scores.mean())
        logger.debug("Mean accuracy validated using Cross Validation: %s",
              scores.mean())
