from QuantUniversity import *
from Logger import *
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
        try:
            cd = pd.read_pickle("./"+filename)
        except Exception as e:
            print ("Error {}. Please check the name again.".format(e))
            logger.error("cannot open file %s %s",filename,e)

        # cd = cd.sort_index()
        # self.cd=cd
        return self.cd.copy()

    def getLogReturnData(self):
        return self.log_return_data

    def setLogReturnData(self,log_return_data):
        self.log_return_data=log_return_data

    def modify_closing_date(self):
        # Adding scaled columns in cd dataset
        cd=self.cd
        column_names = cd.columns.values + "_scaled"
        for i in range(0, len(column_names)):
            cd[column_names[i]] = cd.iloc[:, i] / max(cd.iloc[:, i])

        self.cd_pa = cd.copy()

    def extractData_helper(self):
        cd=Dataset.extractData(self)

        self.modify_closing_date()

        # Creating another dataframe log_return_data
        cd = pd.read_pickle(self.filename)
        column_names = cd.columns.values + "_log_return"
        log_return_data = pd.DataFrame()
        for i in range(0, len(column_names)):
            log_return_data[column_names[i]] = np.log(cd.iloc[:, i] / cd.iloc[:, i].shift())

        # Making these columns to do predictive data analysis
        new_column = column_names[0] + "_positive"
        log_return_data[new_column] = 0
        log_return_data.loc[log_return_data.iloc[:, 0] >= 0, new_column] = 1


        # Creating training_test_data dataframe to do logistic analysis
        column_names = log_return_data.columns.values
        correlated_columns = ["snp", "nyse", "djia"]

        def create_empty_training_test_dataframe():
            new_column = column_names[0] + "_positive"
            new_column_names = []
            for i in range(0, len(column_names)):
                if "snp" in column_names[i] or "nyse" in column_names[i] or "djia" in column_names[i]:
                    for k in range(1, 4):
                        new_column_names.append(column_names[i] + "_" + str(k))
                else:
                    for k in range(0, 3):
                        new_column_names.append(column_names[i] + "_" + str(k))
            new_column_names.insert(0, new_column)
            training_test_data = pd.DataFrame(
                columns=new_column_names)
            return training_test_data.copy()

        new_column = column_names[0] + "_positive"
        log_return_data[new_column] = 0
        log_return_data.loc[log_return_data.iloc[:, 0] >= 0, new_column] = 1
        training_test_data = create_empty_training_test_dataframe()
        column_names_training_test = training_test_data.columns.values
        # print (column_names_training_test)

        for i in range(8, len(log_return_data)):
            temp_dict = {}
            temp_dict[column_names_training_test[0]] = log_return_data[new_column].iloc[i]
            for j in range(0, len(column_names)):
                if correlated_columns[0] in column_names[j] or correlated_columns[1] in column_names[j] or \
                                correlated_columns[2] in column_names[j]:
                    for k in range(1, 4):
                        temp_dict[column_names[j] + "_" + str(k)] = log_return_data[column_names[j]].iloc[i - k]
                else:
                    for k in range(0, 3):
                        temp_dict[column_names[j] + "_" + str(k)] = log_return_data[column_names[j]].iloc[i - k]

            training_test_data = training_test_data.append(temp_dict, ignore_index=True)

        training_test_data = training_test_data.dropna()
        self.training_test_data=training_test_data.copy()

    def sample_data(self):
        training_test_data = self.training_test_data.copy()
        predictors = training_test_data[training_test_data.columns[1:]]
        classes = training_test_data[training_test_data.columns[:1]]

        # 80% of the training data
        training_set_size = int(len(training_test_data) * 0.8)
        test_set_size = len(training_test_data) - training_set_size

        training_predictors = predictors[:training_set_size]
        training_classes = classes[:training_set_size]
        test_predictors = predictors[training_set_size:]
        test_classes = classes[training_set_size:]

        plt.figure(figsize=(45, 15))
        training_predictors.describe()
        return training_predictors,training_classes,\
               test_predictors,test_classes

    def sample_data_regression(self):
        # Creating dataframe for Linear Regression Analysis
        self.extractData()
        self.modify_closing_date()
        closing_data = self.cd_pa.copy() # using
        print ("Closing data : ",closing_data)
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
