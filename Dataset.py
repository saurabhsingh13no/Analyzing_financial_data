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
