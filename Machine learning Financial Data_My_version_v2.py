from EDA import *
from Neural_network import *
from Ensemble import *
from Linear_model import *

from Logger import *

if __name__=='__main__':

    # eda_analyis=EDA('closing_data_pickle')
    # eda_analyis.runAnalysis()

    linear_analysis=linear_model('closing_data_pickle')
    linear_analysis.linearRegression()
    linear_analysis.logistic_regression()

    ensemble_analysis=ensemble('closing_data_pickle')
    ensemble_analysis.randomForestClassifier()
    ensemble_analysis.randomForestRegressor()


    neural_analysis=neural_network('closing_data_pickle')
    neural_analysis.mlpClassifier()
    neural_analysis.mlpRegressor()











