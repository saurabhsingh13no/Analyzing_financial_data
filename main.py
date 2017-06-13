from EDA import *
from Neural_network import *
from Ensemble import *
from Linear_model import *
import json
import quandl
from Logger import *

if __name__=='__main__':

    # Downloading the datasets based on config file
    with open("./config3.json") as f:
        config_json = f.readline()
        # config_json=json.dumps(config_json, ensure_ascii=False)
        json_object = json.loads(config_json)
        closing_date = pd.DataFrame()
        for i in range(0, len(json_object['ids'])):
            column = quandl.get(json_object['DB'] + "/" + json_object['tickers'][i],
                                start_date=json_object['start_date'],
                                end_date=json_object['end_date'],
                                authtoken=json_object['authtoken'])
            logger.debug("Downloaded {0} dataset successfully ".
                  format(json_object['ids'][i]))
            closing_date[json_object['ids'][i]] = column['Close']
        logger.debug("Merging datasets together")
        closing_date = closing_date.fillna(method='ffill')
        closing_date.to_pickle('closing_date_dyanamic.pickle')


    """Run Code to do Exploratory data anlysis"""
    logger.debug("Starting Exploratory Data Analysis")
    print ("Starting Exploratory Data Analysis")
    eda_analyis=EDA('closing_date_dyanamic.pickle')
    eda_analyis.runAnalysis()

    """Run code to do Linear data analysis"""
    logger.debug("Starting Linear Data Analysis")
    print ("Starting Linear Data Analysis")
    linear_analysis=linear_model('closing_date_dyanamic.pickle')
    linear_analysis.logistic_regression()
    linear_analysis.linearRegression()


    """Run code to do data analysis using RandomForest"""
    logger.debug("Performing Data Analysis using RandomForest")
    print ("Performing Data Analysis using RandomForest")
    ensemble_analysis=ensemble('closing_date_dyanamic.pickle')
    ensemble_analysis.randomForestClassifier()
    ensemble_analysis.randomForestRegressor()

    # """Run code to to data analysis using neural network"""
    logger.debug("Performing Data Analysis using Neural Network")
    print ("Performing Data Analysis using Neural Network")
    neural_analysis=neural_network('closing_date_dyanamic.pickle')
    neural_analysis.mlpClassifier()
    neural_analysis.mlpRegressor()











