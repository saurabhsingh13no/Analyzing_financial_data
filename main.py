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
                                start_date=json_object['start_date'], end_date=json_object['end_date'],
                                authtoken=json_object['authtoken'])
            closing_date[json_object['ids'][i]] = column['Close']
        closing_date = closing_date.fillna(method='ffill')
        closing_date.to_pickle('closing_date_dyanamic.pickle')



    # eda_analyis=EDA('closing_date_dyanamic.pickle')
    # eda_analyis.runAnalysis()

    linear_analysis=linear_model('closing_date_dyanamic.pickle')
    linear_analysis.linearRegression()
    # linear_analysis.logistic_regression()
    #
    # ensemble_analysis=ensemble('closing_date_dyanamic.pickle)
    # ensemble_analysis.randomForestClassifier()
    # ensemble_analysis.randomForestRegressor()
    #
    #
    # neural_analysis=neural_network('closing_date_dyanamic.pickle')
    # neural_analysis.mlpClassifier()
    # neural_analysis.mlpRegressor()











