from EDA import *
from Neural_network import *
from Ensemble import *
from Linear_model import *
import json
import quandl
from Logger import *

if __name__=='__main__':

    snp = quandl.get("YAHOO/INDEX_OEX", authtoken="NiLCy_frruoznRS-R7hS", start_date="2010-01-01",
                     end_date="2015-10-01")
    nyse = quandl.get("YAHOO/INDEX_NYA", authtoken="NiLCy_frruoznRS-R7hS", start_date="2010-01-01",
                      end_date="2015-10-01")
    djia = quandl.get("YAHOO/INDEX_DJI", authtoken="NiLCy_frruoznRS-R7hS", start_date="2010-01-01",
                      end_date="2015-10-01")
    nikkei = quandl.get("YAHOO/INDEX_N225", authtoken="NiLCy_frruoznRS-R7hS", start_date="2010-01-01",
                        end_date="2015-10-01")
    hangseng = quandl.get("YAHOO/INDEX_HSI", authtoken="NiLCy_frruoznRS-R7hS", start_date="2010-01-01",
                          end_date="2015-10-01")
    ftse = quandl.get("YAHOO/L_CPI", authtoken="NiLCy_frruoznRS-R7hS", start_date="2010-01-01", end_date="2015-10-01")
    dax = quandl.get("YAHOO/INDEX_GDAXI", authtoken="NiLCy_frruoznRS-R7hS", start_date="2010-01-01",
                     end_date="2015-10-01")
    aord = quandl.get("YAHOO/INDEX_AORD", authtoken="NiLCy_frruoznRS-R7hS", start_date="2010-01-01",
                      end_date="2015-10-01")

    # USE THE BELOW CODE TO MODIFY THE JSON FILE
    # config_json={"1":"snp_close","2":"nyse_close", "3":"djia_close", "4":"nikkei_close",
    #              "5":"hangseng_close", "6":"ftse_close", "7":"dax_close", "8":"aord_close"}
    # config_json_json=json.dumps(config_json)
    # with open('./config_json', 'w') as outfile:
    #     json.dump(config_json_json, outfile)

    # Opening json file
    with open('./config_json') as json_file:
        config_json = json.load(json_file)
    stock_market_dict = {"snp_close": snp, "nyse_close": nyse, "djia_close": djia, "nikkei_close": nikkei,
                         "hangseng_close": hangseng,
                         "ftse_close": ftse, "dax_close": dax, "aord_close": aord}

    config_json = json.loads(config_json)
    closing_date = pd.DataFrame()
    for key, values in config_json.items():
        closing_date[values] = stock_market_dict[values]['Close']

    closing_date = closing_date.fillna(method='ffill')
    closing_date.to_pickle('closing_date_dynamic_pickle')

    # eda_analyis=EDA('closing_data_pickle')
    # eda_analyis.runAnalysis()

    linear_analysis=linear_model('closing_data_dynamic_pickle')
    linear_analysis.linearRegression()
    linear_analysis.logistic_regression()
    #
    # ensemble_analysis=ensemble('closing_data_dynamic_pickle')
    # ensemble_analysis.randomForestClassifier()
    # ensemble_analysis.randomForestRegressor()
    #
    #
    # neural_analysis=neural_network('closing_data_dynamic_pickle')
    # neural_analysis.mlpClassifier()
    # neural_analysis.mlpRegressor()











