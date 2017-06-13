from sklearn2pmml import PMMLPipeline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn2pmml import sklearn2pmml
from collections import Counter
from sklearn import metrics
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import ContinuousDomain
from sklearn.preprocessing import Imputer

cd=pd.read_pickle("closing_date_dynamic.pickle")
plt.figure(figsize=(15,5))
column_names=cd.columns.values+"_log_return"
log_return_data = pd.DataFrame()
for i in range(0,len(column_names)):
    log_return_data[column_names[i]]=np.log(cd.iloc[:,i]/cd.iloc[:,i].shift())

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
training_test_data.head()


predictors = training_test_data[training_test_data.columns[1:]]
classes = training_test_data[training_test_data.columns[:1]]

# 80% of the training data
training_set_size = int(len(training_test_data) * 0.8)
test_set_size = len(training_test_data) - training_set_size

training_predictors = predictors[:training_set_size]
training_classes = classes[:training_set_size]
test_predictors = predictors[training_set_size:]
test_classes = classes[training_set_size:]

plt.figure(figsize=(45,15))
training_predictors.describe()

logistic = LogisticRegression()
pipe=PMMLPipeline([
    ("mapper", DataFrameMapper([list(training_predictors.columns.values)])),
    ('logistic',logistic)])
model=pipe.fit(training_predictors.as_matrix(),training_classes.as_matrix()[:,0])
predicted=pipe.predict(test_predictors.as_matrix())
# Generating confusion matrix
print ("Confusion Matrix : \n",
       metrics.confusion_matrix(test_classes.as_matrix()[:,0], predicted))
sklearn2pmml(pipe,"LogisticRegression.pmml",with_repr=True,user_classpath=)
